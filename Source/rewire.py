from argparse import Namespace
import math
import torch
import numpy as np
from numpy import typing as np_type
from typing import Callable, List, Tuple
from torch.utils.data import DataLoader

from Source.architecture import Network, get_device, SparseLinear, SparseConv2d


def get_tau_schedule(args: Namespace) -> Callable[..., float]:

    class funcs():
        def __init__(self, k):
            self.cosine_anneling = lambda t: 0.5 * (1 + math.cos(t * math.pi / k))
            self.linear = lambda t: 1 - k*t
            self.exp_decay = lambda t: (t + 1)**(-k)

    return getattr(funcs(args.tau_param), "cosine_anneling")

def get_connectivity_masks(network: Network) -> List[torch.Tensor]:
    # Get Connectivity masks
    connection_masks = []
    for w, _ in network.get_weight_bias_masks_numpy():
        if len(w.shape) != 2:
            conn_size = w.shape[2] * w.shape[3]
            connection_masks.append(w.sum(axis = (2, 3)) / conn_size)
        else:
            connection_masks.append(w)
    return connection_masks



def _drop_connections_with_mask(network: Network, drop_masks: List) -> Tuple[Network, List[int]]:
    mask_index, num_drops = 0, []
    for module in network.modules():
        if isinstance(module, SparseLinear) or isinstance(module, SparseConv2d):
            weight_mask, bias_mask = module.get_mask()
            weight_mask[torch.tensor(drop_masks[mask_index], dtype= torch.bool)] = 0
            num_drops.append(int(np.sum(drop_masks[mask_index])))
            module.set_mask(weight_mask, bias_mask) # type: ignore
            mask_index += 1
    return network, num_drops


def drop_connections(network: Network) -> Tuple[Network, List[int]]:
    drop_masks = network.create_drop_masks()
    return  _drop_connections_with_mask(network, drop_masks)


class Growth():
    def __init__(self, train_loader: DataLoader, network: Network):
        self.stable_units = network.get_frozen_units()
        self.u1_units = network.current_u1_units
        self.u0_units = network.current_u0_units
        self.train_loader = train_loader
        try:
            self.conv2lin_mapping_size = network.config_dict["conv_late"]["conv2lin_mapping_size"]
        except:
            self.conv2lin_mapping_size = None

    def grow(self, network: Network, connection_quota: List[int]) -> Tuple[Network, List[int]]:
        # No quota
        if sum(connection_quota) == 0:
            return network, connection_quota
        
        return self._random_grow(network, connection_quota)
    
    # ------- Helper methods ------- # 
    def _grow_connections(self, network: Network, possible_connections: List, connection_quota: List[int]) -> Tuple[Network, List[int]]:
        layer_index = 0
        weight_init = lambda size: torch.zeros(size).to(get_device())
        remainder_connections = []
        for module in network.modules():
            if isinstance(module, SparseLinear) or isinstance(module, SparseConv2d):
                if connection_quota[layer_index] == 0:
                    remainder_connections.append(0)
                    layer_index = layer_index + 1
                    continue
                weight_mask, bias_mask = module.get_mask()
                # Conv layer
                if len(possible_connections[layer_index].shape) == 4:
                    grow_indices = np.nonzero(np.sum(possible_connections[layer_index], axis = (2 , 3)))
                # Linear layer
                else:
                    grow_indices = np.nonzero(possible_connections[layer_index])
                
                conn_shape = (possible_connections[layer_index].shape[2], possible_connections[layer_index].shape[3]) if len(possible_connections[layer_index].shape) == 4 else (1,)
                conn_size = (possible_connections[layer_index].shape[2] * possible_connections[layer_index].shape[3]) if len(possible_connections[layer_index].shape) == 4 else 1

                # There are connections that we can grow
                if len(grow_indices[0]) != 0:
                    # We can partial accommodate grow request (we will have remainder connections)
                    if len(grow_indices[0])*conn_size <= connection_quota[layer_index]:
                        weight_mask[grow_indices] = torch.ones(weight_mask[grow_indices].shape,
                                                               dtype=weight_mask[grow_indices].dtype).to(get_device())
                        module.weight.data[grow_indices] = torch.zeros(module.weight.data[grow_indices].shape,
                                                                       dtype=module.weight.data.dtype).to(get_device())
                        remainder_connections.append(connection_quota[layer_index] - len(grow_indices[0])*conn_size)
                    else:
                        selection = np.random.choice(len(grow_indices[0]),
                                    size = int(connection_quota[layer_index]/ conn_size), replace = False)
                        tgt_selection = torch.tensor(grow_indices[0][selection]).to(get_device())
                        src_selection = torch.tensor(grow_indices[1][selection]).to(get_device())
                        weight_mask[tgt_selection, src_selection] = torch.squeeze(torch.ones((len(tgt_selection), *conn_shape), dtype = weight_mask.dtype)).to(get_device())
                        module.weight.data[tgt_selection, src_selection] = torch.squeeze(weight_init((len(tgt_selection), *conn_shape)))
                        remainder_connections.append(0)
                else:
                    remainder_connections.append(connection_quota[layer_index])
                module.set_mask(weight_mask, bias_mask)  # type: ignore
                layer_index += 1
        return network, remainder_connections
    
        # ------- Connection growth methods ------- #
    def _random_grow(self, network: Network, connection_quota: List[int]) -> Tuple[Network, List[int]]:
        connectivity_masks = [w for w, _ in network.get_weight_bias_masks_numpy()]

        def get_possible_conns(source_units: List, target_units: List, conv2lin_mapping_size):
            possible_connections = []

            for layer_index, (sources, targets) in enumerate(zip(source_units[:-1], target_units[1:])):
                if len(connectivity_masks[layer_index].shape) == 4:
                    conn_shape = (connectivity_masks[layer_index].shape[2], connectivity_masks[layer_index].shape[3])
                else:
                    conn_shape = (1, )
                
                # No target that we can grow
                if len(targets) == 0:
                    pos_conn = np.zeros(connectivity_masks[layer_index].shape)
                    possible_connections.append(pos_conn)
                    continue

                conn_type_1 = np.ones(conn_shape)
                conn_type_0 = np.zeros(conn_shape)
                pos_conn = np.zeros(connectivity_masks[layer_index].shape)

                #Conv2Linear
                if len(connectivity_masks[layer_index].shape) == 2 and len(connectivity_masks[layer_index-1].shape) == 4:
                    for conv_index in sources:
                        start = conv_index*conv2lin_mapping_size
                        end = (conv_index+1)*conv2lin_mapping_size
                        pos_conn[targets, start:end] = 1
                else:
                    pos_conn[np.ix_(targets, sources)] = conn_type_1
                # Remove already existing weights from pos_conn
                if len(connectivity_masks[layer_index].shape) == 4:
                    pos_conn[np.all(connectivity_masks[layer_index][:,:] == conn_type_1, axis = (2, 3))]  = conn_type_0
                else:
                    pos_conn[connectivity_masks[layer_index] != 0] = 0

                possible_connections.append(pos_conn)
            return possible_connections
        
        pos_conn_list = []
        sources = [list((unit_types >= 0).nonzero()[0]) for unit_types, _ in network.unit_type_list]
        targets = [list((unit_types == 0).nonzero()[0]) for unit_types, _ in network.unit_type_list]
        pos_conn = get_possible_conns(sources, targets, self.conv2lin_mapping_size)
        pos_conn_list.append(pos_conn)

        possible_connections = []
        for layer_index in range(len(connectivity_masks)):
            pos_conn = np.zeros(connectivity_masks[layer_index].shape)
            for possible in pos_conn_list:
                if layer_index < len(possible):
                    pos_conn = pos_conn + possible[layer_index]

            possible_connections.append(np.array(pos_conn))

        return self._grow_connections(network, possible_connections, connection_quota)
    