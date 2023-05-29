from argparse import Namespace
from typing import List, Tuple, Dict, Any, Optional
import copy

import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Function
from numpy import typing as np_type
import numpy as np
from torch.autograd import Variable
from Source.helper import get_device
from avalanche.benchmarks import TCLExperience
from torch.utils.data import DataLoader


class MaskedOut_U0(Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, u0_units: List) -> torch.Tensor:
        indices = torch.tensor(u0_units, dtype=torch.long).to(get_device())
        new_x = x.clone()
        new_x[torch.arange(x.shape[0]).unsqueeze(1), u0_units] = 0
        ctx.save_for_backward(indices)
        return new_x
    
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any, Any, Any]:
        indices, = ctx.saved_tensors
        grad_output[torch.arange(grad_output.shape[0]).unsqueeze(1), indices] = 0
        return grad_output, None, None, None

    

class Network(nn.Module):
    def __init__(self, config_dict: Dict, input_size: int, output_size: int, args: Namespace) -> None:
        super(Network, self).__init__()
        self.config_dict = config_dict
        self.conv_layers_early = nn.ModuleList()
        self.conv_layers_late = nn.ModuleList()
        self.hidden_linear = nn.ModuleList()

        self.input_size = 1 if input_size == 784 else input_size
        self.output_size = output_size
        self.memory_mode = args.memory_mode
        self.args = args

        prev_layer_out = self.input_size
        # Early Conv Layers
        if self.config_dict["conv_early"] is None:
            self.conv_layers_early.append(nn.Identity())
        else:
            for ops, layer_config in zip(self.config_dict["conv_early"]["ops"], self.config_dict["conv_early"]["layers"]):
                if len(layer_config) == 3:
                    out_channels, kernel_size, stride = layer_config
                    self.conv_layers_early.append(SparseConv2d(prev_layer_out, out_channels, kernel_size, stride=stride, layer_name="conv_early"))
                else:
                    out_channels = layer_config[0]
                    self.conv_layers_early.append(SparseLinear(prev_layer_out, out_channels, layer_name="conv_early"))
                [self.conv_layers_early.append(op) for op in ops]
                prev_layer_out = out_channels

        # Late Conv Layers
        if self.config_dict["conv_late"] is None:
            self.conv_layers_late.append(nn.Identity())
        else:
            if len(self.config_dict["conv_late"]["layers"]) == 0:
                self.conv_layers_late.append(nn.Identity())
            else:
                for ops, layer_config in zip(self.config_dict["conv_late"]["ops"], self.config_dict["conv_late"]["layers"]):
                    if len(layer_config) == 3:
                        out_channels, kernel_size, stride = layer_config
                        self.conv_layers_late.append(SparseConv2d(prev_layer_out, out_channels, kernel_size, stride=stride, layer_name="conv_late"))
                    else:
                        out_channels = layer_config[0]
                        self.conv_layers_early.append(SparseLinear(prev_layer_out, out_channels, layer_name="conv_late"))
                    [self.conv_layers_late.append(op) for op in ops]
                    prev_layer_out = out_channels

    
        if self.config_dict["mlp_hidden_linear"] is None:
            self.hidden_linear.append(nn.Identity())
        else:
            prev_layer_out = self.config_dict["conv_late"]["conv2lin_size"] if self.config_dict["conv_late"] is not None else prev_layer_out 
            for ops, num_units in zip(self.config_dict["mlp_hidden_linear"]["ops"], self.config_dict["mlp_hidden_linear"]["layers"]):
                self.hidden_linear.append(SparseLinear(prev_layer_out, num_units, layer_name="linear"))
                [self.hidden_linear.append(op) for op in ops]
                prev_layer_out = num_units


        # Penultimate Layer
        self.penultimate_layer = SparseLinear(
                                           self.config_dict["mlp_hidden_linear"]["layers"][-1],
                                           self.config_dict["penultimate_layer_size"], layer_name="linear")

        self._initialize_weights()

        # Load Pretrained Weights if exists
        if args.pretrain_load_path:
            print("Loading Pretrained Weights")
            self.conv_layers_early.load_state_dict(torch.load(args.pretrain_load_path))
        # Continual Learning Attr
        self.classes_seen_so_far = []
        self.current_u0_units = [[]] + [list(range(param.shape[0])) for param in self.parameters() if len(param.shape) != 1]
        self.current_u1_units = [list(range(0)) for param in self.parameters() if len(param.shape) != 1] + [[]]


        
        unit_type_list = [([99]*self.input_size, "input")]
        for m in self.modules():
            if isinstance(m, (SparseLinear, SparseConv2d)):
                units = m.weight.data.shape[0]
                name = m.layer_name
                if args.pretrain_freeze and name == "conv_early":
                    unit_type_list.append(([100]*units, name))
                else:
                    unit_type_list.append(([0]*units, name)) # type: ignore

        self.unit_type_list = [(np.array(unit_types), name) for unit_types, name in unit_type_list]
        self.freeze_masks = None
        self.last_layer_activation = nn.ReLU()
        self.last_layer_bn = nn.Identity()
        if self.args.last_layer_bn:
            self.last_layer_bn = BatchNorm1Custom(self.config_dict["penultimate_layer_size"],
                                                  layer_index=len(self.current_u0_units) - 1)
            
        if args.pretrain_freeze:
            print("Max Promotion to Pretrained Units and removing grads")
            for layer_index, types in enumerate(self.unit_type_list):
               if types[0][0] == 100:
                   self.current_u0_units[layer_index] = []
            # set modules with name "conv_early" to be frozen
            for module in self.modules():
                if isinstance(module, (SparseLinear, SparseConv2d)) and module.layer_name == "conv_early":
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False
                   
            

        
    def l2_loss(self):
        reg_terms = []
        for module in self.modules():
            if isinstance(module, (SparseLinear, SparseConv2d)):
                reg_terms.append(torch.sum((module.weight * module.weight_mask)**2))
                reg_terms.append(torch.sum(module.bias**2))
            elif isinstance(module, (nn.Linear, nn.Conv2d)):
                reg_terms.append(torch.sum(module.weight ** 2))
        return torch.sum(torch.stack(reg_terms))

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (SparseLinear, SparseConv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def set_masks(self, weight_masks: List[torch.Tensor] , bias_masks: List[torch.Tensor]) -> None:
        i = 0
        for m in self.modules():
            if isinstance(m,(SparseLinear, SparseConv2d)):
                m.set_mask(weight_masks[i],bias_masks[i])
                i = i + 1

    def add_seen_classes(self, classes: List) -> None:
        new_classes = set(self.classes_seen_so_far)
        for cls in  classes:
            new_classes.add(cls)
        self.classes_seen_so_far = list(new_classes)


    def get_memory_representations(self, x: torch.Tensor) -> torch.Tensor:
        mlp = len(self.config_dict["conv_early"]["layers"][0]) == 1
        if mlp:
            x = x.view(x.shape[0], -1)
        if self.memory_mode  == "internal":
            # Feedforward conv early
            if self.config_dict["conv_early"] is not None:
                for layer in self.conv_layers_early:
                    x = layer(x)
        elif  self.memory_mode == "raw":
            x = x 
        else:
            raise Exception("Unknown option = {} for args.memory_modes".format(
                self.memory_mode
            ))
        return x
    
    def forward_memo(self, x: torch.Tensor):
        mlp = len(self.config_dict["conv_early"]["layers"][0]) == 1
        if mlp:
            x = x.view(x.shape[0], -1)

        if self.memory_mode  == "raw":
            # Feedforward conv early
            if self.config_dict["conv_early"] is not None:
                for layer in self.conv_layers_early:
                    x = layer(x)


        if self.config_dict["conv_late"] is not None:
            # Feedforward conv late
            for layer in self.conv_layers_late:
                x = layer(x)

        if not mlp:
            x = x.view(-1, self.config_dict["conv_late"]["conv2lin_size"])

        # Feedforward hidden linear
        for layer in self.hidden_linear:
            x = layer(x)

        # Feedforward penultimate layer and CLF layer
        x = self.penultimate_layer(x)
        x = self.last_layer_bn(x)
        x = self.last_layer_activation(x)
        x = MaskedOut_U0.apply(x, self.current_u0_units[-1])

        return x
    
    def forward_pretrain(self, x: torch.Tensor):
        mlp = len(self.config_dict["conv_early"]["layers"][0]) == 1
        if mlp:
            x = x.view(x.shape[0], -1)

        # Feedforward conv early
        if self.config_dict["conv_early"] is not None:
            for layer in self.conv_layers_early:
                x = layer(x)

        if self.config_dict["conv_late"] is not None:
            # Feedforward conv late
            for layer in self.conv_layers_late:
                x = layer(x)

        if not mlp:
            x = x.view(-1, self.config_dict["conv_late"]["conv2lin_size"])

        # Feedforward hidden linear
        for layer in self.hidden_linear:
            x = layer(x)

        # Feedforward penultimate layer and CLF layer
        x = self.penultimate_layer(x)
        x = self.last_layer_bn(x)
        x = self.last_layer_activation(x)
        return x


    def forward(self, x: torch.Tensor):
        mlp = len(self.config_dict["conv_early"]["layers"][0]) == 1
        if mlp:
            x = x.view(x.shape[0], -1)

        # Feedforward conv early
        if self.config_dict["conv_early"] is not None:
            for layer in self.conv_layers_early:
                x = layer(x)

        if self.config_dict["conv_late"] is not None:
            # Feedforward conv late
            for layer in self.conv_layers_late:
                x = layer(x)

        if not mlp:
            x = x.view(-1, self.config_dict["conv_late"]["conv2lin_size"])

        # Feedforward hidden linear
        for layer in self.hidden_linear:
            x = layer(x)

        # Feedforward penultimate layer and CLF layer
        x = self.penultimate_layer(x)
        x = self.last_layer_bn(x)
        x = self.last_layer_activation(x)
        x = MaskedOut_U0.apply(x, self.current_u0_units[-1])

        return x
    
    def get_weight_bias_masks_numpy(self) -> List[Tuple[np_type.NDArray[np.double], np_type.NDArray[np.double]]]:
        weights = []
        for module in self.modules():
            if isinstance(module, SparseLinear) or isinstance(module, SparseConv2d):
                weight_mask, bias_mask = module.get_mask()  # type: ignore
                weights.append((copy.deepcopy(weight_mask).cpu().numpy(),
                                copy.deepcopy(bias_mask).cpu().numpy())) # type: ignore
        return weights
    
    def get_frozen_units(self) -> List:
        frozen_units = []
        for unit_layer, name in self.unit_type_list:
            if "early" in name:
                frozen_units.append((unit_layer > 1).nonzero()[0])
            else:
                frozen_units.append((unit_layer > self.args.stm_num_tasks + 1).nonzero()[0])
        return frozen_units
    
    def get_last_frozen(self):
        return (self.unit_type_list[-1][0] > self.args.stm_num_tasks + 1).nonzero()[0]
    
    def create_drop_masks(self) -> List:
        drop_masks = []
        connectivity_masks = [w for w, _ in self.get_weight_bias_masks_numpy()]
        all_u0_indices = [list((u == 0).nonzero()[0]) for u, _ in self.unit_type_list]
        all_non_u0_indices = [list((u != 0).nonzero()[0]) for u, _ in self.unit_type_list]

        for i, (current_layer_u0_indices, next_layer_u1_indices) in enumerate(zip(all_u0_indices[:-1], all_non_u0_indices[1:])):
            drop_mask = np.zeros(connectivity_masks[i].shape, dtype=np.intc)
            if current_layer_u0_indices:
                #Conv2Linear
                if len(connectivity_masks[i].shape) == 2 and len(connectivity_masks[i-1].shape) == 4:
                    for u0_index in current_layer_u0_indices:
                        start = u0_index*self.config_dict["conv_late"]["conv2lin_mapping_size"]
                        end = (u0_index+1)*self.config_dict["conv_late"]["conv2lin_mapping_size"]
                        drop_mask[next_layer_u1_indices, start:end] = 1
                else:
                    drop_mask[np.ix_(next_layer_u1_indices, current_layer_u0_indices)] = 1
            drop_masks.append(drop_mask * connectivity_masks[i])
        return drop_masks
    
    def get_memory_mask_units(self, unit_type) -> np_type.NDArray:
        return (self.unit_type_list[-1][0] >= unit_type).nonzero()[0]


    
    def promote_units(self) -> None:
        unit_type_list = []
        for unit_layer, name in self.unit_type_list[1:]:
            unit_layer[unit_layer != 0] = unit_layer[unit_layer != 0] + 1
            unit_type_list.append((unit_layer, name))

        unit_type_list = [(self.unit_type_list[0][0], self.unit_type_list[0][1])] + unit_type_list
        self.unit_type_list = unit_type_list

        self.set_current_u0_and_u1_units(self.current_u0_units, [[] for _ in self.current_u1_units]) 
    
    def set_current_u0_and_u1_units(self, u0: List, u1: List) -> None:
        self.current_u1_units = u1
        self.current_u0_units = u0
        for index, (layer_u0_units, layer_u1_units) in enumerate(zip(u0, u1)):
            if layer_u0_units:
                if any(np.array(self.unit_type_list[index][0][layer_u0_units]) > 1):
                    raise Exception("Cannot set unit to smaller timestep")
                else:
                    self.unit_type_list[index][0][layer_u0_units] = 0
            if layer_u1_units:
                if any(np.array(self.unit_type_list[index][0][layer_u1_units]) > 1):
                    raise Exception("Cannot set unit to smaller timestep")
                else:
                    self.unit_type_list[index][0][layer_u1_units] = 1

    def re_initialize_u0(self):
        i = 0
        for m in self.modules():
            if isinstance(m, SparseLinear) or isinstance(m, SparseConv2d): 
                m.weight.data[self.current_u0_units[i+1],:] = nn.init.kaiming_normal_(m.weight.data[self.current_u0_units[i+1],:],
                                                                                    mode='fan_out', nonlinearity='relu')
                m.bias.data[self.current_u0_units[i+1]] = nn.init.constant_(m.bias.data[self.current_u0_units[i+1]], 0.0) # type: ignore
                i += 1
    
    def update_freeze_masks(self) -> None:
        weights = self.get_weight_bias_masks_numpy()
        freeze_masks = []
        list_stable_units = self.get_frozen_units()
        for i, target_stable in enumerate(list_stable_units[1:]):
            target_stable =  np.array(target_stable, dtype=np.int32)
            mask_w = np.zeros(weights[i][0].shape)
            mask_b = np.zeros(weights[i][1].shape)
            if len(target_stable) != 0:
                mask_w[target_stable, :] = 1
                mask_b[target_stable] = 1
            freeze_masks.append((mask_w * weights[i][0], mask_b))

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        freeze_masks = [(torch.tensor(w).to(torch.bool).to(device),
                         torch.tensor(b).to(torch.bool).to(device))
                         for w, b in freeze_masks]
        self.freeze_masks = freeze_masks

        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                layer_index = module.layer_index
                frozen_units = list_stable_units[layer_index] # type: ignore
                frozen_units_binary = torch.zeros(module.num_features)
                frozen_units_binary[frozen_units] = 1
                module.freeze_units(frozen_units_binary) # type: ignore

    def compute_weight_sparsity(self):
        parameters = 0
        ones = 0
        for module in self.modules():
            if isinstance(module,SparseLinear) or isinstance(module, SparseConv2d):
                shape = module.weight.data.shape
                parameters += torch.prod(torch.tensor(shape))
                w_mask, _ = copy.deepcopy(module.get_mask())
                ones += torch.count_nonzero(w_mask)
        return float((parameters - ones) / parameters) * 100
    
    def select_u0_u1_units(self, train_task: TCLExperience, stable_selection_perc: float, episode_index: int) -> Tuple[List[List[int]],  List[List[int]]]:
        loader = DataLoader(train_task.dataset, batch_size = 1024,  shuffle=False)
        x, _, _ = next(iter(loader))
        x = x.to(get_device())
        mlp = len(self.config_dict["conv_early"]["layers"][0]) == 1
        if mlp:
            x = x.view(x.shape[0], -1)
        u0_units_all_layers = []
        u1_units_all_layers = []
        layer_index = 1 
        self.eval()
        # Feedforward conv early
        if self.config_dict["conv_early"] is not None:
            for layer in self.conv_layers_early:
                x = layer(x)
                if isinstance(layer, (torch.nn.ReLU, torch.nn.LeakyReLU)):
                    activation = copy.deepcopy(x.detach()) / 1024.0
                    selection_perc = max(stable_selection_perc, self.args.min_activation_perc)
                    #print("layer-{}: selection {}%".format(layer_index, selection_perc))
                    picked_u0, picked_u1 = pick_u0_and_u1_neurons(self, activation, layer_index, selection_perc)
                    u0_units_all_layers.append(picked_u0)
                    u1_units_all_layers.append(picked_u1)
                    layer_index = layer_index + 1


        # Feedforward conv late
        if self.config_dict["conv_late"] is not None:
            for layer in self.conv_layers_late:
                x = layer(x)
                if isinstance(layer, (torch.nn.ReLU, torch.nn.LeakyReLU)):
                    activation = copy.deepcopy(x.detach()) / 1024.0
                    selection_perc = max(stable_selection_perc, self.args.min_activation_perc)
                    #print("layer-{}: selection {}%".format(layer_index, selection_perc))
                    picked_u0, picked_u1 = pick_u0_and_u1_neurons(self, activation, layer_index, selection_perc)
                    u0_units_all_layers.append(picked_u0)
                    u1_units_all_layers.append(picked_u1)
                    layer_index = layer_index + 1


        if not mlp:
            x = x.view(-1, self.config_dict["conv_late"]["conv2lin_size"])


        # Feedforward hidden linear
        for layer in self.hidden_linear:
            x = layer(x)
            if isinstance(layer, (torch.nn.ReLU, torch.nn.LeakyReLU)):
                activation = copy.deepcopy(x.detach()) / 1024.0
                selection_perc = max(stable_selection_perc, self.args.min_activation_perc)
                #print("layer-{}: selection {}%".format(layer_index, selection_perc))
                picked_u0, picked_u1 = pick_u0_and_u1_neurons(self, activation, layer_index, selection_perc)
                u0_units_all_layers.append(picked_u0)
                u1_units_all_layers.append(picked_u1)
                layer_index = layer_index + 1

        # Forward penultimate
        x = self.penultimate_layer(x)
        x = self.last_layer_bn(x)
        x = self.last_layer_activation(x)
        activation = copy.deepcopy(x.detach()) / 1024.0
        selection_perc = max(stable_selection_perc, self.args.min_activation_perc)
        print("layer-{}: selection {}%".format(layer_index, selection_perc))
        picked_u0, picked_u1 = pick_u0_and_u1_neurons(self, activation, layer_index, selection_perc)
        u0_units_all_layers.append(picked_u0)
        u1_units_all_layers.append(picked_u1)
        self.train()

        # Add input units
        u0_units_all_layers = [[]] + u0_units_all_layers
        u1_units_all_layers = [[]] + u1_units_all_layers
        return u0_units_all_layers, u1_units_all_layers



def pick_u0_and_u1_neurons(network: Network, activation: torch.Tensor, layer_index: int,  stable_selection_perc: float) -> Tuple[List, List]:
    activation = torch.sum(activation, axis = (0, 2, 3)) if len(activation.shape) != 2 else torch.sum(activation, axis = 0) # type: ignore
    u0_and_u1 = list(set(network.current_u0_units[layer_index]).union(network.current_u1_units[layer_index]))
    picked_units = pick_top_neurons(activation, stable_selection_perc)
    picked_u1 = list(set(u0_and_u1).intersection(picked_units))
    picked_u0 = list(set(u0_and_u1).difference(picked_units))
    return picked_u0, picked_u1


def pick_top_neurons(average_layer_activation: torch.Tensor, stable_selection_perc: float) -> List[int]:
    total = sum(average_layer_activation)
    accumulate = 0
    indices = []
    sort_indices = torch.argsort(-average_layer_activation)
    for index in sort_indices:
        index = int(index)
        accumulate = accumulate + average_layer_activation[index]
        indices.append(index)
        if accumulate >= total * stable_selection_perc / 100:
            break
    return indices


class SparseLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias_flag=True, layer_name = ""):
        super(SparseLinear, self).__init__(in_features, out_features, True)
        self.bias_flag = bias_flag
        self.layer_name = layer_name
        
    def set_mask(self, weight_mask: torch.Tensor, bias_mask: torch.Tensor) -> None:
        self.weight_mask = to_var(weight_mask, requires_grad=False)
        self.weight.data = self.weight.data * self.weight_mask.data
        
        self.bias_mask = to_var(bias_mask, requires_grad=False)
        self.bias.data = self.bias.data * self.bias_mask.data

    def get_mask(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.weight_mask, self.bias_mask

    def forward(self, x) -> torch.Tensor:
        weight = self.weight * self.weight_mask
        bias = self.bias * self.bias_mask
        return F.linear(x, weight, bias if self.bias_flag else None)
    
class SparseConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, layer_name = ""):
        super(SparseConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bias_flag = bias
        self.layer_name = layer_name

    def set_mask(self, weight_mask: torch.Tensor, bias_mask: torch.Tensor) -> None:
        self.weight_mask = to_var(weight_mask, requires_grad=False)
        self.weight.data = self.weight.data * self.weight_mask.data
        if self.bias_flag:
            self.bias_mask = to_var(bias_mask, requires_grad=False)
            self.bias.data = self.bias.data * self.bias_mask.data # type: ignore

    def get_mask(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.weight_mask, self.bias_mask if self.bias_flag else None

    def forward(self, x) -> torch.Tensor:
        weight = self.weight * self.weight_mask
        bias = self.bias * self.bias_mask  if self.bias_flag else self.bias
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
       

def to_var(x, requires_grad = False, volatile = False) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().to(get_device())
    else:
        x = torch.tensor(x).to(get_device())
    return Variable(x, requires_grad = requires_grad, volatile = volatile)


def random_prune(network: Network, pruning_perc: float, skip_first_conv = True) -> Network:
    pruning_perc = pruning_perc / 100.0
    weight_masks = []
    bias_masks = []
    first_conv_flag = skip_first_conv
    for module in network.modules():
        layer_pruning_perc = pruning_perc
        if isinstance(module, (SparseLinear, SparseConv2d)) and network.args.pretrain_load_path != "" and module.layer_name == "conv_early":
            print("Skipping pruning for conv_early layer")
            layer_pruning_perc = 0.0

        if isinstance(module, SparseLinear):
            weight_masks.append(torch.from_numpy(np.random.choice([0, 1], module.weight.shape,
                                                                  p =  [layer_pruning_perc, 1 - layer_pruning_perc])))
            # We do not prune biases
            bias_masks.append(torch.from_numpy(np.random.choice([0, 1], module.bias.shape, p =  [0, 1])))
        #Channel wise pruning Conv Layer
        elif isinstance(module, SparseConv2d):
           connectivity_mask = torch.from_numpy(np.random.choice([0, 1],
                                                (module.weight.shape[0],  module.weight.shape[1]),
                                                p =  [0, 1] if first_conv_flag else [layer_pruning_perc, 1 - layer_pruning_perc]))
           first_conv_flag = False
           in_range, out_range = range(module.weight.shape[1]), range(module.weight.shape[0])
           kernel_shape = (module.weight.shape[2], module.weight.shape[3])
           filter_masks = [[np.ones(kernel_shape) if connectivity_mask[out_index, in_index] else np.zeros(kernel_shape)
                            for in_index in in_range]
                            for out_index in out_range]
           weight_masks.append(torch.from_numpy(np.array(filter_masks)).to(torch.float32))
           
           #do not prune biases
           bias_mask = torch.from_numpy(np.random.choice([0, 1], module.bias.shape, p =  [0, 1])).to(torch.float32)  # type: ignore
           bias_masks.append(bias_mask)
    network.set_masks(weight_masks, bias_masks)
    network.to(get_device())
    return network


class BatchNorm1Custom(torch.nn.BatchNorm1d):
    def __init__(self, num_features, layer_index, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm1Custom, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.frozen_units = None
        self.layer_index = layer_index
        self.running_mean_frozen = None
        self.running_var_frozen = None

    def freeze_units(self, frozen_units):
        frozen_units = frozen_units.bool().to(get_device())
        if self.frozen_units is None:
            self.frozen_units = frozen_units
            self.running_mean_frozen = self.running_mean.data.clone()  # type: ignore 
            self.running_var_frozen = self.running_var.data.clone()# type: ignore 
        else:
            new_frozen_units = torch.logical_xor(self.frozen_units, frozen_units)
            self.running_mean_frozen = torch.where(new_frozen_units, self.running_mean.data, self.running_mean_frozen)  # type: ignore 
            self.running_var_frozen = torch.where(new_frozen_units, self.running_var.data, self.running_var_frozen)  # type: ignore 
            self.frozen_units = frozen_units


    def forward(self, input):
        if self.frozen_units  is not  None:
            # Replace the frozen dimensions in self.running_mean and running_var
            self.running_mean.data = torch.where(self.frozen_units,self.running_mean_frozen, self.running_mean.data) # type: ignore 
            self.running_var.data = torch.where(self.frozen_units, self.running_var_frozen, self.running_var.data) # type: ignore 

        # Call the forward method of the parent class
        return super(BatchNorm1Custom, self).forward(input)