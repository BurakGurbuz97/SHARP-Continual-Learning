from argparse import Namespace
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from torch.utils.data import DataLoader
import numpy as np

from  Source.architecture import Network, get_device, SparseConv2d, SparseLinear
from  Source.neocortex import NeocortexKNN
from  Source.memory import MemoryBuffer



def filter_classes(data, labels, min_samples_per_class=2):
    """
    Filters data and labels tensors by removing classes that have less than min_samples_per_class samples in the batch.

    :param data: Tensor of shape (batch_size, representation_size)
    :param labels: Tensor of shape (batch_size)
    :param min_samples_per_class: Minimum number of samples per class required for inclusion
    :return: Filtered data and labels tensors
    """
    unique_labels, counts = torch.unique(labels, return_counts=True)
    valid_classes = unique_labels[counts >= min_samples_per_class]

    mask = torch.tensor([label in valid_classes for label in labels])

    data_filtered = data[mask]
    labels_filtered = labels[mask]

    return data_filtered, labels_filtered


def reset_frozen_gradients(network: Network) -> Network:
    mask_index = 0
    for module in network.modules():
        if isinstance(module, SparseLinear) or  isinstance(module, SparseConv2d):
            if module.weight.requires_grad:
                module.weight.grad[network.freeze_masks[mask_index][0]] = 0  # type: ignore
                if module.bias_flag:
                    module.bias.grad[network.freeze_masks[mask_index][1]] = 0    # type: ignore
            mask_index = mask_index + 1

        
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            if module.frozen_units is not None and module.affine:
                module.weight.grad = torch.where(module.frozen_units, torch.zeros_like(module.weight.grad), module.weight.grad)  # type: ignore
                module.bias.grad = torch.where(module.frozen_units, torch.zeros_like(module.bias.grad), module.bias.grad)  # type: ignore
    return network

def test(network: Network, neocortex: NeocortexKNN, data_loader: DataLoader, episode_id = None, return_preds = False) -> float:
    network.eval()
    predictions = []
    ground_truths = []
    with torch.no_grad():
        for data, target, _ in data_loader:
            data = data.to(get_device())
            target = target.to(get_device())
            output = network.forward(data)
            preds =  neocortex.predict(output.detach(), episode_id)
            predictions.extend(preds)
            ground_truths.extend(target)

    predictions = np.array([int(p) for p in predictions])
    ground_truths = np.array([int(gt) for gt in ground_truths])
    acc1 = sum(predictions == ground_truths) / len(predictions)
    if return_preds:
        return acc1, predictions, ground_truths
    else:
        return acc1

def task_training_supcon(network: Network, epochs: int, loss: nn.Module, optimizer: ..., train_loader: DataLoader,
                  val_loader: DataLoader, args: Namespace, memory_and_range: Optional[Tuple[MemoryBuffer, List]] = None) -> Network:
    for _ in range(epochs):
        network.train()
        epoch_l2_loss = []
        epoch_supcon_loss = []
        epoch_output_loss = []
        for data, target, _ in train_loader:
            if sum(target.unique(return_counts=True)[1] == 1) != 0:
                data, target = filter_classes(data, target)
            optimizer.zero_grad()

            stream_output = network.forward(data)

            if memory_and_range is not None:
                memory, task_range = memory_and_range
                memo_samples, memo_labels = memory.sample_n(n = args.batch_size_memory, tasks = task_range)
                memo_samples = torch.tensor(memo_samples, dtype= torch.float32).to(get_device())
                memo_labels = torch.tensor(memo_labels).to(get_device())
                memo_output = network.forward_memo(memo_samples)
                all_output = torch.vstack((stream_output, memo_output))
                all_targets = torch.concat((target, memo_labels))
            else:
                all_targets = target
                all_output = stream_output
            
            supcon_loss = loss(all_output, all_targets.long())
            reg_loss = (args.weight_decay * network.l2_loss())
            epoch_supcon_loss.append(supcon_loss)
            epoch_l2_loss.append(reg_loss)
            batch_loss = reg_loss  + supcon_loss
            batch_loss.backward()
            if network.freeze_masks:
                network = reset_frozen_gradients(network)
            optimizer.step()
        print("Average training loss input: {}".format(sum(epoch_supcon_loss) / len(epoch_supcon_loss)))
        print("Average l2 loss: {}".format(sum(epoch_l2_loss) / len(epoch_l2_loss)))

    return network
