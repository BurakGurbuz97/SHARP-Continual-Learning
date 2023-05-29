from typing import Tuple
from argparse import Namespace


from torch.utils.data import DataLoader, Subset
from avalanche.benchmarks import TCLExperience
from typing import List
import torch


def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu' 


class PhaseVars():
    def __init__(self, pruned_network: ..., neocortex: ...):
        self.phase_index = 1
        self.previous_model = pruned_network
        self.best_phase_acc = 0.0
        self.prev_phase_stable_and_candidate_stable_units = []
        self.stable_and_candidate_stable_units = pruned_network.list_stable_units
        self.previous_neocortex = neocortex

def get_data_loaders(args: Namespace,
                     train_task: TCLExperience, val_task: TCLExperience,
                     test_task: TCLExperience, task_index: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if args.gpu_prefetch:
        return _get_data_loaders_gpu(args, train_task, val_task, test_task, task_index)
    bs =  args.batch_size
    train_loader = DataLoader(train_task.dataset, batch_size = bs,  shuffle=True, drop_last=True)
    val_loader =  DataLoader(val_task.dataset, batch_size = bs,  shuffle=True, drop_last=True)
    test_loader = DataLoader(test_task.dataset, batch_size = bs,  shuffle=True)
    return (train_loader, val_loader, test_loader)

def _get_data_loaders_gpu(args: Namespace, train_task: TCLExperience,
                          val_task: TCLExperience, test_task: TCLExperience, task_index: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    loaders = []
    bs = args.batch_size
    for task_dataset, skip_last in [(train_task.dataset, True), (val_task.dataset, True), (test_task.dataset, False)]:
        x, y, t = zip(*[(x, y, t) for x, y, t in task_dataset])
        x_tensor = torch.stack(x).clone().detach().to("cuda")
        y_tensor, t_tensor = torch.tensor(y).to("cuda"), torch.tensor(t).to("cuda")
        tensor_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor, t_tensor) # type: ignore
        loaders.append(DataLoader(tensor_dataset, batch_size = bs,  shuffle=True, drop_last=skip_last))
    return tuple(loaders)


def get_n_samples_per_class(dataset: TCLExperience, n: int) -> List:
    indices = {i: [] for i in dataset.classes_in_this_experience}
    for i, (_, y, _) in enumerate(dataset.dataset):
        indices[y].append(i)

    subsets = []
    for i in dataset.classes_in_this_experience:
        dataloader = DataLoader(Subset(dataset.dataset, indices[i][:n]), batch_size=n)
        samples, _, _  = next(iter(dataloader))
        subsets.append((samples, i))
    return subsets