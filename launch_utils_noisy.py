import argparse
import os
import pickle
import shutil
import random
from typing import Tuple, Dict

from avalanche.benchmarks import GenericCLScenario
from avalanche.benchmarks.datasets import CIFAR10, CIFAR100, FashionMNIST, MNIST, EMNIST
from avalanche.benchmarks.generators import benchmark_with_validation_stream, tensors_benchmark
from torch.utils.data.dataset import Dataset
import numpy as np
import torch
from torch.backends import cudnn 
from torchvision import transforms

import model_config


DATASET_PATH =  os.path.join(os.path.abspath('..'), 'datasets')
LOG_PATH = os.path.join(os.path.abspath('.'), 'Logs')

def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    cudnn.benchmark = False
    cudnn.deterministic = True

def get_argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Experiment')
    # Logging params
    parser.add_argument('--experiment_name', type=str, default = 'Test')
    parser.add_argument("--gpu_prefetch", type=int, default = 1)

    # LTM Memory Params
    parser.add_argument('--ltm_per_class', type=int, default = 25)
    parser.add_argument('--ltm_k_nearest', type=int, default = 5)

    # STM Memory Params
    parser.add_argument('--stm_per_class', type=int, default = 25)
    parser.add_argument('--stm_num_tasks', type=int, default = 1) # 1 (current) + 2 (most recent)
    parser.add_argument('--memory_mode', type=str, default="internal", choices=["raw", "internal"])

    # Dataset params
    parser.add_argument('--dataset', type=str, default = 'SplitMNIST')
    parser.add_argument('--number_of_tasks', type=int, default = 13)

    # Architectural params
    parser.add_argument('--model', type=str, default = 'MNISTLIKE')
    parser.add_argument('--prune_perc', type=float, default=60.0)

    # Learning params
    parser.add_argument('--seed', type=int,  default=0)
    parser.add_argument('--deterministic', type=int,  default=1)

    # Anything under torch.optim works. e.g., 'SGD' and 'Adam'
    parser.add_argument('--optimizer', type=str, default = 'Adadelta')
    parser.add_argument('--sgd_momentum', type=float, default = 0.90)
    parser.add_argument('--learning_rate', type=float, default = 1.0)
    parser.add_argument('--batch_size', type=int, default = 256)
    parser.add_argument('--batch_size_memory', type=int, default = 256)
    parser.add_argument('--weight_decay', type=float, default =0.0)
    parser.add_argument('--last_layer_bn', type=int, default = 0)
    parser.add_argument('--supcon_temperature', type=float, default = 0.1)
    parser.add_argument('--reinit',  type=int, default = 0)
    parser.add_argument('--lr_decay_phase',  type=float, default = 1.0)
 

    # Algortihm params
    parser.add_argument('--phase_epochs', type=int, default = 3)
    parser.add_argument('--early_stop_coef', type=float, default=2.0)
    parser.add_argument("--min_activation_perc", type=float, default=80.0)
    parser.add_argument("--max_phases", type=int, default=10)
    parser.add_argument('--tau_param',  type=float, default = 30)

    # Pretrain load and freeze
    parser.add_argument('--pretrain_load_path', type=str, default = "")
    parser.add_argument('--pretrain_freeze', type=int, default = 0) # 0 = no freezing, 1 = freeze all early conv

    # 0 = No log
    # 1 = Accuracies, #stable/#plastix and model checkpoing after learning all tasks
    # 2 = "1" and Accuracies on all earlier tasks and #stable/#plastic after each task
    # 3 = "2" and model checkpoint and average_activations after each task
    # 4 = "3" and #stable, #plastic, #candidate stable units after each phase
    # 5 = "4" and accuracies on current task for each phase
    # 6 = "5" and model checkpoints and average_activations
    parser.add_argument('--verbose_logging', type=int, default = '4', choices=[0, 1, 2, 3, 4, 5, 6])


    # Noisy Scenario Params
    parser.add_argument('--p_novelty', type=float, default=0.3)
    parser.add_argument('--p_sample', type=float, default=1.0)
    parser.add_argument('--class_per_episode', type=int, default=3)


    return parser.parse_args()

def get_model_config_dict(args: argparse.Namespace) -> dict:
    return getattr(model_config, args.model)


def get_log_param_dict(args: argparse.Namespace) -> dict:
    return {
        "LogPath":  LOG_PATH,
        "DirName": args.experiment_name,
        "save_activations_task": args.verbose_logging in [3, 4, 5, 6],
        "save_activations_phases": args.verbose_logging in [6],
        "save_model_phase": args.verbose_logging in [6],
        "eval_model_phase": args.verbose_logging in [5, 6],
        "save_model_task": args.verbose_logging in [3, 4, 5, 6],
        "write_phase_log": args.verbose_logging in [4, 5, 6],
        "write_task_log": args.verbose_logging in [2, 3, 4, 5, 6],
        "write_sequence_log": args.verbose_logging in [1, 2, 3, 4, 5, 6],
        "no_log": args.verbose_logging == 0
    }


def create_log_dirs(args: argparse.Namespace, log_params: dict) -> None:
    dirpath = os.path.join(log_params["LogPath"], log_params["DirName"])
    # Remove existing files/dirs
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    
    if log_params["no_log"]:
        return
    # Create log dirs and save experiment args
    os.makedirs(dirpath)
    with open(os.path.join(dirpath, 'args.pkl'), 'wb') as file:
        pickle.dump(args, file)

    if log_params["write_task_log"]:
        for task_id in range(1, args.number_of_tasks +1):
            os.makedirs(os.path.join(dirpath, "Task_{}".format(task_id)))

def name2dataset(dataset_name: str, dataset_path = os.path.join(os.path.abspath('..'), 'Datasets'))  -> Tuple[Dataset, Dataset, int, int, transforms.Compose]:
    if dataset_name == "SplitMNIST":
        train_dataset = MNIST(DATASET_PATH, train=True, download=True, transform=transforms.ToTensor())
        test_dataset  = MNIST(DATASET_PATH, train=False, download=True, transform=transforms.ToTensor())
        return (train_dataset, test_dataset, 784, 10, transforms.ToTensor())
    if dataset_name == "EMNIST":
        train_dataset = EMNIST(root=DATASET_PATH, train = True, split='letters', download= True, transform=transforms.ToTensor())
        train_dataset.targets = train_dataset.targets - 1
        test_dataset = EMNIST(root=DATASET_PATH, train = False, split='letters', download= True, transform=transforms.ToTensor())
        test_dataset.targets = test_dataset.targets - 1
        return (train_dataset, test_dataset, 784, 26, transforms.ToTensor())
    if dataset_name == "SplitFMNIST":
        train_dataset = FashionMNIST(DATASET_PATH, train=True, download=True, transform=transforms.ToTensor())
        test_dataset  = FashionMNIST(DATASET_PATH, train=False, download=True, transform=transforms.ToTensor())
        return (train_dataset, test_dataset, 784, 10, transforms.ToTensor())
    if dataset_name == "SplitCIFAR10":
        transform = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        train_dataset = CIFAR10(DATASET_PATH, train=True, download=True, transform=transform)
        test_dataset  = CIFAR10(DATASET_PATH, train=False, download=True, transform=transform)
        return (train_dataset, test_dataset, 3, 10, transform)
    if dataset_name == "SplitCIFAR100":
        transform = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        train_dataset = CIFAR100(DATASET_PATH, train=True, download=True, transform=transform)
        test_dataset  = CIFAR100(DATASET_PATH, train=False, download=True, transform=transform)
        return (train_dataset, test_dataset, 3, 100, transform)
    raise Exception("{} is not defined.".format(dataset_name))


def get_experience_streams(args: argparse.Namespace) -> Tuple[GenericCLScenario, int, int, Dict]:
    train_dataset, test_dataset, input_shape, num_classes, transforms = name2dataset(args.dataset)
    episode_train_datasets, episode_test_datasets = [], []
    seen_classes, novel_classes = [], [i for i in range(num_classes)]
    # First Episode
    episode_classes = np.random.choice(novel_classes, size = args.class_per_episode, replace=False)
    seen_classes.extend(episode_classes)
    episode_train_xs, episode_train_ys = [], []
    episode_test_xs, episode_test_ys = [], []
    for class_ in episode_classes:
        novel_classes.remove(class_)
        x_train, y_train = get_class_samples(train_dataset, class_, args.p_sample)
        episode_train_xs.append(x_train)
        episode_train_ys.append(y_train)
        x_test, y_test = get_class_samples(test_dataset, class_, 1)
        episode_test_xs.append(x_test)
        episode_test_ys.append(y_test)
    episode_train_datasets.append(randomize_examples(torch.vstack(episode_train_xs), torch.concat(episode_train_ys)))
    episode_test_datasets.append((torch.vstack(episode_test_xs), torch.concat(episode_test_ys)))

    # Remaining Episodes
    while(novel_classes):
        new_episode_classes = []
        episode_train_xs, episode_train_ys = [], []
        for class_ in episode_classes:
            if not novel_classes or random.uniform(0, 1) > args.p_novelty:
                classes_to_pick = set(seen_classes).difference(set(new_episode_classes))
                new_class = random.choice(list(classes_to_pick))
            else:
                new_class = random.choice(novel_classes)
                novel_classes.remove(new_class)
                seen_classes.append(new_class)
            new_episode_classes.append(new_class)
            x_train, y_train = get_class_samples(train_dataset, new_class, args.p_sample)
            episode_train_xs.append(x_train)
            episode_train_ys.append(y_train)
        episode_train_datasets.append(randomize_examples(torch.vstack(episode_train_xs), torch.concat(episode_train_ys)))
        episode_classes = new_episode_classes

        episode_test_xs, episode_test_ys = [], []
        for class_ in seen_classes:
            x_test, y_test = get_class_samples(test_dataset, class_, 1)
            episode_test_xs.append(x_test)
            episode_test_ys.append(y_test)
        episode_test_datasets.append((torch.vstack(episode_test_xs), torch.concat(episode_test_ys)))

    # Create Avalanche objects
    generic_scenario = tensors_benchmark(train_tensors=episode_train_datasets,
                                         test_tensors=episode_test_datasets,
                                         task_labels=[0]*len(episode_train_datasets),
                                         train_transform=None, eval_transform=None)

    stream_with_val = benchmark_with_validation_stream(generic_scenario, validation_size=0.01,
                                                       output_stream="val", shuffle=True)
    task2classes = dict((index, stream.classes_in_this_experience)
                            for index, stream in enumerate(stream_with_val.train_stream, 1))
    return (stream_with_val, input_shape, num_classes, task2classes)
    


def get_class_samples(dataset: Dataset, sample_class: int, p_sample: float) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = dataset.targets == sample_class  # type: ignore
    num_examples = int(sum(idx) * p_sample)
    x_all = dataset.data[idx]  # type: ignore
    x = x_all[torch.randperm(x_all.shape[0])[:num_examples]]
    x = (x.float() / 255)
    if x.shape[1] == 28 and x.shape[2] == 28: 
        x = torch.unsqueeze(x, 1)
    y = torch.tensor([sample_class] * num_examples)
    return (x, y)


def randomize_examples(x_tensor: torch.Tensor, y_tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randperm(x_tensor.size(0))
    return (x_tensor[idx], y_tensor[idx])