import argparse
import os
import pickle
import shutil
import random
from typing import Tuple, Dict

from avalanche.benchmarks import GenericCLScenario
from avalanche.benchmarks.classic import SplitCIFAR10, SplitCIFAR100, SplitFMNIST, SplitMNIST
from avalanche.benchmarks.generators import benchmark_with_validation_stream, nc_benchmark
from avalanche.benchmarks.datasets import EMNIST
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
    parser.add_argument('--experiment_name', type=str, default = 'Test_CE')
    parser.add_argument("--gpu_prefetch", type=int, default = 1)

    # LTM Memory Params
    parser.add_argument('--ltm_per_class', type=int, default = 25)
    parser.add_argument('--ltm_k_nearest', type=int, default = 5)

    # STM Memory Params
    parser.add_argument('--stm_per_class', type=int, default = 50)
    parser.add_argument('--stm_num_tasks', type=int, default = 1) # 1 (current) + 2 (most recent)
    parser.add_argument('--memory_mode', type=str, default="internal", choices=["raw", "internal"])

    # Dataset params
    parser.add_argument('--dataset', type=str, default = 'EMNIST')
    parser.add_argument('--number_of_tasks', type=int, default = 13)

    # Architectural params
    parser.add_argument('--model', type=str, default = 'MNISTLIKE')
    parser.add_argument('--prune_perc', type=float, default=60.0)

    # Learning params
    parser.add_argument('--seed', type=int,  default=0)
    parser.add_argument('--permute_seed', type=int,  default=0)
    parser.add_argument('--deterministic', type=int,  default=1)

    # Anything under torch.optim works. e.g., 'SGD' and 'Adam'
    parser.add_argument('--optimizer', type=str, default = 'Adadelta')
    parser.add_argument('--sgd_momentum', type=float, default = 0.90)
    parser.add_argument('--learning_rate', type=float, default = 1.0)
    parser.add_argument('--batch_size', type=int, default = 256)
    parser.add_argument('--batch_size_memory', type=int, default = 256)
    parser.add_argument('--weight_decay', type=float, default =0.0)
    parser.add_argument('--supcon_temperature', type=float, default = 0.1)
    parser.add_argument('--reinit',  type=int, default = 1)
   
 

    # Algortihm params
    parser.add_argument('--phase_epochs', type=int, default = 3)
    parser.add_argument("--min_activation_perc", type=float, default=60.0)
    parser.add_argument("--max_phases", type=int, default=10)
    parser.add_argument('--tau_param',  type=float, default = 30) 

    # Pretrain load and freeze
    parser.add_argument('--pretrain_load_path', type=str, default = "")
    parser.add_argument('--pretrain_freeze', type=int, default = 0) # 0 = no freezing, 1 = freeze all early conv

    # parser.add_argument('--last_layer_bn', type=int, default = 0) Not used anymore
    # parser.add_argument('--lr_decay_phase',  type=float, default = 1.0)
    # parser.add_argument('--early_stop_coef', type=float, default=2.0) Not used anymore

    parser.add_argument('--verbose_logging', type=int, default = '4', choices=[0, 1, 2, 3, 4, 5, 6])

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


def get_experience_streams(args: argparse.Namespace) -> Tuple[GenericCLScenario, int, int, Dict]:
    if args.dataset == "EMNIST":
        emnist_train = EMNIST(root=DATASET_PATH, train = True, split='letters', download= True)
        emnist_train.targets = emnist_train.targets - 1
        emnist_test = EMNIST(root=DATASET_PATH, train = False, split='letters', download= True)
        emnist_test.targets = emnist_test.targets - 1
        stream = nc_benchmark(train_dataset=emnist_train,test_dataset=emnist_test, # type: ignore
                              n_experiences=args.number_of_tasks, task_labels = False, shuffle = False,
                              seed = args.seed, fixed_class_order=list(map(lambda x: int(x), emnist_train.targets.unique())),
                              train_transform = transforms.ToTensor(),
                              eval_transform=transforms.ToTensor())
        
        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.01, output_stream="val", shuffle=True)
        task2classes = dict((index, stream.classes_in_this_experience)
                            for index, stream in enumerate(stream_with_val.train_stream, 1))
        return (stream_with_val, 784, 46, task2classes)
    
    if args.dataset == "EMNIST_P":
        emnist_train = EMNIST(root=DATASET_PATH, train = True, split='letters', download= True)
        emnist_train.targets = emnist_train.targets - 1
        emnist_test = EMNIST(root=DATASET_PATH, train = False, split='letters', download= True)
        emnist_test.targets = emnist_test.targets - 1
        local_random = random.Random()
        local_random.seed(args.permute_seed)
        original_list = list(map(lambda x: int(x), emnist_train.targets.unique()))
        if args.permute_seed != 0:
            original_list = local_random.sample(original_list, len(original_list))
        stream = nc_benchmark(train_dataset=emnist_train,test_dataset=emnist_test, # type: ignore
                              n_experiences=args.number_of_tasks, task_labels = False, shuffle = False,
                              seed = args.seed, fixed_class_order=original_list,
                              train_transform = transforms.ToTensor(),
                              eval_transform=transforms.ToTensor())
        
        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.01, output_stream="val", shuffle=True)
        task2classes = dict((index, stream.classes_in_this_experience)
                            for index, stream in enumerate(stream_with_val.train_stream, 1))
        return (stream_with_val, 784, 46, task2classes)

    if args.dataset == "SplitMNIST":
        stream = SplitMNIST(n_experiences = args.number_of_tasks,
                            seed = args.seed, dataset_root=DATASET_PATH, fixed_class_order=list(range(10)))
        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.01, output_stream="val", shuffle=True)
        task2classes = dict((index, stream.classes_in_this_experience)
                            for index, stream in enumerate(stream_with_val.train_stream, 1))
        return (stream_with_val, 784, 10, task2classes)
    
    if args.dataset == "SplitFMNIST":
        stream = SplitFMNIST(n_experiences = args.number_of_tasks,
                             seed = args.seed, dataset_root=DATASET_PATH, fixed_class_order=list(range(10)))
        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.01, output_stream="val", shuffle=True)
        task2classes = dict((index, stream.classes_in_this_experience)
                            for index, stream in enumerate(stream_with_val.train_stream, 1))
        return (stream_with_val, 784, 10, task2classes)
    
    if args.dataset == "SplitCIFAR10":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        stream = SplitCIFAR10(n_experiences = args.number_of_tasks,
                              seed = args.seed, dataset_root=DATASET_PATH, fixed_class_order=list(range(10)),
                              train_transform=transform, eval_transform=transform)
        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.01, output_stream="val", shuffle=True)
        task2classes = dict((index, stream.classes_in_this_experience)
                            for index, stream in enumerate(stream_with_val.train_stream, 1))
        return (stream_with_val, 3, 10, task2classes)
    
    if args.dataset == "SplitCIFAR100":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        stream = SplitCIFAR100(n_experiences = args.number_of_tasks,
                               seed = args.seed, dataset_root=DATASET_PATH, fixed_class_order=list(range(100)),
                               train_transform=transform, eval_transform=transform)
        stream_with_val = benchmark_with_validation_stream(stream, validation_size=0.01, output_stream="val", shuffle=True)
        task2classes = dict((index, stream.classes_in_this_experience)
                            for index, stream in enumerate(stream_with_val.train_stream, 1))
        return (stream_with_val, 3, 100, task2classes)
    raise Exception("Dataset {} is not defined!".format(args.dataset))