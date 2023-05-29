from argparse import Namespace
from avalanche.benchmarks import GenericCLScenario
from typing import List, Tuple
from torch.utils.data import DataLoader
import torch
import os
import csv
import pickle
import numpy as np
from numpy import typing as np_type
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from Source.architecture import Network
from Source.train_and_eval import test
from Source.helper import get_data_loaders, get_device
from Source.neocortex import NeocortexKNN


def acc_prev_tasks(args: Namespace, neocortex: NeocortexKNN, task_index: int,
                   scenario: GenericCLScenario, network: Network, til_eval = False) -> List[Tuple[str, List]]:
    all_accuracies = []
    all_preds = []
    all_trues = []
    for episode_id, (train_task, val_task, test_task) in enumerate(zip(scenario.train_stream[:task_index+1], 
                                                scenario.val_stream[:task_index+1],  # type: ignore
                                                scenario.test_stream[:task_index+1])):
        episode_id = episode_id if til_eval else None
        task_classes = str(test_task.classes_in_this_experience)
        train_loader, val_loader, test_loader = get_data_loaders(args, train_task, val_task, test_task)
        train_acc = 0.0#test(network, neocortex, train_loader, episode_id)
        val_acc = 0.0#test(network, neocortex, val_loader, episode_id)
        test_acc, preds, ground_truths = test(network, neocortex, test_loader, episode_id, return_preds=True) # type: ignore
        all_preds.append(preds)
        all_trues.append(ground_truths)
        all_accuracies.append((task_classes, [train_acc, val_acc, test_acc]))
    return all_accuracies, all_preds, all_trues

def _write_units(writer, network: Network):
    weights = network.get_weight_bias_masks_numpy()
    all_units = [list(range(weights[0][0].shape[1]))] + [list(range(w[1].shape[0])) for w in weights]
    writer.writerow(["All Units"] + [len(u) for u in all_units])

    unit_type_list = network.unit_type_list
    writer.writerow(["Plastic Units"] + [len((u == 0).nonzero()[0]) for u, _ in unit_type_list])
    writer.writerow(["Learner Units"] + [len((u == 1).nonzero()[0]) for u, _ in unit_type_list])
    writer.writerow(["Memory Units"] + [len(( (u <= network.args.stm_num_tasks + 1) & (u > 1) ).nonzero()[0]) for u, _ in unit_type_list])
    writer.writerow(["Stable Units"] + [len((u > network.args.stm_num_tasks + 1).nonzero()[0]) for u, _ in unit_type_list])
    return writer


class PhaseLog():
    def __init__(self, scenario: GenericCLScenario,log_params: dict, phase_index: int, task_classes: List, task_index: int):
        self.log_params = log_params
        self.phase_index = phase_index
        self.task_classes = task_classes
        self.task_index = task_index
        self.scenario = scenario

    def _open_dir(self) -> str:
        dirpath = os.path.join(self.log_params["LogPath"], self.log_params["DirName"],
                              "Task_{}".format(self.task_index), "Phase_{}".format(self.phase_index))
        os.makedirs(dirpath)
        return dirpath

    def archive(self, network: Network, neocortex: NeocortexKNN,   train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
        if self.log_params["write_phase_log"]:
            dirpath = self._open_dir()
            csvfile = open(os.path.join(dirpath, "Phase_{}.csv".format(self.phase_index)), 'w', newline='')
            writer = csv.writer(csvfile)
            writer = _write_units(writer, network)
            
            if self.log_params["eval_model_phase"]:
                writer.writerow(["Train Accuracy", test(network, neocortex, train_loader)])
                writer.writerow(["Validation Accuracy", test(network, neocortex, val_loader)])
                writer.writerow(["Test Accuracy", test(network, neocortex, test_loader)])

            csvfile.close()

            # if self.log_params["save_model_phase"]:
            #     torch.save(network.state_dict(), os.path.join(dirpath, 'network_end_of_phase.pth'))


class TaskLog():
    def __init__(self, args: Namespace, log_params: dict, task_index: int, scenario: GenericCLScenario):
        self.log_params = log_params
        self.scenario = scenario
        self.task_index = task_index
        self.args = args

    def archive(self, network: Network, neocortex: NeocortexKNN):
        if self.log_params["write_task_log"]:
            dirpath = os.path.join(self.log_params["LogPath"], self.log_params["DirName"],
                              "Task_{}".format(self.task_index))
            csvfile = open(os.path.join(dirpath, "Task_{}.csv".format(self.task_index)), 'w', newline='')

            writer = csv.writer(csvfile)
            writer = _write_units(writer, network)

            # TIL
            # This function assumes task_index starts from 0 so we have -1
            prev_task_accs, _, _ = acc_prev_tasks(self.args, neocortex, self.task_index - 1, self.scenario, network, til_eval=True)
            writer.writerow(["Task Incremental Learning"])
            for task_classes, (train_acc, val_acc, test_acc) in prev_task_accs:
                writer.writerow([str(task_classes), "Train Acc: {:.2f}".format(train_acc),
                                 "Val Acc: {:.2f}".format(val_acc), "Test Acc: {:.2f}".format(test_acc)])
                
            # CIL
            # This function assumes task_index starts from 0 so we have -1
            prev_task_accs, all_preds, all_trues = acc_prev_tasks(self.args, neocortex, self.task_index - 1, self.scenario, network)
            writer.writerow(["Class Incremental Learning"])
            for task_classes, (train_acc, val_acc, test_acc) in prev_task_accs:
                writer.writerow([str(task_classes), "Train Acc: {:.2f}".format(train_acc),
                                 "Val Acc: {:.2f}".format(val_acc), "Test Acc: {:.2f}".format(test_acc)])
                
            # Create confusin matrix
            """
            ground_truth = np.concatenate(all_trues)
            predictions  = np.concatenate(all_preds)
            cm = confusion_matrix(ground_truth, predictions)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            multiplier = 20 if self.args.dataset == "SplitCIFAR100" else 2
            plt.figure(figsize=(len(prev_task_accs)*multiplier + 5, len(prev_task_accs)*multiplier))
            sns.set(font_scale=1.4)
            ax = sns.heatmap(cm, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 16})
            ax.set_xlabel('Predicted Labels', fontsize=18)
            ax.set_ylabel('True Labels', fontsize=18)
            ax.set_title('Confusion Matrix', fontsize=22)
            plt.savefig(os.path.join(dirpath, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
            """

            # if self.log_params["save_model_task"]:
            #     torch.save(network.state_dict(), os.path.join(dirpath, 'network_end_of_phase.pth'))
            csvfile.close()

class SequenceLog():
    def __init__(self, args: Namespace, log_params: dict, scenario: GenericCLScenario):
        self.log_params = log_params
        self.scenario = scenario
        self.args = args

    def archive(self, network: Network, neocortex: NeocortexKNN):
        if self.log_params["write_sequence_log"]:
            dirpath = os.path.join(self.log_params["LogPath"], self.log_params["DirName"])
            csvfile = open(os.path.join(dirpath, "End_of_Sequence.csv"), 'w', newline='')
            writer = csv.writer(csvfile)
            writer = _write_units(writer, network)
            # This function assumes task_index starts from 0 so we have -1
            prev_task_accs, _, _ = acc_prev_tasks(self.args, neocortex, self.args.number_of_tasks - 1, self.scenario, network)
            for task_classes, (train_acc, val_acc, test_acc) in prev_task_accs:
                writer.writerow([str(task_classes), "Train Acc: {:.2f}".format(train_acc), "Val Acc: {:.2f}".format(val_acc), "Test Acc: {:.2f}".format(test_acc)])

            # torch.save(network.state_dict(), os.path.join(dirpath, 'network_final.pth'))
            csvfile.close()
