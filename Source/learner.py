from argparse import Namespace

from typing import Dict, Optional, Tuple
from avalanche.benchmarks import GenericCLScenario, TCLExperience
import torch
from torch.optim.lr_scheduler import StepLR


from  Source.architecture import random_prune, Network
from  Source.train_and_eval import  task_training_supcon
from  Source.track import TaskLog, SequenceLog, PhaseLog
from  Source.helper import get_data_loaders, get_n_samples_per_class, get_device
from  Source.neocortex import  NeocortexKNN
from  Source.memory import MemoryBuffer
from  Source.rewire import  get_tau_schedule, drop_connections, Growth
from  Source.supcon import SupConLoss



class Learner():

    def __init__(self, args: Namespace, network: Network, scenario: GenericCLScenario,
                 log_params: dict, config_dict: dict, input_size: int, task2classes: Dict):
        self.args = args
        self.config_dict = config_dict
        self.input_size = input_size
        self.optim_obj = getattr(torch.optim, args.optimizer)
        self.network = random_prune(network.to(get_device()), args.prune_perc)
        self.neocortex = NeocortexKNN(args, config_dict, task2classes)
        self.memory = MemoryBuffer(args, task2classes, input_size) if args.stm_num_tasks else None
        self.original_scenario = scenario
        self.tau_schedule = get_tau_schedule(args)
        self.log_params = log_params
        self.seen_classes = set()
        print("Model: \n", self.network)

    def end_episode(self, train_task, episode_index, memory_and_range):
        print("Ending the learning episode.")
        self.network.promote_units()
        self.network.update_freeze_masks()

        # Push to Memory
        if self.memory:
            subsets = get_n_samples_per_class(train_task, self.args.stm_per_class)
            all_samples = [self.network.get_memory_representations(samples.to(get_device())).detach()
                           for samples, _ in subsets]
            labels = [label for _, label in subsets]
            self.memory.insert_samples(all_samples, labels)

        if  self.args.reinit:
            self.network.re_initialize_u0()

        self.create_ltm_representations(train_task, memory_and_range)
            
        task_logger = TaskLog(self.args, self.log_params, episode_index + 1, self.original_scenario)
        task_logger.archive(self.network, self.neocortex)



    def learn_next_episode(self, episode_index: int, train_task: TCLExperience, val_task: TCLExperience, test_task: TCLExperience) -> Network:
        print("****** Learning Episode-{}   Classes: {} ******".format(episode_index + 1, train_task.classes_in_this_experience))
        self.network.add_seen_classes(train_task.classes_in_this_experience)
        train_loader, val_loader, test_loader = get_data_loaders(self.args, train_task, val_task, test_task, episode_index)

        # Select all units before the task
        phase_index = 1
        selection_perc = self.tau_schedule(phase_index + 1) * 100
        loss = SupConLoss(self.args)
        while(True):
            print('Selecting Units')
            u0_units, u1_units = self.network.select_u0_u1_units(train_task, selection_perc, episode_index)
            self.network.set_current_u0_and_u1_units(u0_units, u1_units)
            print('Dropping connections')
            self.network, connection_quota = drop_connections(self.network)
            print('Growing connections.')
            connection_grower = Growth(train_loader, self.network)
            self.network, connection_quota = connection_grower.grow(self.network, connection_quota)
            if sum(connection_quota) != 0:
                print("Warning: Cannot accomodate all connection growth request. Density will decrease.")
                print(connection_quota)

            print("Sparsity phase-{}: {:.2f}".format(phase_index, self.network.compute_weight_sparsity()))
            epochs = self.args.phase_epochs
            memory_and_range = None
            if episode_index and self.memory is not None:
                episode_range = list(range(episode_index + 1 - self.args.stm_num_tasks, episode_index + 1))
                episode_range = [i for i in episode_range if i > 0] 
                print("This is Episode-{} and we have Episode(s) {} in Memory".format(episode_index + 1, episode_range))
                memory_and_range = (self.memory, episode_range)

            current_lr = self.args.learning_rate*(self.args.lr_decay_phase**(phase_index-1))
            print("Learning Rate: ", current_lr)
            if self.args.optimizer == "SGD":
                optimizer = self.optim_obj(self.network.parameters(),
                                           lr= current_lr, weight_decay= 0.0,
                                           momentum = self.args.sgd_momentum)
            else:
                optimizer = self.optim_obj(self.network.parameters(),
                                           lr= current_lr, weight_decay= 0.0)

            # Phase Training
            self.network = task_training_supcon(self.network, epochs,
                                loss, optimizer, train_loader, val_loader, self.args, memory_and_range)
            phase_logger = PhaseLog(self.original_scenario, self.log_params, phase_index,
                                    train_task.classes_in_this_experience, episode_index + 1)
            phase_logger.archive(self.network, self.neocortex, train_loader, val_loader, test_loader)        
            
            # Increased Val
            if phase_index == self.args.max_phases:
                break
            
            phase_index = phase_index + 1
            selection_perc = self.tau_schedule(phase_index) * 100

        self.end_episode(train_task, episode_index, memory_and_range)
        return self.network
    
    def create_ltm_representations(self, dataset: TCLExperience, memory_and_range: Optional[Tuple]) -> None:
        classes_data = get_n_samples_per_class(dataset, self.args.ltm_per_class)
        input_batches = [data.to(get_device()) for data, _ in classes_data]
        labels = [label for _, label in classes_data]
        representations = []
        self.network.eval()
        for input_batch in input_batches:
            output = self.network.forward(input_batch)
            representations.append(output.detach()) # type: ignore


        valid_units = self.network.get_memory_mask_units(unit_type = 2)
        self.neocortex.add_class_representation(representations, labels, list(valid_units))



        if memory_and_range:
            memory, task_range = memory_and_range
            samples, memory_labels, ages = memory.get_n_from_classes(self.args.ltm_per_class, task_range)
            self.network.eval()
            for class_samples, label, age  in zip(samples, memory_labels, ages):
                output = self.network.forward_memo(class_samples.to(get_device())).detach()
                valid_units = self.network.get_memory_mask_units(unit_type = 3 + age)
                self.neocortex.add_class_representation([output], [label], list(valid_units))  # type: ignore

        return None
    
    def learn_all_episodes(self) -> Network:
        for task_index, (train_task, val_task, test_task) in enumerate(zip(self.original_scenario.train_stream,
                                                                           self.original_scenario.val_stream,  # type: ignore
                                                                           self.original_scenario.test_stream)):
            network = self.learn_next_episode(task_index, train_task, val_task, test_task) 
        sequence_logger = SequenceLog(self.args, self.log_params, self.original_scenario)
        sequence_logger.archive(network, self.neocortex)  # type: ignore
        return network  # type: ignore




            

