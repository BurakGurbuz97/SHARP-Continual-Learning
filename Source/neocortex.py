from typing import List, Tuple, Dict
import torch
import argparse
from torch.nn.functional import normalize
import numpy as np


from Source.helper import get_device


class NeocortexKNN():
    def __init__(self, args: argparse.Namespace, config_dict: Dict, task2classes: Dict) -> None:
        self.args = args
        self.n_neighbors = args.ltm_k_nearest
        self.config_dict = config_dict
        self.class_representations = dict()
        self.class_representation_masks = dict()
        self.task2classes = task2classes

    def add_class_representation(self, class_representations: List[torch.Tensor], labels: List, valid_units: List[int]) -> None:
        
        for representations, label in zip(class_representations, labels):
            self.class_representations[label] = representations.to(get_device())
            mask = torch.zeros((self.config_dict["penultimate_layer_size"])).to(get_device())
            if valid_units:
                mask[valid_units] = 1
            else:
                mask[:] = 1
            self.class_representation_masks[label] = mask

    def get_class_masked_representations(self) -> Tuple[torch.Tensor, torch.Tensor]:
        masked_representations = []
        labels = []
        for label, representations in self.class_representations.items():
            masked_representations.append(representations * self.class_representation_masks[label])
            labels.extend([label]*len(representations))

        return torch.vstack(masked_representations), torch.tensor(labels)
    

    def get_class_representations_and_masks(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        class_representations = []
        class_masks = []
        labels = []
        for label, representations in self.class_representations.items():
            class_representations.append(representations)
            class_masks.extend([self.class_representation_masks[label]]*len(representations))
            labels.extend([label]*len(representations))
        return torch.vstack(class_representations).to(get_device()), torch.vstack(class_masks).to(get_device()), torch.tensor(labels).to(get_device())
    
    def predict(self, representations: torch.Tensor, episode_id = None) -> torch.Tensor:
        class_distances = []
        labels = []
        # Calculate Distances
        for label, class_ltm_representations in self.class_representations.items():
            # Mask Out irrelevant units 
            masked_class_ltm_representations = class_ltm_representations[:, self.class_representation_masks[label].to(torch.bool)]
            masked_representations = representations[:, self.class_representation_masks[label].to(torch.bool)]
            masked_class_ltm_representations = normalize(masked_class_ltm_representations, p = 2, dim = 1)
            masked_representations = normalize(masked_representations, p = 2, dim = 1)
            dists = torch.sqrt(torch.sum((masked_class_ltm_representations[:, None, :] - masked_representations[None, :, :]) ** 2,
                                         dim=-1))
            labels.extend([label]*len(class_ltm_representations))
            class_distances.append(dists)
        class_distances = torch.vstack(class_distances)
        if episode_id is not None:
            class_distances[:(self.args.ltm_per_class) * min(self.task2classes[episode_id+1]), :] = 99999
            class_distances[(self.args.ltm_per_class) * (max(self.task2classes[episode_id+1])+1):, :] = 99999
        labels = torch.tensor(labels).to(get_device())

        # Predict Classes
        predictions = []
        for sample_index in range(len(representations)):
            indices = torch.topk(class_distances[:, sample_index], k = self.n_neighbors, largest = False)[1].to(get_device())
            predictions.append(int(torch.mode(labels[indices])[0]))

        return torch.tensor(predictions)
            