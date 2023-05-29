import argparse
from typing import List, Dict, Tuple

import torch
from numpy.typing import NDArray
import numpy as np


class QuantizedTensor:
    def __init__(self, input_tensor: torch.Tensor, bits: int = 8):
        self.bits = bits
        self.min_val = input_tensor.min()
        self.max_val = input_tensor.max()
        
        # Normalize the input tensor to the range [0, 1]
        normalized_tensor = (input_tensor - self.min_val) / (self.max_val - self.min_val)

        # Scale the normalized tensor to the quantization range
        quantization_range = 2 ** self.bits - 1
        scaled_tensor = normalized_tensor * quantization_range

        # Round the values and convert them to integers
        self.quantized_tensor = torch.round(scaled_tensor).to(torch.uint8)

    def dequantize(self) -> torch.Tensor:
        # Convert the quantized tensor back to float32
        dequantized_normalized_tensor = self.quantized_tensor.to(torch.float32) / (2 ** self.bits - 1)
        dequantized_tensor = dequantized_normalized_tensor * (self.max_val - self.min_val) + self.min_val
        return dequantized_tensor

class MemoryBuffer():

    def __init__(self, args: argparse.Namespace,  task2classes: Dict, representation_size: int):
        self.args = args
        self.memory = {}
        self.task2classes = task2classes
        self.representation_size = representation_size
        for _, classes in self.task2classes.items():
            for class_ in classes:
                self.memory[class_] = None

    def insert_samples(self, all_samples: List[torch.Tensor], labels: List) -> None:
        for label, class_samples in zip(labels, all_samples):
           self.memory[int(label)] = QuantizedTensor(class_samples)
        return None
            

    def sample_n(self, n: int, tasks: List) -> Tuple[NDArray, NDArray]:
        samples_per_class = self.get_samples_per_class(n, tasks)
        samples = []
        labels = []
        i = 0
        for task in tasks:
            for class_ in self.task2classes[task]:
                class_samples = self.memory[class_].dequantize().cpu()
                try:
                    samples.extend(class_samples[np.random.choice(class_samples.shape[0],
                                                        samples_per_class[i], replace=False)])
                except:
                    # If not enough memory samples, sample with replacement.
                    samples.extend(class_samples[np.random.choice(class_samples.shape[0],
                                                        samples_per_class[i], replace=True)])
                labels.extend(list([class_])*samples_per_class[i])
                i = i + 1
        return np.stack(samples), np.array(labels)
    
    def get_n_from_classes(self, n: int, tasks: List) -> Tuple:
        labels = []
        samples = []
        time = 0
        ages = []
        for task in reversed(tasks):
            for class_ in self.task2classes[task]:
                class_samples = self.memory[class_].dequantize().cpu()
                try:
                    samples.append(class_samples[np.random.choice(class_samples.shape[0],n, replace=False)])
                except:
                    # If not enough memory samples, sample with replacement.
                    samples.append(class_samples[np.random.choice(class_samples.shape[0],n, replace=True)])
                labels.append(class_)
                ages.append(time)
            time = time + 1
        return samples, labels, ages


    def get_samples_per_class(self, n: int, tasks: List) -> List:
        number_of_classes = self.task2numclasses(tasks)
        a =  int(n / number_of_classes)
        remaining = n % number_of_classes
        samples_per_class = [a for _ in range(number_of_classes)]
        for i in range(remaining):
            samples_per_class[i] = samples_per_class[i] + 1
        return samples_per_class
    
    def task2numclasses(self, tasks: List) -> int:
        numclasses = 0
        for task in tasks:
            numclasses = numclasses + len(self.task2classes[task]) 
        return numclasses