#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate research


# MNIST SHARP 3 Seeds
python launch.py --experiment_name "MNIST_SHARP_SEED0" --gpu_prefetch "1" --ltm_per_class "25" --ltm_k_nearest "5" --stm_per_class "25" --stm_num_tasks "1" --memory_mode "internal" --dataset "SplitMNIST" --number_of_tasks "5" --model "MNISTLIKE" --prune_perc "60.0" --seed "0" --permute_seed "0" --deterministic "1" --optimizer "Adadelta" --sgd_momentum "0.9" --learning_rate "1.0" --batch_size "256" --batch_size_memory "256" --weight_decay "0.0" --last_layer_bn "0" --supcon_temperature "0.1" --reinit "1" --lr_decay_phase "1.0" --phase_epochs "3" --early_stop_coef "2.0" --min_activation_perc "90.0" --max_phases "10" --tau_param "30.0" --pretrain_load_path "" --pretrain_freeze "0" --verbose_logging "4"
python launch.py --experiment_name "MNIST_SHARP_SEED1" --gpu_prefetch "1" --ltm_per_class "25" --ltm_k_nearest "5" --stm_per_class "25" --stm_num_tasks "1" --memory_mode "internal" --dataset "SplitMNIST" --number_of_tasks "5" --model "MNISTLIKE" --prune_perc "60.0" --seed "1" --permute_seed "0" --deterministic "1" --optimizer "Adadelta" --sgd_momentum "0.9" --learning_rate "1.0" --batch_size "256" --batch_size_memory "256" --weight_decay "0.0" --last_layer_bn "0" --supcon_temperature "0.1" --reinit "1" --lr_decay_phase "1.0" --phase_epochs "3" --early_stop_coef "2.0" --min_activation_perc "90.0" --max_phases "10" --tau_param "30.0" --pretrain_load_path "" --pretrain_freeze "0" --verbose_logging "4"
python launch.py --experiment_name "MNIST_SHARP_SEED2" --gpu_prefetch "1" --ltm_per_class "25" --ltm_k_nearest "5" --stm_per_class "25" --stm_num_tasks "1" --memory_mode "internal" --dataset "SplitMNIST" --number_of_tasks "5" --model "MNISTLIKE" --prune_perc "60.0" --seed "2" --permute_seed "0" --deterministic "1" --optimizer "Adadelta" --sgd_momentum "0.9" --learning_rate "1.0" --batch_size "256" --batch_size_memory "256" --weight_decay "0.0" --last_layer_bn "0" --supcon_temperature "0.1" --reinit "1" --lr_decay_phase "1.0" --phase_epochs "3" --early_stop_coef "2.0" --min_activation_perc "90.0" --max_phases "10" --tau_param "30.0" --pretrain_load_path "" --pretrain_freeze "0" --verbose_logging "4"


# FMNIST SHARP 3 Seeds
python launch.py --experiment_name "FMNIST_SHARP_SEED0" --gpu_prefetch "1" --ltm_per_class "25" --ltm_k_nearest "25" --stm_per_class "25" --stm_num_tasks "1" --memory_mode "internal" --dataset "SplitFMNIST" --number_of_tasks "5" --model "MNISTLIKE" --prune_perc "60.0" --seed "0" --permute_seed "0" --deterministic "1" --optimizer "Adadelta" --sgd_momentum "0.9" --learning_rate "1.0" --batch_size "1024" --batch_size_memory "1024" --weight_decay "0.0" --last_layer_bn "0" --supcon_temperature "0.2" --reinit "1" --lr_decay_phase "1.0" --phase_epochs "3" --early_stop_coef "2.0" --min_activation_perc "75.0" --max_phases "12" --tau_param "30.0" --pretrain_load_path "" --pretrain_freeze "0" --verbose_logging "4"
python launch.py --experiment_name "FMNIST_SHARP_SEED1" --gpu_prefetch "1" --ltm_per_class "25" --ltm_k_nearest "25" --stm_per_class "25" --stm_num_tasks "1" --memory_mode "internal" --dataset "SplitFMNIST" --number_of_tasks "5" --model "MNISTLIKE" --prune_perc "60.0" --seed "1" --permute_seed "0" --deterministic "1" --optimizer "Adadelta" --sgd_momentum "0.9" --learning_rate "1.0" --batch_size "1024" --batch_size_memory "1024" --weight_decay "0.0" --last_layer_bn "0" --supcon_temperature "0.2" --reinit "1" --lr_decay_phase "1.0" --phase_epochs "3" --early_stop_coef "2.0" --min_activation_perc "75.0" --max_phases "12" --tau_param "30.0" --pretrain_load_path "" --pretrain_freeze "0" --verbose_logging "4"
python launch.py --experiment_name "FMNIST_SHARP_SEED2" --gpu_prefetch "1" --ltm_per_class "25" --ltm_k_nearest "25" --stm_per_class "25" --stm_num_tasks "1" --memory_mode "internal" --dataset "SplitFMNIST" --number_of_tasks "5" --model "MNISTLIKE" --prune_perc "60.0" --seed "2" --permute_seed "0" --deterministic "1" --optimizer "Adadelta" --sgd_momentum "0.9" --learning_rate "1.0" --batch_size "1024" --batch_size_memory "1024" --weight_decay "0.0" --last_layer_bn "0" --supcon_temperature "0.2" --reinit "1" --lr_decay_phase "1.0" --phase_epochs "3" --early_stop_coef "2.0" --min_activation_perc "75.0" --max_phases "12" --tau_param "30.0" --pretrain_load_path "" --pretrain_freeze "0" --verbose_logging "4"


# EMNIST SHARP 3 Seeds
python launch.py --experiment_name "EMNIST_SHARP_SEED0" --gpu_prefetch "1" --ltm_per_class "25" --ltm_k_nearest "5" --stm_per_class "25" --stm_num_tasks "1" --memory_mode "internal" --dataset "EMNIST" --number_of_tasks "13" --model "MNISTLIKE" --prune_perc "60.0" --seed "0" --permute_seed "0" --deterministic "1" --optimizer "Adadelta" --sgd_momentum "0.9" --learning_rate "1.0" --batch_size "256" --batch_size_memory "256" --weight_decay "0.0" --last_layer_bn "0" --supcon_temperature "0.1" --reinit "1" --lr_decay_phase "1.0" --phase_epochs "3" --early_stop_coef "2.0" --min_activation_perc "60.0" --max_phases "15" --tau_param "30.0" --pretrain_load_path "" --pretrain_freeze "0" --verbose_logging "4"
python launch.py --experiment_name "EMNIST_SHARP_SEED1" --gpu_prefetch "1" --ltm_per_class "25" --ltm_k_nearest "5" --stm_per_class "25" --stm_num_tasks "1" --memory_mode "internal" --dataset "EMNIST" --number_of_tasks "13" --model "MNISTLIKE" --prune_perc "60.0" --seed "1" --permute_seed "0" --deterministic "1" --optimizer "Adadelta" --sgd_momentum "0.9" --learning_rate "1.0" --batch_size "256" --batch_size_memory "256" --weight_decay "0.0" --last_layer_bn "0" --supcon_temperature "0.1" --reinit "1" --lr_decay_phase "1.0" --phase_epochs "3" --early_stop_coef "2.0" --min_activation_perc "60.0" --max_phases "15" --tau_param "30.0" --pretrain_load_path "" --pretrain_freeze "0" --verbose_logging "4"
python launch.py --experiment_name "EMNIST_SHARP_SEED2" --gpu_prefetch "1" --ltm_per_class "25" --ltm_k_nearest "5" --stm_per_class "25" --stm_num_tasks "1" --memory_mode "internal" --dataset "EMNIST" --number_of_tasks "13" --model "MNISTLIKE" --prune_perc "60.0" --seed "2" --permute_seed "0" --deterministic "1" --optimizer "Adadelta" --sgd_momentum "0.9" --learning_rate "1.0" --batch_size "256" --batch_size_memory "256" --weight_decay "0.0" --last_layer_bn "0" --supcon_temperature "0.1" --reinit "1" --lr_decay_phase "1.0" --phase_epochs "3" --early_stop_coef "2.0" --min_activation_perc "60.0" --max_phases "15" --tau_param "30.0" --pretrain_load_path "" --pretrain_freeze "0" --verbose_logging "4"


# CIFAR10 SHARP 3 Seeds
python launch.py --experiment_name "CIFAR10_SHARP_SEED0" --gpu_prefetch "1" --ltm_per_class "50" --ltm_k_nearest "5" --stm_per_class "200" --stm_num_tasks "1" --memory_mode "internal" --dataset "SplitCIFAR10" --number_of_tasks "5" --model "VGG_Small" --prune_perc "60.0" --seed "0" --permute_seed "0" --deterministic "1" --optimizer "Adadelta" --sgd_momentum "0.9" --learning_rate "1.0" --batch_size "256" --batch_size_memory "256" --weight_decay "0.0" --last_layer_bn "0" --supcon_temperature "0.2" --reinit "1" --lr_decay_phase "1.0" --phase_epochs "5" --early_stop_coef "2.0" --min_activation_perc "70.0" --max_phases "15" --tau_param "30.0" --pretrain_load_path "" --pretrain_freeze "0" --verbose_logging "4"
python launch.py --experiment_name "CIFAR10_SHARP_SEED1" --gpu_prefetch "1" --ltm_per_class "50" --ltm_k_nearest "5" --stm_per_class "200" --stm_num_tasks "1" --memory_mode "internal" --dataset "SplitCIFAR10" --number_of_tasks "5" --model "VGG_Small" --prune_perc "60.0" --seed "1" --permute_seed "0" --deterministic "1" --optimizer "Adadelta" --sgd_momentum "0.9" --learning_rate "1.0" --batch_size "256" --batch_size_memory "256" --weight_decay "0.0" --last_layer_bn "0" --supcon_temperature "0.2" --reinit "1" --lr_decay_phase "1.0" --phase_epochs "5" --early_stop_coef "2.0" --min_activation_perc "70.0" --max_phases "15" --tau_param "30.0" --pretrain_load_path "" --pretrain_freeze "0" --verbose_logging "4"
python launch.py --experiment_name "CIFAR10_SHARP_SEED2" --gpu_prefetch "1" --ltm_per_class "50" --ltm_k_nearest "5" --stm_per_class "200" --stm_num_tasks "1" --memory_mode "internal" --dataset "SplitCIFAR10" --number_of_tasks "5" --model "VGG_Small" --prune_perc "60.0" --seed "2" --permute_seed "0" --deterministic "1" --optimizer "Adadelta" --sgd_momentum "0.9" --learning_rate "1.0" --batch_size "256" --batch_size_memory "256" --weight_decay "0.0" --last_layer_bn "0" --supcon_temperature "0.2" --reinit "1" --lr_decay_phase "1.0" --phase_epochs "5" --early_stop_coef "2.0" --min_activation_perc "70.0" --max_phases "15" --tau_param "30.0" --pretrain_load_path "" --pretrain_freeze "0" --verbose_logging "4"


# CIFAR100 SHARP 3 Seeds
python launch.py --experiment_name "CIFAR100_SHARP_SEED0" --gpu_prefetch "1" --ltm_per_class "50" --ltm_k_nearest "5" --stm_per_class "100" --stm_num_tasks "1" --memory_mode "internal" --dataset "SplitCIFAR100" --number_of_tasks "10" --model "VGG_Small" --prune_perc "60.0" --seed "0" --permute_seed "0" --deterministic "1" --optimizer "Adadelta" --sgd_momentum "0.9" --learning_rate "1.0" --batch_size "64" --batch_size_memory "64" --weight_decay "0.0" --last_layer_bn "0" --supcon_temperature "0.05" --reinit "1" --lr_decay_phase "1.0" --phase_epochs "5" --early_stop_coef "2.0" --min_activation_perc "70.0" --max_phases "15" --tau_param "30.0" --pretrain_load_path "" --pretrain_freeze "0" --verbose_logging "4"
python launch.py --experiment_name "CIFAR100_SHARP_SEED1" --gpu_prefetch "1" --ltm_per_class "50" --ltm_k_nearest "5" --stm_per_class "100" --stm_num_tasks "1" --memory_mode "internal" --dataset "SplitCIFAR100" --number_of_tasks "10" --model "VGG_Small" --prune_perc "60.0" --seed "1" --permute_seed "0" --deterministic "1" --optimizer "Adadelta" --sgd_momentum "0.9" --learning_rate "1.0" --batch_size "64" --batch_size_memory "64" --weight_decay "0.0" --last_layer_bn "0" --supcon_temperature "0.05" --reinit "1" --lr_decay_phase "1.0" --phase_epochs "5" --early_stop_coef "2.0" --min_activation_perc "70.0" --max_phases "15" --tau_param "30.0" --pretrain_load_path "" --pretrain_freeze "0" --verbose_logging "4"
python launch.py --experiment_name "CIFAR100_SHARP_SEED2" --gpu_prefetch "1" --ltm_per_class "50" --ltm_k_nearest "5" --stm_per_class "100" --stm_num_tasks "1" --memory_mode "internal" --dataset "SplitCIFAR100" --number_of_tasks "10" --model "VGG_Small" --prune_perc "60.0" --seed "2" --permute_seed "0" --deterministic "1" --optimizer "Adadelta" --sgd_momentum "0.9" --learning_rate "1.0" --batch_size "64" --batch_size_memory "64" --weight_decay "0.0" --last_layer_bn "0" --supcon_temperature "0.05" --reinit "1" --lr_decay_phase "1.0" --phase_epochs "5" --early_stop_coef "2.0" --min_activation_perc "70.0" --max_phases "15" --tau_param "30.0" --pretrain_load_path "" --pretrain_freeze "0" --verbose_logging "4"




