import os
import torch
from launch_utils import get_argument_parser, get_experience_streams, get_model_config_dict
from launch_utils import set_seeds, get_log_param_dict, create_log_dirs
from Source import architecture, learner


if __name__ == '__main__':
    args = get_argument_parser()
    log_params = get_log_param_dict(args)
    create_log_dirs(args, log_params)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if args.deterministic: set_seeds(args.seed)
    scenario, input_size, output_size, task2classes = get_experience_streams(args)
    config_dict = get_model_config_dict(args)
    network = architecture.Network(config_dict, input_size, output_size, args)
    network = learner.Learner(args, network, scenario, log_params,
                              config_dict, input_size, task2classes)
    network.learn_all_episodes()