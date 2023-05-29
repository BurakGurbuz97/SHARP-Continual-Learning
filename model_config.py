from torch import nn
from custom_layers import BatchNorm1Custom, BatchNorm2Custom



MNISTLIKE = {
    "conv_early": {
        "ops": [(nn.ReLU(), nn.MaxPool2d(2, 2)),
                (nn.ReLU(), nn.MaxPool2d(2, 2))],
        # (out_channels, kernel_size, stride)
        "layers": [(16, 3, 1),
                   (16, 3, 1)],
    },

    "conv_late": {
        "ops": [],
        # (out_channels, kernel_size)
        "layers": [],

        # Number of flat features after last conv layer.
        "conv2lin_size": 16 * 7*7,
        "conv2lin_mapping_size": 7*7
    },

    "mlp_hidden_linear": {
        "ops": [(nn.ReLU(), ),],
        "layers": [500]
    },

    "penultimate_layer_size":  500,
}


VGG_Small = {

    "conv_early": {
        "ops": [(nn.ReLU(), ),
                (nn.ReLU(), nn.MaxPool2d(2)),
                (nn.ReLU(), ),
                (nn.ReLU(), nn.MaxPool2d(2)),
                ],
        # (out_channels, kernel_size, stride)
        "layers": [(64, 3, 1),
                   (64, 3, 1),
                   (64, 3, 1),
                   (64, 3, 1),],
    },

    "conv_late": {
        "ops": [(nn.ReLU(), ),
                (nn.ReLU(), ),
                (nn.ReLU(), nn.MaxPool2d(2))],
        # (out_channels, kernel_size, stride)
        "layers": [(128, 3, 1),
                   (128, 3, 1),
                   (128, 3, 1)],

        # Number of flat features after last conv layer.
        "conv2lin_size": 128 * 4*4,
        "conv2lin_mapping_size": 4*4
    },

    "mlp_hidden_linear": {
        "ops": [(nn.ReLU(), )],
        "layers": [1024]
    },

    "penultimate_layer_size":  1024
}


VGG_Small_Double = {

    "conv_early": {
        "ops": [(nn.ReLU(), ),
                (nn.ReLU(), nn.MaxPool2d(2)),
                (nn.ReLU(), ),
                (nn.ReLU(), nn.MaxPool2d(2)),
                ],
        # (out_channels, kernel_size, stride)
        "layers": [(64, 3, 1),
                   (64, 3, 1),
                   (64, 3, 1),
                   (64, 3, 1),],
    },

    "conv_late": {
        "ops": [(nn.ReLU(), ),
                (nn.ReLU(), ),
                (nn.ReLU(), nn.MaxPool2d(2))],
        # (out_channels, kernel_size, stride)
        "layers": [(128, 3, 1),
                   (128, 3, 1),
                   (128, 3, 1)],

        # Number of flat features after last conv layer.
        "conv2lin_size": 128 * 4*4,
        "conv2lin_mapping_size": 4*4
    },

    "mlp_hidden_linear": {
        "ops": [(nn.ReLU(), )],
        "layers": [2048]
    },

    "penultimate_layer_size":  2048
}


CONV_BIR = {
     "conv_early": {
        "ops": [(nn.ReLU(), ),
                (nn.ReLU(), ),
                (nn.ReLU(), ),
                (nn.ReLU(), ),
                (nn.ReLU(), ),
                ],
        # (out_channels, kernel_size, stride)
        "layers": [(16, 3, 1),
                   (32, 3, 2),
                   (64, 3, 2),
                   (128, 3, 2),
                   (256, 3, 2)],
    },

    "conv_late": {
        "ops": [],
        # (out_channels, kernel_size)
        "layers": [],

        # Number of flat features after last conv layer.
        "conv2lin_size": 256 * 2 * 2,
        "conv2lin_mapping_size": 2 * 2
    },

    "mlp_hidden_linear": {
        "ops": [(nn.ReLU(), ), (nn.ReLU(), )],
        "layers": [2000, 2000]
    },

    "penultimate_layer_size":  2000


}