import torch


def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu' 

class BatchNorm1Custom(torch.nn.BatchNorm1d):
    def __init__(self, num_features, layer_index, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm1Custom, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.frozen_units = None
        self.layer_index = layer_index
        self.running_mean_frozen = None
        self.running_var_frozen = None

    def freeze_units(self, frozen_units):
        frozen_units = frozen_units.bool().to(get_device())
        if self.frozen_units is None:
            self.frozen_units = frozen_units
            self.running_mean_frozen = self.running_mean.data.clone()  # type: ignore 
            self.running_var_frozen = self.running_var.data.clone()# type: ignore 
        else:
            new_frozen_units = torch.logical_xor(self.frozen_units, frozen_units)
            self.running_mean_frozen = torch.where(new_frozen_units, self.running_mean.data, self.running_mean_frozen)  # type: ignore 
            self.running_var_frozen = torch.where(new_frozen_units, self.running_var.data, self.running_var_frozen)  # type: ignore 
            self.frozen_units = frozen_units


    def forward(self, input):
        if self.frozen_units  is not  None:
            # Replace the frozen dimensions in self.running_mean and running_var
            self.running_mean.data = torch.where(self.frozen_units,self.running_mean_frozen, self.running_mean.data) # type: ignore 
            self.running_var.data = torch.where(self.frozen_units, self.running_var_frozen, self.running_var.data) # type: ignore 

        # Call the forward method of the parent class
        return super(BatchNorm1Custom, self).forward(input)


class BatchNorm2Custom(torch.nn.BatchNorm2d):
    def __init__(self, num_features, layer_index, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm2Custom, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.frozen_units = None
        self.layer_index = layer_index
        self.running_mean_frozen = None
        self.running_var_frozen = None

    def freeze_units(self, frozen_units):
        frozen_units = frozen_units.bool().to(get_device())
        if self.frozen_units is None:
            self.frozen_units = frozen_units
            self.running_mean_frozen = self.running_mean.data.clone()  # type: ignore 
            self.running_var_frozen = self.running_var.data.clone()  # type: ignore 
        else:
            new_frozen_units = torch.logical_xor(self.frozen_units, frozen_units)
            self.running_mean_frozen = torch.where(new_frozen_units, self.running_mean.data, self.running_mean_frozen)   # type: ignore 
            self.running_var_frozen = torch.where(new_frozen_units, self.running_var.data, self.running_var_frozen)  # type: ignore 
            self.frozen_units = frozen_units

    def forward(self, input):
        if self.frozen_units is not None:
            # Replace the frozen dimensions in self.running_mean and running_var
            self.running_mean.data = torch.where(self.frozen_units, self.running_mean_frozen, self.running_mean.data)  # type: ignore 
            self.running_var.data = torch.where(self.frozen_units, self.running_var_frozen, self.running_var.data)  # type: ignore 

        # Call the forward method of the parent class
        return super(BatchNorm2Custom, self).forward(input)