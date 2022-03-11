Module bitorch.models.base
==========================

Classes
-------

`Model(dataset: Union[bitorch.datasets.base.BasicDataset, Type[bitorch.datasets.base.BasicDataset]])`
:   Base class for Bitorch models
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Descendants

    * bitorch.models.lenet.LeNet
    * bitorch.models.resnet.Resnet
    * bitorch.models.resnet_e.ResnetE

    ### Class variables

    `dump_patches: bool`
    :

    `name`
    :

    `training: bool`
    :

    ### Static methods

    `add_argparse_arguments(parser: argparse.ArgumentParser) ‑> None`
    :   allows additions to the argument parser if required, e.g. to add layer count, etc.
        
        ! please note that the inferred variable names of additional cli arguments are passed as
        keyword arguments to the constructor of this class !
        
        Args:
            parser (ArgumentParser): the argument parser

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   forwards the input tensor through the model.
        
        Args:
            x (torch.Tensor): input tensor
        
        Returns:
            torch.Tensor: the model output

    `initialize(self) ‑> None`
    :   initializes model weights a little differently for BNNs.

    `model(self) ‑> torch.nn.modules.module.Module`
    :   getter method for model
        
        Returns:
            Module: the main torch.nn.Module of this model