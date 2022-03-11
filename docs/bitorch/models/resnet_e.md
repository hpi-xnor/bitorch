Module bitorch.models.resnet_e
==============================
Resnet_E implementation from `"Back to Simplicity: How to Train Accurate BNNs from Scratch?"
<https://arxiv.org/abs/1906.08637>`_ paper.

Classes
-------

`ResnetE(resnete_num_layers: int, dataset: bitorch.datasets.base.BasicDataset)`
:   Base class for Bitorch models
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * bitorch.models.base.Model
    * torch.nn.modules.module.Module

    ### Descendants

    * bitorch.models.resnet_e.ResnetE18
    * bitorch.models.resnet_e.ResnetE34

    ### Class variables

    `dump_patches: bool`
    :

    `name`
    :

    `resnet_spec`
    :

    `training: bool`
    :

    ### Static methods

    `create(num_layers: int, classes: int = 1000, initial_layers: str = 'imagenet', image_channels: int = 3) ‑> torch.nn.modules.module.Module`
    :   Creates a ResNetE complying to given layer number.
        
        Args:
            num_layers (int): number of layers to be build.
            classes (int, optional): number of output classes. Defaults to 1000.
            initial_layers (str, optional): name of set of initial layers to be used. Defaults to "imagenet".
            image_channels (int, optional): number of channels of input images. Defaults to 3.
        
        Raises:
            ValueError: raised if no resnet specification for given num_layers is listed in the resnet_spec dict above
        
        Returns:
            Module: resnetE model

`ResnetE18(*args, **kwargs)`
:   ResNetE-18 model from `"Back to Simplicity: How to Train Accurate BNNs from Scratch?"
    <https://arxiv.org/abs/1906.08637>`_ paper.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * bitorch.models.resnet_e.ResnetE
    * bitorch.models.base.Model
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `name`
    :

    `training: bool`
    :

`ResnetE34(*args, **kwargs)`
:   ResNetE-34 model from `"Back to Simplicity: How to Train Accurate BNNs from Scratch?"
    <https://arxiv.org/abs/1906.08637>`_ paper.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * bitorch.models.resnet_e.ResnetE
    * bitorch.models.base.Model
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `name`
    :

    `training: bool`
    :