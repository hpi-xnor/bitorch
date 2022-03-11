Module bitorch.models
=====================

Sub-modules
-----------
* bitorch.models.base
* bitorch.models.common_layers
* bitorch.models.lenet
* bitorch.models.resnet
* bitorch.models.resnet_e

Classes
-------

`LeNet(dataset: bitorch.datasets.base.BasicDataset, lenet_quantized: bool = False)`
:   LeNet model, both in quantized and full precision version
    
    builds the model, depending on mode in either quantized or full_precision mode
    
    Args:
        lenet_quantized (bool, optional): toggles use of quantized version of lenet. Default is False.

    ### Ancestors (in MRO)

    * bitorch.models.base.Model
    * torch.nn.modules.module.Module

    ### Class variables

    `activation_function`
    :   Applies the element-wise function:
        
        .. math::
            \text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}
        
        Shape:
            - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
            - Output: :math:`(*)`, same shape as the input.
        
        .. image:: ../scripts/activation_images/Tanh.png
        
        Examples::
        
            >>> m = nn.Tanh()
            >>> input = torch.randn(2)
            >>> output = m(input)

    `dump_patches: bool`
    :

    `name`
    :

    `num_channels_conv`
    :

    `num_fc`
    :

    `training: bool`
    :

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

`Resnet(resnet_version: int, resnet_num_layers: int, dataset: bitorch.datasets.base.BasicDataset)`
:   Base class for Bitorch models
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * bitorch.models.base.Model
    * torch.nn.modules.module.Module

    ### Descendants

    * bitorch.models.resnet.Resnet152V1
    * bitorch.models.resnet.Resnet152V2
    * bitorch.models.resnet.Resnet18V1
    * bitorch.models.resnet.Resnet18V2
    * bitorch.models.resnet.Resnet34V1
    * bitorch.models.resnet.Resnet34V2
    * bitorch.models.resnet.Resnet50V1
    * bitorch.models.resnet.Resnet50V2

    ### Class variables

    `dump_patches: bool`
    :

    `name`
    :

    `resnet_block_versions`
    :

    `resnet_net_versions`
    :

    `resnet_spec`
    :

    `training: bool`
    :

    ### Methods

    `create_resnet(self, version: int, num_layers: int, classes: int = 1000, initial_layers: str = 'imagenet', image_channels: int = 3) ‑> torch.nn.modules.module.Module`
    :   Creates a resnet complying to given version and layer number.
        
        Args:
            version (int): version of resnet to be used. availavle versions are 1 or 2
            num_layers (int): number of layers to be build.
            classes (int, optional): number of output classes. Defaults to 1000.
            initial_layers (str, optional): name of set of initial layers to be used. Defaults to "imagenet".
            image_channels (int, optional): number of channels of input images. Defaults to 3.
        
        Raises:
            ValueError: raised if no resnet specification for given num_layers is listed in the resnet_spec dict above
            ValueError: raised if invalid resnet version was passed
        
        Returns:
            Module: resnet model

`Resnet152V1(*args: Any, **kwargs: Any)`
:   ResNet-152 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * bitorch.models.resnet.Resnet
    * bitorch.models.base.Model
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `name`
    :

    `training: bool`
    :

`Resnet152V2(*args: Any, **kwargs: Any)`
:   ResNet-152 V2 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * bitorch.models.resnet.Resnet
    * bitorch.models.base.Model
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `name`
    :

    `training: bool`
    :

`Resnet18V1(*args: Any, **kwargs: Any)`
:   ResNet-18 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * bitorch.models.resnet.Resnet
    * bitorch.models.base.Model
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `name`
    :

    `training: bool`
    :

`Resnet18V2(*args: Any, **kwargs: Any)`
:   ResNet-18 V2 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * bitorch.models.resnet.Resnet
    * bitorch.models.base.Model
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `name`
    :

    `training: bool`
    :

`Resnet34V1(*args: Any, **kwargs: Any)`
:   ResNet-34 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * bitorch.models.resnet.Resnet
    * bitorch.models.base.Model
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `name`
    :

    `training: bool`
    :

`Resnet34V2(*args: Any, **kwargs: Any)`
:   ResNet-34 V2 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * bitorch.models.resnet.Resnet
    * bitorch.models.base.Model
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `name`
    :

    `training: bool`
    :

`Resnet50V1(*args: Any, **kwargs: Any)`
:   ResNet-50 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * bitorch.models.resnet.Resnet
    * bitorch.models.base.Model
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `name`
    :

    `training: bool`
    :

`Resnet50V2(*args: Any, **kwargs: Any)`
:   ResNet-50 V2 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * bitorch.models.resnet.Resnet
    * bitorch.models.base.Model
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `name`
    :

    `training: bool`
    :

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

`ResnetE18(*args: Any, **kwargs: Any)`
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

`ResnetE34(*args: Any, **kwargs: Any)`
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