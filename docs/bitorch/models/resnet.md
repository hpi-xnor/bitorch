Module bitorch.models.resnet
============================

Classes
-------

`BasicBlockV1(in_channels: int, out_channels: int, stride: int)`
:   BasicBlock V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 18, 34 layers.
    
    builds body and downsampling layers
    
    Args:
        in_channels (int): input channels for building block
        out_channels (int): output channels for building block
        stride (int): stride to use in convolutions

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   forwards the input tensor x through the building block.
        
        Args:
            x (torch.Tensor): the input tensor
        
        Returns:
            torch.Tensor: the output of this building block after adding the input tensor as residual value.

`BasicBlockV2(in_channels: int, out_channels: int, stride: int)`
:   BasicBlock V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 18, 34 layers.
    
    builds body and downsampling layers
    
    Args:
        in_channels (int): input channels for building block
        out_channels (int): output channels for building block
        stride (int): stride to use in convolutions

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   forwards the input tensor x through the building block.
        
        Args:
            x (torch.Tensor): the input tensor
        
        Returns:
            torch.Tensor: the output of this building block after adding the input tensor as residual value.

`BottleneckV1(in_channels: int, out_channels: int, stride: int)`
:   Bottleneck V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 50, 101, 152 layers.
    
    builds body and downsampling layers
    
    Args:
        in_channels (int): input channels for building block
        out_channels (int): output channels for building block
        stride (int): stride to use in convolutions

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   forwards the input tensor x through the building block.
        
        Args:
            x (torch.Tensor): the input tensor
        
        Returns:
            torch.Tensor: the output of this building block after adding the input tensor as residual value.

`BottleneckV2(in_channels: int, out_channels: int, stride: int)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    builds body and downsampling layers
    
    Args:
        in_channels (int): input channels for building block
        out_channels (int): output channels for building block
        stride (int): stride to use in convolutions

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   forwards the input tensor x through the building block.
        
        Args:
            x (torch.Tensor): the input tensor
        
        Returns:
            torch.Tensor: the output of this building block after adding the input tensor as residual value.

`ResNetV1(block: torch.nn.modules.module.Module, layers: list, channels: list, classes: int, initial_layers: str = 'imagenet', image_channels: int = 3)`
:   ResNet V1 model from
    `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    
    Creates ResNetV1 model.
    
    Args:
        block (Module): Block to be used for building the layers.
        layers (list): layer sizes
        channels (list): channel num used for input/output channel size of layers. there must always be one more
            channels than there are layers.
        classes (int): number of output classes
        initial_layers (str, optional): name of set for initial layers. refer to common_layers.py.
            Defaults to "imagenet".
        image_channels (int, optional): input channels of images. Defaults to 3.
    
    Raises:
        ValueError: raised if the number of channels does not match number of layer + 1

    ### Ancestors (in MRO)

    * bitorch.models.resnet.SpecificResnet
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

`ResNetV2(block: torch.nn.modules.module.Module, layers: list, channels: list, classes: int = 1000, initial_layers: str = 'imagenet', image_channels: int = 3)`
:   ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    
    Creates ResNetV2 model.
    
    Args:
        block (Module): Block to be used for building the layers.
        layers (list): layer sizes
        channels (list): channel num used for input/output channel size of layers. there must always be one more
            channels than there are layers.
        classes (int): number of output classes
        initial_layers (str, optional): name of set for initial layers. refer to common_layers.py.
            Defaults to "imagenet".
        image_channels (int, optional): input channels of images. Defaults to 3.
    
    Raises:
        ValueError: raised if the number of channels does not match number of layer + 1

    ### Ancestors (in MRO)

    * bitorch.models.resnet.SpecificResnet
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

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

`SpecificResnet(classes: int, channels: list)`
:   Superclass for ResNet models
    
    builds feature and output layers
    
    Args:
        classes (int): number of output classes
        channels (list): the channels used in the net

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Descendants

    * bitorch.models.resnet.ResNetV1
    * bitorch.models.resnet.ResNetV2

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   forwards the input tensor through the resnet modules
        
        Args:
            x (torch.Tensor): input tensor
        
        Returns:
            torch.Tensor: forwarded tensor

    `make_feature_layers(self, block: torch.nn.modules.module.Module, layers: list, channels: list) ‑> List[torch.nn.modules.module.Module]`
    :   builds the given layers with the specified block.
        
        Args:
            block (Module): the block of which the layer shall consist
            layers (list): the number of blocks each layer shall consist of
            channels (list): the channels
        
        Returns:
            nn.Sequential: [description]

    `make_layer(self, block: torch.nn.modules.module.Module, layers: int, in_channels: int, out_channels: int, stride: int) ‑> torch.nn.modules.container.Sequential`
    :   builds a layer by stacking blocks in a sequential models.
        
        Args:
            block (Module): the block of which the layer shall consist
            layers (int): the number of blocks to stack
            in_channels (int): the input channels of this layer
            out_channels (int): the output channels of this layer
            stride (int): the stride to be used in the convolution layers
        
        Returns:
            nn.Sequential: the model containing the building blocks