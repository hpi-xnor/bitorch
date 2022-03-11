Module bitorch.layers.debug_layers
==================================

Classes
-------

`InputGraphicalDebug(figure: object = None, images: list = None, debug_interval: int = 100, num_outputs: int = 10)`
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
    
    Debugs the given layer by drawing weights/inputs in given matplotlib plot images.
    
    Args:
        figure (object): figure to draw in
        images (list): list of images to update with given data
        debug_interval (int, optional): interval at which debug output shall be prompted. Defaults to 100.
        num_outputs (int, optional): number of weights/inputs that shall be debugged. Defaults to 10.
    
    Raises:
        ValueError: raised if number of images does not match desired number of outputs.

    ### Ancestors (in MRO)

    * bitorch.layers.debug_layers._GraphicalDebug
    * bitorch.layers.debug_layers._Debug
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   forwards the given tensor without modification, debug output if activated
        
        Args:
            x (torch.Tensor): tensor to be debugged
        
        Returns:
            torch.Tensor: input tensor x

`InputPrintDebug(debug_interval: int = 100, num_outputs: int = 10, name: str = 'Debug')`
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
    
    inits values.
    
    Args:
        debug_interval (int, optional): interval at which debug output shall be prompted. Defaults to 100.
        num_outputs (int, optional): number of weights/inputs that shall be debugged. Defaults to 10.
        name (str, optional): name of debug layer, only relevant for print debugging

    ### Ancestors (in MRO)

    * bitorch.layers.debug_layers._PrintDebug
    * bitorch.layers.debug_layers._Debug
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   forwards the given tensor without modification, debug output if activated
        
        Args:
            x (torch.Tensor): tensor to be debugged
        
        Returns:
            torch.Tensor: input tensor x

`ShapePrintDebug(debug_interval: int = 100, num_outputs: int = 10, name: str = 'Debug')`
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
    
    inits values.
    
    Args:
        debug_interval (int, optional): interval at which debug output shall be prompted. Defaults to 100.
        num_outputs (int, optional): number of weights/inputs that shall be debugged. Defaults to 10.
        name (str, optional): name of debug layer, only relevant for print debugging

    ### Ancestors (in MRO)

    * bitorch.layers.debug_layers._PrintDebug
    * bitorch.layers.debug_layers._Debug
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   prints the shape of x, leaves x untouched
        
        Args:
            x (torch.Tensor): the tensor to be debugged
        
        Returns:
            torch.Tensor: input tensor x

`WeightGraphicalDebug(module: torch.nn.modules.module.Module, *args: Any, **kwargs: Any)`
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
    
    stores given module
    
    Args:
        module (torch.nn.Module): module the weights of which shall be debugged

    ### Ancestors (in MRO)

    * bitorch.layers.debug_layers._GraphicalDebug
    * bitorch.layers.debug_layers._Debug
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Forwards the input tensor through the debug model and outputs debug information about the given modules weights.
        
        Args:
            x (torch.Tensor): the Tensor to be forwarded untouched.
        
        Returns:
            torch.Tensor: the input tensor

`WeightPrintDebug(module: torch.nn.modules.module.Module, *args: Any, **kwargs: Any)`
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
    
    stores given module
    
    Args:
        module (torch.nn.Module): module the weights of which shall be debugged

    ### Ancestors (in MRO)

    * bitorch.layers.debug_layers._PrintDebug
    * bitorch.layers.debug_layers._Debug
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Forwards the input tensor through the debug model and outputs debug information about the given modules weights.
        
        Args:
            x (torch.Tensor): the Tensor to be forwarded untouched.
        
        Returns:
            torch.Tensor: the input tensor