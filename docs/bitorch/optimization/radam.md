Module bitorch.optimization.radam
=================================
RAdam implementation copied from https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam/radam.py.

It has been proposed in `On the Variance of the Adaptive Learning Rate and Beyond`.
https://arxiv.org/abs/1908.03265

Classes
-------

`AdamW(params: Union[Iterable[torch.Tensor], Iterable[Dict[Any, Any]]], lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0, warmup: int = 0)`
:   Base class for all optimizers.
    
    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.
    
    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Methods

    `step(self, closure: Callable = None) ‑> Optional[float]`
    :   Performs a single optimization step (parameter update).
        
        Args:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        
        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.

`PlainRAdam(params: Union[Iterable[torch.Tensor], Iterable[Dict[Any, Any]]], lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0, degenerated_to_sgd: bool = True)`
:   Base class for all optimizers.
    
    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.
    
    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Methods

    `step(self, closure: Callable = None) ‑> Optional[float]`
    :   Performs a single optimization step (parameter update).
        
        Args:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        
        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.

`RAdam(params: Union[Iterable[torch.Tensor], Iterable[Dict[Any, Any]]], lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0, degenerated_to_sgd: bool = True)`
:   Base class for all optimizers.
    
    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.
    
    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    
    Initialises RAdam optimizer
    
    Args:
        params (Union[Iterable[Tensor], Iterable[Dict[Any, Any]]]): iterable of parameters to optimize or dicts
            defining parameter groups
        lr (float, optional): learning range. Defaults to 1e-3.
        betas (Tuple[float, float], optional): coefficients used for computing running averages of gradient and its
            square. Defaults to (0.9, 0.999).
        eps (float, optional): term added to the denominator to improve numerical stability. Defaults to 1e-8.
        weight_decay (float, optional): weight decay (L2 penality). Defaults to 0.
        degenerated_to_sgd (bool, optional): toggles wether to use sgd step. Defaults to True.
    
    Raises:
        ValueError: thrown if lr <= 0.0
        ValueError: thrown if eps <= 0.0
        ValueError: thrown if first beta value <= 0
        ValueError: thrown if second beta value <= 0

    ### Ancestors (in MRO)

    * torch.optim.optimizer.Optimizer

    ### Methods

    `step(self, closure: Callable = None) ‑> Optional[float]`
    :   Performs a single optimization step (parameter update).
        
        Args:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        
        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.