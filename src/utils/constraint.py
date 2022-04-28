"""
Reference code: https://github.com/eelcovdw/pytorch-constrained-opt/
"""
import torch


class Clamp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, min=0, max=100):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class Constraint(torch.nn.Module):
    def __init__(self, bound, relation, name=None, start_val=0.):
        """
        Adds a constraint to a loss function by turning the loss into a lagrangian.
        Alpha is used for a moving average as described in [1].
        Note that this is similar as using an optimizer with momentum.
        [1] Rezende, Danilo Jimenez, and Fabio Viola.
            "Taming vaes." arXiv preprint arXiv:1810.00597 (2018).
        Args:
            bound: Constraint bound.
            relation (str): relation of constraint,
                using naming convention from operator module (eq, le, ge).
                Defaults to 'ge'.
            name (str, optional): Constraint name
            start_val (float, optional): Start value of multiplier. If an activation function
                is used the true start value might be different, because this is pre-activation.
        """

        super().__init__()
        self.name = name
        if isinstance(bound, (int, float)):
            self.bound = torch.Tensor([bound])
        elif isinstance(bound, list):
            self.bound = torch.Tensor(bound)
        else:
            self.bound = bound

        if relation in {'ge', 'le', 'eq'}:
            self.relation = relation
        else:
            raise ValueError('Unknown relation: {}'.format(relation))

        if torch.cuda.is_available():
            self._multiplier = torch.nn.Parameter(
                torch.full((len(self.bound),), start_val).cuda() if torch.cuda.is_available() else torch.full((len(self.bound),), start_val).cuda()
            )
            self.bound = self.bound.cuda()
        else:
            self._multiplier = torch.nn.Parameter(
                torch.full((len(self.bound),), start_val).cuda() if torch.cuda.is_available() else torch.full((len(self.bound),), start_val)
            )

        self.clamp = Clamp()

    @property
    def multiplier(self):

        return self.clamp.apply(self._multiplier)

    def forward(self, value):
        # Apply moving average, defined in [1]

        if self.relation in {'ge', 'eq'}:
            loss = self.bound.to(value.device) - value
        elif self.relation == 'le':
            loss = (value/self.bound.to(value.device)) - 1

        return loss * self.multiplier.to(value.device)


class Wrapper:
    """
    Simple class wrapper around  obj = obj_type(*args, **kwargs).
    Overwrites methods from obj with methods defined in Wrapper,
    else uses method from obj.
    """
    def __init__(self, obj_type, *args, **kwargs):
        self.obj = obj_type(*args, **kwargs)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.obj, attr)

    def __repr__(self):
        return 'Wrapped(' + self.obj.__repr__() + ')'


class ConstraintOptimizer(Wrapper):
    """
    Pytorch Optimizers only do gradient descent, but lagrangian needs
    gradient ascent for the multipliers. ConstraintOptimizer changes
    step() method of optimizer to do ascent instead of descent.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, *args, **kwargs):
        super().__init__(optimizer, *args, **kwargs)

    def step(self, *args, **kwargs):

        # Maximize instead of minimize.

        for group in self.obj.param_groups:
            for p in group['params']:
                p.grad = -p.grad

        self.obj.step(*args, **kwargs)

    def __repr__(self):
        return 'ConstraintOptimizer (' + self.obj.__repr__() + ')'
