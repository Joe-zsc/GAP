import torch.nn as nn
import torch as th

import numpy as np

def _reshape_helper(tensor):
    if len(tensor.size()) == 1:
        return tensor.view(-1, 1)
    return tensor

def discount(gamma, rewards, dones, bootstrap=0.0):
    """
    ## Description

    Discounts rewards at an rate of gamma.

    ## References

    1. Sutton, Richard, and Andrew Barto. 2018. Reinforcement Learning, Second Edition. The MIT Press.

    ## Arguments

    * `gamma` (float) - Discount factor.
    * `rewards` (tensor) - Tensor of rewards.
    * `dones` (tensor) - Tensor indicating episode termination.
      Entry is 1 if the transition led to a terminal (absorbing) state, 0 else.
    * `bootstrap` (float, *optional*, default=0.0) - Bootstrap the last
      reward with this value.

    ## Returns

    * tensor - Tensor of discounted rewards.

    ## Example

    ~~~python
    rewards = th.ones(23, 1) * 8
    dones = th.zeros_like(rewards)
    dones[-1] += 1.0
    discounted = ch.rl.discount(0.99,
                                rewards,
                                dones,
                                bootstrap=1.0)
    ~~~

    """
    rewards = _reshape_helper(rewards)
    dones = _reshape_helper(dones).reshape_as(rewards)

    msg = "dones and rewards must have equal length."
    assert rewards.size(0) == dones.size(0), msg

    if not isinstance(bootstrap, (int, float)):
        bootstrap = totensor(bootstrap).reshape_as(rewards[0].unsqueeze(0))

    R = th.zeros_like(rewards) + bootstrap
    discounted = th.zeros_like(rewards)
    length = discounted.size(0)
    for t in reversed(range(length)):
        R = R * (1.0 - dones[t])
        R = rewards[t] + gamma * R
        discounted[t] += R[0]
    return discounted


def temporal_difference(gamma, rewards, dones, values, next_values):
    """
    ## Description

    Returns the temporal difference residual.

    ## Reference

    1. Sutton, Richard S. 1988. “Learning to Predict by the Methods of Temporal Differences.” Machine Learning 3 (1): 9–44.
    2. Sutton, Richard, and Andrew Barto. 2018. Reinforcement Learning, Second Edition. The MIT Press.

    ## Arguments

    * `gamma` (float) - Discount factor.
    * `rewards` (tensor) - Tensor of rewards.
    * `dones` (tensor) - Tensor indicating episode termination.
      Entry is 1 if the transition led to a terminal (absorbing) state, 0 else.
    * `values` (tensor) - Values for the states producing the rewards.
    * `next_values` (tensor) - Values of the state obtained after the
      transition from the state used to compute the last value in `values`.

    ## Example

    ~~~python
    values = vf(replay.states())
    next_values = vf(replay.next_states())
    td_errors = temporal_difference(0.99,
                                    replay.reward(),
                                    replay.done(),
                                    values,
                                    next_values)
    ~~~
    """

    values = _reshape_helper(values)
    next_values = _reshape_helper(next_values)
    rewards = _reshape_helper(rewards).reshape_as(values)
    dones = _reshape_helper(dones).reshape_as(values)

    not_dones = 1.0 - dones
    return rewards + gamma * not_dones * next_values - values


def update_module(module, updates=None, memo=None):
    r"""
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)

    **Description**

    Updates the parameters of a module in-place, in a way that preserves differentiability.

    The parameters of the module are swapped with their update values, according to:
    \[
    p \gets p + u,
    \]
    where \(p\) is the parameter, and \(u\) is its corresponding update.


    **Arguments**

    * **module** (Module) - The module to update.
    * **updates** (list, *optional*, default=None) - A list of gradients for each parameter
        of the model. If None, will use the tensors in .update attributes.

    **Example**
    ~~~python
    error = loss(model(X), y)
    grads = torch.autograd.grad(
        error,
        model.parameters(),
        create_graph=True,
    )
    updates = [-lr * g for g in grads]
    l2l.update_module(model, updates=updates)
    ~~~
    """
    if memo is None:
        memo = {}
    if updates is not None:
        params = list(module.parameters())
        if not len(updates) == len(list(params)):
            msg = "WARNING:update_module(): Parameters and updates have different length. ("
            msg += str(len(params)) + " vs " + str(len(updates)) + ")"
            print(msg)
        for p, g in zip(params, updates):
            p.update = g

    # Update the params
    for param_key in module._parameters:
        p = module._parameters[param_key]
        if p in memo:
            module._parameters[param_key] = memo[p]
        else:
            if p is not None and hasattr(p, "update") and p.update is not None:
                updated = p + p.update
                p.update = None
                memo[p] = updated
                module._parameters[param_key] = updated

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        buff = module._buffers[buffer_key]
        if buff in memo:
            module._buffers[buffer_key] = memo[buff]
        else:
            if buff is not None and hasattr(buff, "update") and buff.update is not None:
                updated = buff + buff.update
                buff.update = None
                memo[buff] = updated
                module._buffers[buffer_key] = updated

    # Then, recurse for each submodule
    for module_key in module._modules:
        module._modules[module_key] = update_module(
            module._modules[module_key],
            updates=None,
            memo=memo,
        )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(module, "flatten_parameters"):
        module._apply(lambda x: x)
    return module


def clone_module(module, memo=None):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)

    **Description**

    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.clone().

    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.

    **Arguments**

    * **module** (Module) - Module to be cloned.

    **Return**

    * (Module) - The cloned module.

    **Example**

    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate all the way to net.
    ~~~
    """
    # NOTE: This function might break in future versions of PyTorch.

    # TODO: This function might require that module.forward()
    #       was called in order to work properly, if forward() instanciates
    #       new variables.
    # NOTE: This can probably be implemented more cleanly with
    #       clone = recursive_shallow_copy(model)
    #       clone._apply(lambda t: t.clone())

    if memo is None:
        # Maps original data_ptr to the cloned tensor.
        # Useful when a Module uses parameters from another Module; see:
        # https://github.com/learnables/learn2learn/issues/174
        memo = {}

    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, "_parameters"):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    # Third, handle the buffers if necessary
    if hasattr(clone, "_buffers"):
        for buffer_key in module._buffers:
            if (
                clone._buffers[buffer_key] is not None
                and clone._buffers[buffer_key].requires_grad
            ):
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[buff_ptr] = cloned

    # Then, recurse for each submodule
    if hasattr(clone, "_modules"):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(clone, "flatten_parameters"):
        clone = clone._apply(lambda x: x)
    return clone


class LinearValue(nn.Module):
    """
    <a href="https://github.com/seba-1511/cherry/blob/master/cherry/models/robotics.py" class="source-link">[Source]</a>

    ## Description

    A linear state-value function, whose parameters are found by minimizing
    least-squares.

    ## Credit

    Adapted from Tristan Deleu's implementation.

    ## References

    1. Duan et al. 2016. “Benchmarking Deep Reinforcement Learning for Continuous Control.”
    2. [https://github.com/tristandeleu/pytorch-maml-rl](https://github.com/tristandeleu/pytorch-maml-rl)

    ## Example

    ~~~python
    states = replay.state()
    rewards = replay.reward()
    dones = replay.done()
    returns = ch.td.discount(gamma, rewards, dones)
    baseline = LinearValue(input_size)
    baseline.fit(states, returns)
    next_values = baseline(replay.next_states())
    ~~~
    """

    def __init__(self, input_size, reg=1e-5):
        """
        ## Arguments

        * `inputs_size` (int) - Size of input.
        * `reg` (float, *optional*, default=1e-5) - Regularization coefficient.
        """
        super(LinearValue, self).__init__()
        self.linear = nn.Linear(2 * input_size + 4, 1, bias=False)
        self.reg = reg

    def _features(self, states):
        length = states.size(0)
        ones = th.ones(length, 1).to(states.device)
        al = (
            th.arange(length, dtype=th.float32, device=states.device).view(-1, 1)
            / 100.0
        )
        return th.cat([states, states**2, al, al**2, al**3, ones], dim=1)

    def fit(self, states, returns):
        """
        ## Description

        Fits the parameters of the linear model by the method of least-squares.

        ## Arguments

        * `states` (tensor) - States collected with the policy to evaluate.
        * `returns` (tensor) - Returns associated with those states (ie, discounted rewards).
        """
        features = self._features(states)
        reg = self.reg * th.eye(features.size(1))
        reg = reg.to(states.device)
        A = features.t() @ features + reg
        b = features.t() @ returns
        if hasattr(th, "linalg") and hasattr(th.linalg, "lstsq"):
            coeffs = th.linalg.lstsq(A, b).solution
        elif hasattr(th, "lstsq"):  # Required for torch < 1.3.0
            coeffs, _ = th.lstsq(b, A)
        else:
            coeffs, _ = th.gels(b, A)
        self.linear.weight.data = coeffs.data.t()

    def forward(self, state):
        """
        ## Description

        Computes the value of a state using the linear function approximator.

        ## Arguments

        * `state` (Tensor) - The state to evaluate.
        """
        features = self._features(state)
        return self.linear(features)


def totensor(array, dtype=None):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/_torch.py)

    **Description**

    Converts the argument `array` to a torch.tensor 1xN, regardless of its
    type or dimension.

    **Arguments**

    * **array** (int, float, ndarray, tensor) - Data to be converted to array.
    * **dtype** (dtype, *optional*, default=None) - Data type to use for representation.
    By default, uses `torch.get_default_dtype()`.

    **Returns**

    * Tensor of shape 1xN with the appropriate data type.

    **Example**

    ~~~python
    array = [5, 6, 7.0]
    tensor = cherry.totensor(array, dtype=th.float32)
    array = np.array(array, dtype=np.float64)
    tensor = cherry.totensor(array, dtype=th.float16)
    ~~~

    """
    if dtype is None:
        dtype = th.get_default_dtype()
    if isinstance(array, (list, tuple)):
        array = th.cat([totensor(x) for x in array], dim=0)
    else:
        if isinstance(array, int):
            array = float(array)
        if isinstance(array, float):
            array = [
                array,
            ]
        if isinstance(array, list):
            array = np.array(array)
        if isinstance(
            array, (np.ndarray, np.bool_, np.float32, np.float64, np.int32, np.int64)
        ):
            if array.dtype == np.bool_:
                array = array.astype(np.uint8)
        if not isinstance(array, th.Tensor):
            array = th.tensor(array, dtype=dtype)
    while array.ndim < 2:
        array = array.unsqueeze(0)
    return array


def generalized_advantage(
    gamma,
    tau,
    rewards,
    dones,
    values,
    next_value,
):
    """
    ## Description

    Computes the generalized advantage estimator. (GAE)

    ## References

    1. Schulman et al. 2015. “High-Dimensional Continuous Control Using Generalized Advantage Estimation”
    2. https://github.com/joschu/modular_rl/blob/master/modular_rl/core.py#L49

    ## Arguments

    * `gamma` (float) - Discount factor.
    * `tau` (float) - Bias-variance trade-off.
    * `rewards` (tensor) - Tensor of rewards.
    * `dones` (tensor) - Tensor indicating episode termination.
      Entry is 1 if the transition led to a terminal (absorbing) state, 0 else.
    * `values` (tensor) - Values for the states producing the rewards.
    * `next_value` (tensor) - Value of the state obtained after the
      transition from the state used to compute the last value in `values`.

    ## Returns

    * tensor - Tensor of advantages.

    ## Example

    ~~~python
    mass, next_value = policy(replay[-1].next_state)
    advantages = generalized_advantage(0.99,
                                       0.95,
                                       replay.reward(),
                                       replay.value(),
                                       replay.done(),
                                       next_value)
    ~~~
    """

    rewards = _reshape_helper(rewards)
    dones = _reshape_helper(dones)
    values = _reshape_helper(values)
    next_value = _reshape_helper(next_value)
    next_value = totensor(next_value).reshape_as(values[0].unsqueeze(0))

    msg = "rewards, values, and dones must have equal length."
    assert len(values) == len(rewards) == len(dones), msg

    next_values = th.cat((values[1:], next_value), dim=0)
    td_errors = temporal_difference(gamma, rewards, dones, values, next_values)
    advantages = discount(tau * gamma, td_errors, dones)
    return advantages


def normalize(tensor, epsilon=1e-8):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/_torch.py)

    **Description**

    Normalizes a tensor to have zero mean and unit standard deviation values.

    **Arguments**

    * **tensor** (tensor) - The tensor to normalize.
    * **epsilon** (float, *optional*, default=1e-8) - Numerical stability constant for
    normalization.

    **Returns**

    * A new tensor, containing the normalized values.

    **Example**

    ~~~python
    tensor = torch.arange(23) / 255.0
    tensor = cherry.normalize(tensor, epsilon=1e-3)
    ~~~

    """
    if tensor.numel() <= 1:
        return tensor
    return (tensor - tensor.mean()) / (tensor.std() + epsilon)
