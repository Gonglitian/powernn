"""Tensor operations (with autograd context)"""
import numpy as np


def build_binary_ops_tensor(ts1, ts2, grad_fn_ts1, grad_fn_ts2, values):
    requires_grad = ts1.requires_grad or ts2.requires_grad
    dependency = []
    if ts1.requires_grad:
        dependency.append(dict(tensor=ts1, grad_fn=grad_fn_ts1))
    if ts2.requires_grad:
        dependency.append(dict(tensor=ts2, grad_fn=grad_fn_ts2))
    tensor_cls = ts1.__class__
    return tensor_cls(values, requires_grad, dependency)


def build_unary_ops_tensor(ts, grad_fn, values):
    requires_grad = ts.requires_grad
    dependency = []
    if ts.requires_grad:
        dependency.append(dict(tensor=ts, grad_fn=grad_fn))
    tensor_cls = ts.__class__
    return tensor_cls(values, requires_grad, dependency)


def handle_broadcasting(grad, ts):
    """
    处理tensor计算时broadcast带来的前后tensor的grad.shape不一致问题。

    正向传播时，若需要求梯度的tensor是经过broadcast生成下一个节点的，则当下一个节点反向传播时，该节点的grad.shape必与
    需要求梯度的tensor的grad.shape不一致。

    Args:
        grad (_type_): _description_
        ts (_type_): _description_
    """
    # handle broadcasting (5, 3) + (3,) -> (5, 3)
    for _ in range(grad.ndim - ts.values.ndim):
        grad = grad.sum(axis=0)
    # handle broadcasting (5, 3) + (1, 3) -> (5, 3)
    for i, dim in enumerate(ts.shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


def to_Tensor(obj):
    # avoid looping import
    from Tensor import to_Tensor
    return to_Tensor(obj)


def add_(ts1, ts2):
    """    
    c = a + b

    D_c / D_a = 1.0

    D_c / D_b = 1.0

    also need to handle broadcasting

    Args:
        ts1 (_type_): _description_
        ts2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    values = ts1.values + ts2.values

    def grad_fn_ts1(grad):
        return handle_broadcasting(grad, ts1)

    def grad_fn_ts2(grad):
        return handle_broadcasting(grad, ts2)

    return build_binary_ops_tensor(
        ts1, ts2, grad_fn_ts1, grad_fn_ts2, values)


def sub_(ts1, ts2):
    return ts1 + (-ts2)


def mul_(ts1, ts2):
    values = ts1.values * ts2.values

    # c = a * b
    # D_c / D_a = b
    # D_c / D_b = a
    def grad_fn_ts1(grad):
        grad = grad * ts2.values
        return handle_broadcasting(grad, ts1)

    def grad_fn_ts2(grad):
        grad = grad * ts1.values
        return handle_broadcasting(grad, ts2)

    return build_binary_ops_tensor(
        ts1, ts2, grad_fn_ts1, grad_fn_ts2, values)


def div_(ts1, ts2):
    values = ts1.values / ts2.values

    # c = a / b
    # D_c / D_a = 1 / b
    # D_c / D_b = -a / b**2
    def grad_fn_ts1(grad):
        grad = grad / ts2.values
        return handle_broadcasting(grad, ts1)

    def grad_fn_ts2(grad):
        grad = -grad * ts1.values / ts2.values ** 2
        return handle_broadcasting(grad, ts2)

    return build_binary_ops_tensor(
        ts1, ts2, grad_fn_ts1, grad_fn_ts2, values)


def pow_(ts1, ts2):
    values = ts1.values ** ts2.values

    # c = a ** b
    # D_c / D_a = b * a ** (b-1)
    # D_c / D_b = ln(a) * a ** b
    def grad_fn_ts1(grad):
        grad = grad * ts2.values * ts1.values ** (ts2.values - 1)
        return handle_broadcasting(grad, ts1)

    def grad_fn_ts2(grad):
        grad = grad * (np.log(ts1.values) * values)
        return handle_broadcasting(grad, ts2)

    return build_binary_ops_tensor(
        ts1, ts2, grad_fn_ts1, grad_fn_ts2, values)


def dot_(ts1, ts2):
    values = ts1.values @ ts2.values

    # c = a @ b
    # D_c / D_a = grad @ b.T
    # D_c / D_b = a.T @ grad
    def grad_fn_ts1(grad):
        return grad @ ts2.values.T

    def grad_fn_ts2(grad):
        return ts1.values.T @ grad

    return build_binary_ops_tensor(
        ts1, ts2, grad_fn_ts1, grad_fn_ts2, values)


def maximum_(ts1, ts2):
    values = np.maximum(ts1.values, ts2.values)

    def grad_fn_ts1(grad):
        grad = grad * (ts1.values >= ts2.values)
        return handle_broadcasting(grad, ts1)

    def grad_fn_ts2(grad):
        grad = grad * (ts2.values > ts1.values)
        return handle_broadcasting(grad, ts2)

    return build_binary_ops_tensor(
        ts1, ts2, grad_fn_ts1, grad_fn_ts2, values)


def minimum_(ts1, ts2):
    values = np.minimum(ts1.values, ts2.values)

    def grad_fn_ts1(grad):
        grad = grad * (ts1.values <= ts2.values)
        return handle_broadcasting(grad, ts1)

    def grad_fn_ts2(grad):
        grad = grad * (ts2.values < ts1.values)
        return handle_broadcasting(grad, ts2)

    return build_binary_ops_tensor(
        ts1, ts2, grad_fn_ts1, grad_fn_ts2, values)


def exp_(ts):
    values = np.exp(ts.values)

    def grad_fn(grad):
        return values * grad

    return build_unary_ops_tensor(ts, grad_fn, values)


def max_(ts, axis=None):
    values = np.max(ts.values, axis=axis)

    def grad_fn(grad):
        # 保留value中最大的元素的梯度
        return grad * (ts.values.max(axis=axis, keepdims=1) == ts.values)

    return build_unary_ops_tensor(ts, grad_fn, values)


def min_(ts, axis=None):
    values = np.min(ts.values, axis=axis)

    def grad_fn(grad):
        # 保留value中最小的元素的梯度
        return grad * (ts.values.min(axis=axis, keepdims=1) == ts.values)

    return build_unary_ops_tensor(ts, grad_fn, values)


def log_(ts):
    values = np.log(ts.values)

    def grad_fn(grad):
        return grad / ts.values

    return build_unary_ops_tensor(ts, grad_fn, values)


def sum_(ts, axis):
    values = ts.values.sum(axis=axis)
    if axis is not None:
        repeat = ts.values.shape[axis]

    def grad_fn(grad):
        if axis is None:
            # 默认对所有元素求和，恢复梯度形状
            grad = grad * np.ones_like(ts.values)
        else:
            # 沿指定维度恢复梯度形状
            grad = np.expand_dims(grad, axis)
            grad = np.repeat(grad, repeat, axis)
        return grad

    return build_unary_ops_tensor(ts, grad_fn, values)


def transpose_(ts, axes=None):
    values = ts.values.transpose(axes)

    # 默认所有轴反转
    if axes is None:
        axes = reversed(range(ts.values.ndim))
    axes = list(axes)

    # recover to original shape
    def grad_fn(grad):
        return grad.transpose(np.argsort(axes))

    return build_unary_ops_tensor(ts, grad_fn, values)


def getitem_(ts, key):
    values = ts.values[key]

    def grad_fn(grad):
        recover_grad = np.zeros_like(ts.values)
        recover_grad[key] = grad
        return recover_grad

    return build_unary_ops_tensor(ts, grad_fn, values)


def neg_(ts):
    values = -ts.values

    def grad_fn(grad):
        return -grad

    return build_unary_ops_tensor(ts, grad_fn, values)


def reshape_(ts, newshape):
    shape = ts.values.shape
    values = ts.values.reshape(newshape)

    def grad_fn(grad):
        return grad.reshape(shape)

    return build_unary_ops_tensor(ts, grad_fn, values)


def pad_(ts, pad_width, mode):
    values = np.pad(ts.values, pad_width=pad_width, mode=mode)
    slices = list()
    for size, (before, after) in zip(values.shape, pad_width):
        print(before, after)
        slices.append(slice(before, size-after))

    def grad_fn(grad):
        return grad[tuple(slices)]

    return build_unary_ops_tensor(ts, grad_fn, values)


def flatten_(ts):
    shape = ts.shape
    values = ts.values.ravel()

    def grad_fn(grad):
        return grad.reshape(shape)
    return build_unary_ops_tensor(ts, grad_fn, values)


def clip_(ts, min, max):
    # 清零被整流的元素的梯度
    values = ts.values.clip(min, max)
    mask = np.ones(ts.shape, dtype=bool)
    if min is not None:
        mask &= ts.values >= min
    if max is not None:
        mask &= ts.values <= max

    def grad_fn(grad):
        return grad * mask
    return build_unary_ops_tensor(ts, grad_fn, values)


def max(obj, axis=None):
    return max_(to_Tensor(obj), axis=axis)


def maximum(obj1, obj2):
    return maximum_(to_Tensor(obj1), to_Tensor(obj2))


def minimum(obj1, obj2):
    return minimum_(to_Tensor(obj1), to_Tensor(obj2))


def exp(obj):
    return exp_(to_Tensor(obj))


def sum(obj, axis=None):
    return sum_(to_Tensor(obj), axis=axis)


def log(obj):
    return log_(to_Tensor(obj))


def reshape(obj, newshape):
    return reshape_(to_Tensor(obj), newshape)


def pad(obj, pad_width, mode = "constant"):
    return pad_(to_Tensor(obj), pad_width, mode=mode)


def flatten(obj):
    return flatten_(to_Tensor(obj))


def clip(obj, min=None, max=None):
    return clip_(to_Tensor(obj), min, max)
