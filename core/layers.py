"""Network layers and activation layers."""

import numpy as np

import core.ops as ops
from core.initializer import XavierUniformInit
from core.initializer import ZerosInit


class Layer(object):

    def __init__(self, name):
        self.name = name

        self.params, self.grads = {}, {}
        self.is_training = True

    # note:override this in your layer
    def forward(self, inputs):
        raise NotImplementedError

    def set_phase(self, phase):
        self.is_training = True if phase == "TRAIN" else False

    # todo pprint  parameters of layer
    # note:you can override this
    def __repr__(self) -> str:
        return f""


class Dense(Layer):

    def __init__(self,
                 num_out,
                 num_in=None,
                 w_init=XavierUniformInit(),
                 b_init=ZerosInit()):
        super().__init__("Linear")

        # note:bind initialization function for this layer
        self.initializers = {"w": w_init, "b": b_init}
        self.shapes = {"w": [num_in, num_out], "b": [1, num_out]}
        self.params = {"w": None, "b": None}

        self.is_init = False
        if num_in is not None:
            self._init_parameters(num_in)

        self.inputs = None

    def forward(self, inputs):
        # note:initialize parameters denpending on shape of inputs while this layer hasn't been initialized
        if not self.is_init:
            self._init_parameters(inputs.shape[1])

        self.inputs = inputs
        # todo:change this
        return inputs @ self.params["w"] + self.params["b"]

    # todo
    def _init_parameters(self, input_size):
        self.shapes["w"][0] = input_size

        self.params["w"] = self.initializers["w"](shape=self.shapes["w"])
        self.params["b"] = self.initializers["b"](shape=self.shapes["b"])

        self.params["w"].zero_grad()
        self.params["b"].zero_grad()
        self.is_init = True


class Softmax(Layer):

    def __init__(self,
                 num_in=None,
                 ):
        super().__init__("Softmax")
        self.params = {"w": None, "b": None}
        self.inputs = None

    def forward(self, inputs):
        return ops.exp(inputs) / ops.sum(ops.exp(inputs), axis=0)


class Activation(Layer):

    def __init__(self, name):
        super().__init__(name)
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return self.func(inputs)

    def func(self, x):
        raise NotImplementedError


class Sigmoid(Activation):

    def __init__(self):
        super().__init__("Sigmoid")

    def func(self, x):
        return 1.0 / (1.0 + ops.exp(-x))


class Tanh(Activation):

    def __init__(self):
        super().__init__("Tanh")

    def func(self, x):
        return (1.0 - ops.exp(-x)) / (1.0 + ops.exp(-x))


class ReLU(Activation):

    def __init__(self):
        super().__init__("ReLU")

    def func(self, x):
        return ops.clip(x, 0.0)
