"""Network layers and activation layers."""

import numpy as np

import ops as ops
from initializer import XavierUniformInit
from initializer import ZerosInit


class Layer(object):

    def __init__(self, name):
        self.name = name

        self.params, self.grads = {}, {}
        self.is_training = True

    def forward(self, inputs):
        raise NotImplementedError

    def set_phase(self, phase):
        self.is_training = True if phase == "TRAIN" else False

    def __repr__(self) -> str:
        return f""
class Dense(Layer):

    def __init__(self,
                 num_out,
                 num_in=None,
                 w_init=XavierUniformInit(),
                 b_init=ZerosInit()):
        super().__init__("Linear")
        self.initializers = {"w": w_init, "b": b_init} #为参数绑定初始化函数
        self.shapes = {"w": [num_in, num_out], "b": [1, num_out]}
        self.params = {"w": None, "b": None}

        self.is_init = False
        if num_in is not None:
            self._init_parameters(num_in)

        self.inputs = None

    def forward(self, inputs):
        # lazy initialize
        #若在前向传播时参数未初始化，自动通过输入的shape来初始化w和b
        if not self.is_init:
            self._init_parameters(inputs.shape[1])

        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def _init_parameters(self, input_size):
        self.shapes["w"][0] = input_size

        self.params["w"] = self.initializers["w"](shape=self.shapes["w"])
        self.params["b"] = self.initializers["b"](shape=self.shapes["b"])
        self.params["w"].zero_grad()
        self.params["b"].zero_grad()
        self.is_init = True


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
        return 1.0 / (1.0 + np.exp(-x))


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
