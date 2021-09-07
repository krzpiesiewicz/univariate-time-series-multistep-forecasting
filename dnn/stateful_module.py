from abc import ABC, abstractmethod

import torch
from torch import nn


class StatefulModule(nn.Module):
    def __init__(self, debug=False, **kwargs):
        super().__init__()
        self.debug = debug
        self.reset(batch_size=None)

    def stateful_submodules(self):
        return [
            module
            for module in self.children()
            if issubclass(type(module), StatefulModule)
        ]

    def set(self, **kwargs):
        if "debug" in kwargs:
            self.debug = kwargs["debug"]

        self.__set__(**kwargs)
        for module in self.stateful_submodules():
            module.set(**kwargs)

        if "batch_size" in kwargs:
            self.reset(batch_size=kwargs["batch_size"])

    def reset(self, **kwargs):
        if self.debug:
            print(f"self.reset {self.__class__}({self.__hash__()})")
        if "batch_size" in kwargs:
            self.batch_size = kwargs["batch_size"]
        self.__reset__(**kwargs)
        for module in self.stateful_submodules():
            if self.debug:
                print(
                    f"children of {self.__hash__()}: {module.__class__}({module.__hash__()})"
                )
            module.reset(**kwargs)

    def forward(self, x, *args, **kwargs):
        if self.batch_size != x.size(0):
            self.reset(batch_size=x.size(0))
        return self.__forward__(x, *args, **kwargs)

    @abstractmethod
    def __set__(self, **kwargs):
        pass

    @abstractmethod
    def __reset__(self, **kwargs):
        pass

    @abstractmethod
    def __forward__(self, x, *args):
        pass
