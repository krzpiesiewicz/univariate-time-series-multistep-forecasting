import torch

from dnn.stateful_module import StatefulModule
from dnn.stateful_lstms import StackedStatefulLSTMs


class StatefulLSTMsEncoder(StatefulModule):
    def __init__(self, input_size, hidden_sizes, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.stacked_lstms = StackedStatefulLSTMs(
            input_size=self.input_size, hidden_sizes=self.hidden_sizes, **kwargs
        )

    def __forward__(self, x):
        if x.size(1) > 0:
            out, (hs, cs) = self.stacked_lstms(x)
            h = hs[-1]
        else:
            h = self.stacked_lstms.lstms[-1].hidden_state
        return h


class StatefulLSTMsDecoder(StatefulModule):
    def __init__(
        self,
        input_size,
        hidden_sizes,
        recursive_input=False,
        output_len=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.stacked_lstms = StackedStatefulLSTMs(
            input_size=self.input_size, hidden_sizes=self.hidden_sizes, **kwargs
        )
        self.set(recursive_input=recursive_input, output_len=output_len)

    def set_hidden_state(self, hidden_state):
        self.reset(batch_size=hidden_state.size(1))
        self.stacked_lstms.lstms[0].set_hidden_state(hidden_state)

    def __set__(self, **kwargs):
        if "recursive_input" in kwargs:
            self.recursive_input = kwargs["recursive_input"]
        if "output_len" in kwargs:
            self.output_len = kwargs["output_len"]

    def __forward__(self, x, h=None, ex=None):
        self.reset()
        self.stacked_lstms.lstms[0].set_hidden_state(h)

        device = x.get_device()

        if self.recursive_input:
            output_size = (
                self.input_size if ex is None else self.input_size - ex.size(2)
            )
            out = torch.zeros((x.size(0), self.output_len, output_size)).to(device)
            x0 = x[:, 0:1, :]
            for i in range(self.output_len):
                x0 = x0.reshape((x.size(0), 1, x.size(2)))
                #                 print(f"x0.shape: {x0.shape}")
                if ex is not None:
                    #                     print(f"ex[:, i : i + 1, :].shape: {ex[:, i : i + 1, :].shape}")
                    x0 = torch.cat((x0, ex[:, i : i + 1, :]), 2)
                #                     print(f"x0 + ex.shape: {x0.shape}")
                #                 print(self.stacked_lstms)
                x0, _ = self.stacked_lstms(x0)
                out[:, i : i + 1, :] = x0
        else:
            if x.size(1) != self.output_len:
                x = torch.zeros((x.size(0), self.output_len, self.input_size)).to(
                    device
                )
            if ex is not None:
                input_size += ex.size(2)
                x = torch.cat((x, ex), 1)
            out, (hs, cs) = self.stacked_lstms(x)
            out = out[:, : self.output_len, :]

        out = out.reshape((out.size(0), out.size(2), out.size(1)))
        return out
    
    
class StatefulLSTMsAutoencoder(StatefulModule):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def __set__(self, **kwargs):
        if "output_len" in kwargs:
            self.output_len = kwargs["output_len"]

    def __forward__(self, x, y=None):
        assert self.output_len is not None
        self.decoder.set(output_len=self.output_len)
        encoded_h = self.encoder(x)
        if y is None:
            if x.size(1) >= 2:
                y = x[:, -2:-1, :]
            else:
                raise Exception(
                    "x.size(1) has to be at least 2 or y has to be provided"
                )
        decoded = self.decoder(x=y, h=encoded_h)
        return decoded
