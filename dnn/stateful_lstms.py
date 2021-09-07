import torch
from torch import nn
from torch.autograd import Variable

from dnn.stateful_module import StatefulModule


class SatefulLSTM(StatefulModule):
    def __init__(self, input_size, hidden_size, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = 1
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        self.hidden_state = None
        self.cell_state = None

    def set_hidden_state(self, hidden_state):
        self.reset(batch_size=hidden_state.size(1))
        if self.debug:
            print(f"{hidden_state.shape} == (self) {self.hidden_state.shape}")
        assert hidden_state.shape == self.hidden_state.shape
        self.hidden_state = hidden_state

    def set_cell_state(self, cell_state):
        if self.batch_size is None:
            self.reset(batch_size=cell_state.size(1))
        assert cell_state.shape == self.cell_state.shape
        self.cell_state = cell_state
        
    def states_to(self, device):
        if self.cell_state.get_device() != device:
            self.cell_state = self.cell_state.to(device)
        if self.hidden_state.get_device() != device:
            self.hidden_state = self.hidden_state.to(device)

    def __reset__(self, **kwargs):
        if self.debug:
            print(f"LSTM.__reset__(): batch_size {self.batch_size}")
        if "batch_size" in kwargs:
            self.batch_size = kwargs["batch_size"]
        if self.batch_size is None:
            self.hidden_state = None
            self.cell_state = None
        else:
            self.hidden_state = torch.zeros(
                self.num_layers, self.batch_size, self.hidden_size
            )
            self.cell_state = torch.zeros(
                self.num_layers, self.batch_size, self.hidden_size
            )
            if self.debug:
                print(f"LSTM self.hidden_state.shape {self.hidden_state.shape}")

    def __forward__(self, x):
        device = x.get_device()
        self.states_to(device)
        
        h_0 = Variable(self.hidden_state)
        c_0 = Variable(self.cell_state)
        if self.debug:
            print(x.shape)
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        self.hidden_state = h_n
        self.cell_state = c_n
        return out, (h_n, c_n)
    
    
class StackedStatefulLSTMs(StatefulModule):
    def __init__(self, input_size, hidden_sizes, dropout=0.0, dropouts=None, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        if dropouts is None:
            self.dropouts = [dropout for _ in range(len(hidden_sizes))] + [0.0]
        else:
            self.dropouts = dropouts
        self.lstms = nn.ModuleList(
            [
                SatefulLSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    **kwargs,
                )
                for input_size, hidden_size, dropout in zip(
                    [self.input_size] + self.hidden_sizes,
                    self.hidden_sizes,
                    self.dropouts,
                )
            ]
        )
        self.reset()

    def __forward__(self, x):
        hs = []
        cs = []
        for lstm in self.lstms:
            out, (h_n, c_n) = lstm(x)
            hs.append(h_n)
            cs.append(c_n)
            x = out
        return out, (hs, cs)
