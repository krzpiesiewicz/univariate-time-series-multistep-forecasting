from dnn.stateful_module import StatefulModule


class StatefulLSTMsForecaster(StatefulModule):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def __set__(self, **kwargs):
        if "output_len" in kwargs:
            self.output_len = kwargs["output_len"]

    def __forward__(self, x, ex=None, y=None):
        assert self.output_len is not None
        self.decoder.set(output_len=self.output_len)
        encoded_h = self.encoder(x)
        if y is not None and y.size(1) > 0:
            y0 = y[:, 0:1, :]
        else:
            if x.size(1) > 0:
                y0 = x[:, -1:, :]
            else:
                raise Exception("One of x or y has to greater than zero in length.")
        out = self.decoder(x=y0, h=encoded_h, ex=ex)
        return out
