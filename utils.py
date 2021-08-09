import torch
import torch.nn as nn


class ChainRunner(nn.Module):
    """
    Takes an instance of augment.EffectChain and applies it on pytorch tensors.
    """

    def __init__(self, chain):
        super().__init__()
        self.chain = chain

    def forward(self, x):
        """
        x: torch.Tensor, (channels, length). Must be placed on CPU.
        """
        src_info = {'channels': x.size(0),  # number of channels
                    'length': x.size(1),   # length of the sequence
                    'precision': 32,       # precision (16, 32 bits)
                    'rate': 16000.0,       # sampling rate
                    'bits_per_sample': 32}  # size of the sample

        target_info = {'channels': 1,
                       'length': x.size(1),
                       'precision': 32,
                       'rate': 16000.0,
                       'bits_per_sample': 32}

        y = self.chain.apply(
            x, src_info=src_info, target_info=target_info)

        # sox might misbehave sometimes by giving nan/inf if sequences are too short (or silent)
        # and the effect chain includes eg `pitch`
        if torch.isnan(y).any() or torch.isinf(y).any():
            return x.clone()
        return y