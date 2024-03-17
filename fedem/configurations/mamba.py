# type: ignore

import math
from typing import Union

from transformers import PretrainedConfig


class MambaConfig(PretrainedConfig):
    model_type = "mamba"

    def __init__(
        self,
        vocab_size=50277,
        d_state=16,
        d_model=2560,
        d_conv=4,
        expand=2,
        conv_bias=True,
        bias=False,
        n_layer=64,
        dt_rank: Union[int, str] = "auto",
        pad_vocab_size_multiple=8,
        initializer_range=0.02,
        **kwargs,
    ):
        """
        Configuration class for Mamba model.

        Args:
            vocab_size (int, optional): Vocabulary size. Defaults to 50277.
            d_state (int, optional): State dimension. Defaults to 16.
            d_model (int, optional): Model dimension. Defaults to 2560.
            d_conv (int, optional): Convolution dimension. Defaults to 4.
            expand (int, optional): Expansion factor. Defaults to 2.
            conv_bias (bool, optional): Convolution bias flag. Defaults to True.
            bias (bool, optional): Bias flag. Defaults to False.
            n_layer (int, optional): Number of layers. Defaults to 64.
            dt_rank (int | str, optional): Tensor rank or 'auto' for automatic computation. Defaults to 'auto'.
            pad_vocab_size_multiple (int, optional): Vocabulary size multiple for padding. Defaults to 8.
            initializer_range (float, optional): Initializer range. Defaults to 0.02.
            **kwargs: Additional keyword arguments.

        Attributes:
            model_type (str): Model type identifier.
            d_inner (int): Inner dimension based on expansion and model dimension ratio.

        """
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.conv_bias = conv_bias
        self.expand = expand
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.d_conv = d_conv
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = dt_rank
        self.initializer_range = initializer_range
        self.bias = bias

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )
        super().__init__(
            **kwargs,
        )
