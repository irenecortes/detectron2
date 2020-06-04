# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from typing import Any, List, Tuple, Union
import torch

class Tokens:
    """
    Stores token annotation data. GT Instances have a 'gt_tokens' property containing
    the code of each token. This tensor has shape (N) where N is the number of instances.
    """

    def __init__(self, tokens: Union[torch.Tensor, np.ndarray, List[List[float]]]):
        """
        Arguments:
            tokens: A Tensor, numpy array, or list of the code of each token.
                The shape should be (N) where N is the number of
                instances.
        """
        device = tokens.device if isinstance(tokens, torch.Tensor) else torch.device("cpu")
        # tokens = torch.as_tensor(tokens, dtype=str, device=device)
        # assert tokens.dim() == 3 and tokens.shape[2] == 3, tokens.shape
        self.tensor = np.array(tokens)

    def __len__(self) -> int:
        return len(self.tensor)

    # def to(self, *args: Any, **kwargs: Any) -> "Tokens":
    #     return type(self)(self.tensor.to(*args, **kwargs))

    def get_str(self):
        return self.tensor[0]

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Tokens":
        """
        Create a new `Tokens` by indexing on this `Tokens`.

        The following usage are allowed:

        1. `new_tkns = tkns[3]`: return a `Tokens` which contains only one instance.
        2. `new_tkns = tkns[2:10]`: return a slice of key points.
        3. `new_tkns = tkns[vector]`, where vector is a torch.ByteTensor
           with `length = len(tkns)`. Nonzero elements in the vector will be selected.

        Note that the returned Tokens might share storage with this Tokens,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, torch.Tensor):
            item = item.cpu().numpy()
        if isinstance(item, int):
            return Tokens([self.tensor[item]])
        return Tokens(self.tensor[item])

    def __repr__(self) -> str:
        return "Tokens (" + str(self.tensor) + ")"
        return s
