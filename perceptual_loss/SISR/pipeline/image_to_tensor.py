import numpy as np
import torch

class ImageToTensor:
    def __init__(self, keys, to_float32=True):
        self.keys = keys
        self.to_float32 = to_float32

    def __call__(self, results):
        for key in self.keys:
            # deal with gray scale img: expand a color channel
            if len(results[key].shape) == 2:
                results[key] = results[key][..., None]
            if self.to_float32 and not isinstance(results[key], np.float32):
                results[key] = results[key].astype(np.float32)
            results[key] = torch.from_numpy(results[key].transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__ + (
            f'(keys={self.keys}, to_float32={self.to_float32})')