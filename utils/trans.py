import warnings
import random
import torchvision.transforms.functional as F

from PIL import Image
from torch import Tensor
from torchvision import transforms
from collections.abc import Sequence
from torchvision.transforms.functional import InterpolationMode


class Resize(object):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size

        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation

    def __call__(self, results):
        lr = results['lr']
        gt = results['gt']

        results['lr'] = F.resize(lr, self.size, self.interpolation)
        results['gt'] = F.resize(gt, self.size, self.interpolation)

        return results

class RandomCrop(object):
    def __init__(self, img_scale=None):
        if img_scale is None:
            self.img_scale = None
        else:
            assert isinstance(img_scale, (int, tuple))
            self.img_scale = img_scale

    def __call__(self, results):
        lr = results['lr']
        gt = results['gt']

        if isinstance(self.img_scale, int):
            th, tw = self.img_scale, self.img_scale
        else:
            th, tw = self.img_scale

        lr_w, lr_h = lr.size
        gt_w, gt_h = gt.size
        w, h = min(lr_w, gt_w), min(lr_h, gt_h)
        # th = tw = opt.crop_size
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        results['lr'] = F.crop(lr, i, j, th, tw)
        results['gt'] = F.crop(gt, i, j, th, tw)

        return results

class ToTensor(object):
    def __call__(self, results):
        lr = results['lr']
        gt = results['gt']

        results['lr'] = F.to_tensor(lr)
        results['gt'] = F.to_tensor(gt)

        return results

class RandomFlip(object):
    def __init__(self, flip_ratio=0.5):
        if flip_ratio is None:
            self.flip_ratio = None
        else:
            assert isinstance(flip_ratio, float)
            self.flip_ratio = flip_ratio

    def __call__(self, results):
        lr = results['lr']
        gt = results['gt']

        flip_prob = random.random()
        flip_transform = transforms.Compose([RandomHorizontalFlip(flip_prob)])

        results['lr'] = flip_transform(lr)
        results['gt'] = flip_transform(gt)

        return results

class RandomHorizontalFlip(object):
    def __init__(self, prob=None):
        self.prob = prob

    def __call__(self, img):
        if (self.prob is None and random.random() < 0.5) or self.prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)

        return img

class Normalize(object):
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std

    def __call__(self, results):
        lr = results['lr']
        gt = results['gt']

        assert isinstance(lr, Tensor)
        assert isinstance(gt, Tensor)

        results['lr'] = F.normalize(lr, mean=self.mean, std=self.std)
        results['gt'] = F.normalize(gt, mean=self.mean, std=self.std)

        return results


def _interpolation_modes_from_int(i: int):
    inverse_modes_mapping = {
        0: InterpolationMode.NEAREST,
        2: InterpolationMode.BILINEAR,
        3: InterpolationMode.BICUBIC,
        4: InterpolationMode.BOX,
        5: InterpolationMode.HAMMING,
        1: InterpolationMode.LANCZOS,
    }
    return inverse_modes_mapping[i]
