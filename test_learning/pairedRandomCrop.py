import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch

class PairedRandomCrop:
    def __init__(self, gt_patch_size):
        self.gt_patch_size = gt_patch_size

    def __call__(self, results):
        scale = results['scale']
        lq_patch_size = self.gt_patch_size // scale

        lq_is_list = isinstance(results['lq'], list)
        if not lq_is_list:
            results['lq'] = [results['lq']]
        gt_is_list = isinstance(results['gt'], list)
        if not gt_is_list:
            results['gt'] = [results['gt']]

        h_lq, w_lq, _ = results['lq'][0].shape
        h_gt, w_gt, _ = results['gt'][0].shape

        if h_gt != h_lq * scale or w_gt != w_lq * scale:
            raise ValueError(
                f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                f'multiplication of LQ ({h_lq}, {w_lq}).')
        if h_lq < lq_patch_size or w_lq < lq_patch_size:
            raise ValueError(
                f'LQ ({h_lq}, {w_lq}) is smaller than patch size ',
                f'({lq_patch_size}, {lq_patch_size}). Please check '
                f'{results["lq_path"][0]} and {results["gt_path"][0]}.')

        # randomly choose top and left coordinates for lq patch
        top = np.random.randint(h_lq - lq_patch_size + 1)
        left = np.random.randint(w_lq - lq_patch_size + 1)
        # crop lq patch
        results['lq'] = [
            v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
            for v in results['lq']
        ]
        # crop corresponding gt patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        results['gt'] = [
            v[top_gt:top_gt + self.gt_patch_size,
              left_gt:left_gt + self.gt_patch_size, ...] for v in results['gt']
        ]

        if not lq_is_list:
            results['lq'] = results['lq'][0]
        if not gt_is_list:
            results['gt'] = results['gt'][0]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(gt_patch_size={self.gt_patch_size})'
        return repr_str

if __name__ == '__main__':
    transform = PairedRandomCrop(288)
    print(transform)
    test = {'lq': r'D:\Program_self\paper_re\data\SR\train\lq\0001x4.png',
            'gt': r'D:\Program_self\paper_re\data\SR\train\gt\0001.png'}

    img_gt = imageio.imread(test['gt'])
    img_lq = imageio.imread(test['lq'])

    to_do = {'lq': [img_lq],
             'gt': [img_gt],
             'scale': 4}

    res = transform(to_do)

    patch_lq = res['lq'][0]
    patch_gt = res['gt'][0]

    plt.figure(2)
    plt.imshow(patch_lq)
    plt.show()

    plt.imshow(patch_gt)
    plt.show()

    print(patch_lq.shape, patch_gt.shape)