def get_lq_gt_img(results):
    assert isinstance(results, dict)

    scale = results['scale']

    lq_is_list = isinstance(results['lq'], list)
    if not lq_is_list:
        results['lq'] = [results['lq']]
    gt_is_list = isinstance(results['gt'], list)
    if not gt_is_list:
        results['gt'] = [results['gt']]

    h_lq, w_lq, _ = results['lq'][0].shape
    h_gt, w_gt, _ = results['gt'][0]

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