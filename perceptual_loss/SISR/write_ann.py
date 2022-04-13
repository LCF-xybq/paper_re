from base_sr_dataset import BaseSRDataset
import os
import imageio

if __name__ == '__main__':
    gt_folder = r'D:\Program_self\paper_re\data\SR\train\gt'
    ann_txt = r'D:\Program_self\paper_re\perceptual_loss\SISR\ann.txt'

    img_list = BaseSRDataset.scan_folder(gt_folder)
    with open(ann_txt, 'a') as f:
        for img in img_list:
            image = imageio.imread(img)
            h, w, c = image.shape
            f.write(f'{img} ({h},{w},{c})\n')