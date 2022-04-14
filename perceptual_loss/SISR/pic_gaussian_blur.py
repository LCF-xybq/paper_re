import cv2
import imageio
import os
import matplotlib.pyplot as plt
from PIL import Image
from shutil import copy

if __name__ == '__main__':
    pth = r'D:\Program_self\paper_re\data\SR\train\lr\0001x4.png'
    src_pth = r'D:\Program_self\paper_re\data\SR\train\lr_raw'
    trg_pth = r'D:\Program_self\paper_re\data\SR\train\lr'

    images_raw = os.listdir(src_pth)
    for i in range(len(images_raw)):
        img_pth = os.path.join(src_pth, images_raw[i])
        img_raw = imageio.imread(img_pth)

        img_blur = cv2.GaussianBlur(img_raw, (3, 3), 1.0)


        img_blur = img_blur.astype("uint8")
        img = Image.fromarray(img_blur)
        img_name = os.path.join(trg_pth, images_raw[i])
        img.save(img_name)
