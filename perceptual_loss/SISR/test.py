import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


images = []

pth = r'D:\Program_self\paper_re\data\SR\train\lq'

for root, _, fnames in sorted(os.walk(pth)):
    for fname in fnames:
        if is_image_file(fname):
            path = os.path.join(root, fname)
            images.append(path)

print(images)