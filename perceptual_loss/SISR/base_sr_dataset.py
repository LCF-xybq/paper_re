import os
import copy
from base_dataset import BaseDataset

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')

class BaseSRDataset(BaseDataset):
    def __init__(self, transform, scale, test_mode=False):
        super(BaseSRDataset, self).__init__(transform, test_mode)
        self.scale = scale

    @staticmethod
    def scan_folder(path):
        """
            path: image folder(only images)
            return images list from given folder
        """
        images = os.listdir(path)
        images = [os.path.join(path, img)
                  for img in images
                  if os.path.splitext(img)[-1] in IMG_EXTENSIONS]

        assert images, f'{path} has no valid image file.'
        return images

    def __getitem__(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        results['scale'] = self.scale
        return self.transform(results)

# if __name__ == '__main__':
#     pth = r'D:\Program_self\paper_re\data\SR\train\lq'
#     images = BaseSRDataset.scan_folder(pth)
#     print(type(images), len(images))
#     for img in images:
#         print(img)