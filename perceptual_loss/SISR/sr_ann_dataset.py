import os

from base_sr_dataset import BaseSRDataset

class SRAnnDataset(BaseSRDataset):
    def __init__(self, lq_folder, gt_folder, ann_file,
                 transform, scale, test_mode=False,):
        super(SRAnnDataset, self).__init__(transform, scale, test_mode)
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.ann_file = str(ann_file)
        self.scale = scale
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        data_infos = []

        with open(self.ann_file, 'r') as f:
            for line in f:
                gt_name = line.split(' ')[0]
                basename, ext = os.path.splitext(os.path.basename(gt_name))
                lq_name = f'{basename}x{self.scale}{ext}'
                data_infos.append(
                    dict(
                        lq_path = os.path.join(self.lq_folder, lq_name),
                        gt_path = os.path.join(self.gt_folder, gt_name)
                    )
                )
        return data_infos
