# coding: utf-8

""" SUN RGBD Dataset for Baseline method [Finetune] (fine-tune with few novel class data).

Author: Zhao Na
Date: Oct, 2020
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'cfg'))
from sunrgbd_cfg import cfg, get_class2scans
from sunrgbd import SunrgbdNovelDatasetConfig, SunrgbdDataset


class SunrgbdNovelDataset(SunrgbdDataset):

    def __init__(self, num_novel_class=3, num_points=40000, use_color=False, use_height=False, augment=False):
        super(SunrgbdNovelDataset, self).__init__(num_points, use_color, use_height, augment)

        self.num_novel_class = num_novel_class

        self.dataset_config = SunrgbdNovelDatasetConfig(num_novel_class)

        class2scans = get_class2scans(self.data_path)
        all_scan_names = [scan_name for class_name in self.dataset_config.types for scan_name in class2scans[class_name]]
        self.scan_names = list(set(all_scan_names))
        print('Training classes: {0} | number of scenes: {1}'.format(self.dataset_config.types, len(self.scan_names)))

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        return self._process_one_scene(scan_name, self.train_data_path, self.dataset_config)


if __name__=='__main__':
    dset = SunrgbdNovelDataset(num_points=20000, use_color=False, use_height=True, augment=True)
    for i_example in range(4):
        example = dset.__getitem__(1)