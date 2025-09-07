import os
import os.path as osp
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from mmengine.utils import scandir
from mmengine.logging import print_log

from mmseg.registry import DATASETS
from mmcv.transforms import Compose
from mmcv import imread
from mmseg.evaluation.metrics import iou_metric


@DATASETS.register_module()
class CustomDataset_cityscape_clips(Dataset):
    """Custom Cityscapes dataset for segmentation clips."""

    def __init__(self,
                 data_root,
                 img_dir,
                 ann_dir=None,
                 pipeline=None,
                 classes=None,
                 palette=None,
                 test_mode=False):
        self.data_root = data_root
        self.img_dir = osp.join(data_root, img_dir)
        self.ann_dir = osp.join(data_root, ann_dir) if ann_dir else None
        self.pipeline = Compose(pipeline) if pipeline else None
        self.CLASSES = classes
        self.PALETTE = palette
        self.test_mode = test_mode

        # Scan image directory
        self.img_infos = self.load_annotations()

    def load_annotations(self):
        """Load image file names and corresponding annotation paths."""
        img_infos = []
        for filename in scandir(self.img_dir, suffix=('.png', '.jpg'), recursive=True):
            info = dict(filename=filename)
            if self.ann_dir:
                seg_filename = filename.replace('leftImg8bit', 'gtFine_labelIds')
                info['ann'] = osp.join(self.ann_dir, seg_filename)
            img_infos.append(info)
        return img_infos

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        info = self.img_infos[idx]
        img_path = osp.join(self.img_dir, info['filename'])
        img = imread(img_path)

        ann = None
        if not self.test_mode and 'ann' in info:
            ann = np.array(Image.open(info['ann']), dtype=np.int64)

        results = dict(
            img=img,
            gt_seg_map=ann,
            filename=info['filename']
        )

        if self.pipeline:
            results = self.pipeline(results)

        return results
