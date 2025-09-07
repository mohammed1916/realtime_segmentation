import os.path as osp
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from mmengine.utils import scandir
from mmcv import imread
from mmcv.transforms import Compose


class BaseClipDataset(Dataset):
    """Base dataset for video/clip inputs (e.g., Cityscapes video format).

    Args:
        data_root (str): Root path.
        img_dir (str): Directory with images.
        ann_dir (str, optional): Directory with annotations.
        pipeline (list[dict] | None): MMCV/MMSeg pipeline configs.
        used_labels (list[int] | None): Subset of labels to keep.
        dilation (list[int] | None): Frame offsets to load.
        istraining (bool): Training flag.
        classes (list[str] | None): Class names.
        palette (list[list[int]] | None): Color palette for masks.
        test_mode (bool): Test mode flag.
    """

    def __init__(self,
                 data_root,
                 img_dir,
                 ann_dir=None,
                 pipeline=None,
                 used_labels=None,
                 dilation=None,
                 istraining=True,
                 classes=None,
                 palette=None,
                 test_mode=False):
        self.data_root = data_root
        self.img_dir = osp.join(data_root, img_dir)
        self.ann_dir = osp.join(data_root, ann_dir) if ann_dir else None
        self.pipeline = Compose(pipeline) if pipeline else None
        self.used_labels = used_labels
        self.dilation = dilation if dilation is not None else [0]
        self.istraining = istraining
        self.CLASSES = classes
        self.PALETTE = palette
        self.test_mode = test_mode

        # Scan image directory
        self.img_infos = self.load_annotations()

    def load_annotations(self):
        """Scan image files and build metadata."""
        img_infos = []
        for filename in scandir(self.img_dir, suffix=('.png', '.jpg'), recursive=True):
            info = dict(img_path=osp.join(self.img_dir, filename))  # use img_path
            if self.ann_dir:
                seg_filename = filename.replace('leftImg8bit', 'gtFine_labelIds')
                info['seg_map_path'] = osp.join(self.ann_dir, seg_filename)
            img_infos.append(info)
        return img_infos

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        """Load multiple frames (clip) according to dilation values."""
        info = self.img_infos[idx]

        # --- Load center frame ---
        clip_imgs = []
        center_img = imread(info['img_path'])
        clip_imgs.append(center_img)

        # --- Collect neighbor frames with dilation ---
        for d in self.dilation:
            if d == 0:
                continue
            neighbor_idx = idx + d
            if 0 <= neighbor_idx < len(self.img_infos):
                neighbor_path = self.img_infos[neighbor_idx]['img_path']
                clip_imgs.append(imread(neighbor_path))

        # --- Annotation (only for center frame) ---
        ann = None
        if not self.test_mode and 'seg_map_path' in info:
            ann = np.array(Image.open(info['seg_map_path']), dtype=np.int64)

            # Map raw Cityscapes IDs to trainIds
            import cityscapesscripts.helpers.labels as CSLabels
            id2trainId = {label.id: label.trainId for label in CSLabels.id2label.values()}
            ann = np.vectorize(lambda x: id2trainId.get(x, 255))(ann)  # 255 = ignore label

        # --- Build results dict ---
        results = dict(
            img=clip_imgs,                     # list of frames
            gt_semantic_seg=[ann],             # list of masks (aligns with clips)
            img_path=info['img_path'],
            seg_map_path=info.get('seg_map_path', None),
            seg_fields=['gt_semantic_seg']     # tells pipeline what to transform
        )

        if self.pipeline:
            results = self.pipeline(results)

        return results


