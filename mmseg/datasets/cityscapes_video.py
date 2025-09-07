import numpy as np
from PIL import Image
import os.path as osp
import tempfile
import mmengine

from mmseg.registry import DATASETS
from .custom import BaseClipDataset


@DATASETS.register_module()
class CityscapesDataset_clips(BaseClipDataset):
    """Cityscapes clip-based dataset.

    Uses fixed suffixes for Cityscapes convention.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Build Cityscapes color palette
        import cityscapesscripts.helpers.labels as CSLabels
        self.palette = np.zeros((len(CSLabels.id2label), 3), dtype=np.uint8)
        for label_id, label in CSLabels.id2label.items():
            self.palette[label_id] = label.color

    @staticmethod
    def _convert_to_label_id(result):
        """Convert trainId to id for cityscapes evaluation."""
        if isinstance(result, str):
            result = np.load(result)
        import cityscapesscripts.helpers.labels as CSLabels
        result_copy = result.copy()
        for trainId, label in CSLabels.trainId2label.items():
            result_copy[result == trainId] = label.id
        return result_copy

    def results2img(self, results, imgfile_prefix, to_label_id):
        """Write segmentation results to images."""
        mmengine.utils.mkdir_or_exist(imgfile_prefix)
        result_files = []
        prog_bar = mmengine.utils.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]
            if to_label_id:
                result = self._convert_to_label_id(result)
            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            output = Image.fromarray(result.astype(np.uint8)).convert('P')
            output.putpalette(self.palette)
            output.save(png_filename)
            result_files.append(png_filename)
            prog_bar.update()
        return result_files

    def format_results(self, results, imgfile_prefix=None, to_label_id=True):
        """Format results for Cityscapes evaluation."""
        assert isinstance(results, list)
        assert len(results) == len(self)

        if imgfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            imgfile_prefix = tmp_dir.name
        else:
            tmp_dir = None
        result_files = self.results2img(results, imgfile_prefix, to_label_id)
        return result_files, tmp_dir
