# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample


@HOOKS.register_module()
class SegVisualizationHook(Hook):
    """Segmentation Visualization Hook. Used to visualize validation and
    testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw
        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for visualization will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')
        self._test_index = 0

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[SegDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        # Try to find an image path from the data sample metainfo or the
        # raw data_batch. Be defensive: some custom datasets (e.g.
        # CityscapesDataset_clips) do not expose `SegDataSample.img_path`.
        def _resolve_img_path(data_sample, data_batch):
            # prefer explicit attribute if present
            img_path = getattr(data_sample, 'img_path', None)
            if img_path:
                return img_path
            # then try metainfo dict
            metainfo = getattr(data_sample, 'metainfo', None)
            if isinstance(metainfo, dict):
                img_path = metainfo.get('img_path') or metainfo.get('filename')
                if img_path:
                    return img_path
            # finally try the dataloader batch (may contain lists)
            if isinstance(data_batch, dict):
                if 'img_path' in data_batch:
                    v = data_batch['img_path']
                    if isinstance(v, (list, tuple)):
                        return v[0]
                    return v
                if 'img_metas' in data_batch:
                    try:
                        im = data_batch['img_metas']
                        if isinstance(im, (list, tuple)) and len(im) > 0:
                            im0 = im[0]
                            if isinstance(im0, dict):
                                return im0.get('img_path') or im0.get('filename')
                    except Exception:
                        pass
            return None

        img_path = _resolve_img_path(outputs[0], data_batch)
        if img_path is None:
            # Nothing we can visualize for this sample; skip safely.
            return

        try:
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        except Exception:
            # If loading fails, skip visualization rather than crashing.
            return

        window_name = f'val_{osp.basename(img_path)}'

        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                window_name,
                img,
                data_sample=outputs[0],
                show=self.show,
                wait_time=self.wait_time,
                step=total_curr_iter)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[SegDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        for i, data_sample in enumerate(outputs):
            self._test_index += 1

            # resolve image path with the same fallback used in val
            def _resolve_img_path_for_test(data_sample, data_batch, idx):
                img_path = getattr(data_sample, 'img_path', None)
                if img_path:
                    return img_path
                metainfo = getattr(data_sample, 'metainfo', None)
                if isinstance(metainfo, dict):
                    img_path = metainfo.get('img_path') or metainfo.get('filename')
                    if img_path:
                        return img_path
                if isinstance(data_batch, dict):
                    if 'img_path' in data_batch:
                        v = data_batch['img_path']
                        if isinstance(v, (list, tuple)) and len(v) > idx:
                            return v[idx]
                        return v
                    if 'img_metas' in data_batch:
                        try:
                            im = data_batch['img_metas']
                            if isinstance(im, (list, tuple)) and len(im) > idx:
                                im0 = im[idx]
                                if isinstance(im0, dict):
                                    return im0.get('img_path') or im0.get('filename')
                        except Exception:
                            pass
                return None

            img_path = _resolve_img_path_for_test(data_sample, data_batch, i)
            if img_path is None:
                # skip samples without a resolvable path
                continue

            window_name = f'test_{osp.basename(img_path)}'
            try:
                img_bytes = get(img_path, backend_args=self.backend_args)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            except Exception:
                continue

            self._visualizer.add_datasample(
                window_name,
                img,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                step=self._test_index)
