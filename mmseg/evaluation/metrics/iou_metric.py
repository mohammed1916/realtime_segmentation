# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable

from mmseg.registry import METRICS


@METRICS.register_module()
class IoUMetric(BaseMetric):
    """IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
    # Defensive: dataset_meta may be None when dataset didn't provide
    # metainfo (seen with CityscapesDataset_clips). First try a dataset
    # class fallback, then infer number of classes from model prediction
    # channels if necessary.
        if not getattr(self, 'dataset_meta', None) or 'classes' not in self.dataset_meta:
            # Cityscapes fallback: try to import dataset class and use its CLASSES/PALETTE
            try:
                import mmseg.datasets.cityscapes_video as _cs_mod
                cls = getattr(_cs_mod, 'CityscapesDataset_clips', None)
                if cls is not None and hasattr(cls, 'CLASSES'):
                    self.dataset_meta = {'classes': tuple(getattr(cls, 'CLASSES'))}
                    pal = getattr(cls, 'PALETTE', None)
                    if pal is not None:
                        self.dataset_meta['palette'] = pal
                    try:
                        print_log('IoUMetric: populated dataset_meta from CityscapesDataset_clips.CLASSES', logger='mmseg', level='INFO')
                    except Exception:
                        print('IoUMetric: populated dataset_meta from CityscapesDataset_clips.CLASSES')
            except Exception:
                pass

            # If fallback succeeded, use it. Otherwise try to infer from prediction tensor shape.
            if getattr(self, 'dataset_meta', None) and 'classes' in self.dataset_meta:
                num_classes = len(self.dataset_meta['classes'])
            else:
                try:
                    sample_pred = data_samples[0]['pred_sem_seg']['data']
                    # prediction is (C, H, W) or (1, C, H, W)
                    if hasattr(sample_pred, 'ndim') and sample_pred.ndim == 3:
                        num_classes = sample_pred.shape[0]
                    elif hasattr(sample_pred, 'ndim') and sample_pred.ndim == 4:
                        num_classes = sample_pred.shape[1]
                    else:
                        # fallback conservative default
                        num_classes = 19
                    try:
                        print_log(
                            f"IoUMetric: dataset_meta missing, inferred num_classes={num_classes}",
                            logger='mmseg',
                            level='WARNING')
                    except Exception:
                        # MMLogger may not be initialized in some debug runs; fall back
                        # to stdout to avoid crashing the evaluation loop.
                        print(f"IoUMetric WARNING: dataset_meta missing, inferred num_classes={num_classes}")
                except Exception:
                    num_classes = 19
                    try:
                        print_log(
                            "IoUMetric: unable to infer num_classes from data_samples; using default=19",
                            logger='mmseg',
                            level='WARNING')
                    except Exception:
                        print("IoUMetric WARNING: unable to infer num_classes from data_samples; using default=19")

            # One-time diagnostic: print data_batch/data_samples structure to help
            # identify where dataset class names might be available.
            try:
                if not getattr(self, '_logged_data_sample_structure', False):
                    try:
                        print_log(f'IoUMetric DEBUG: data_batch keys={list(data_batch.keys())}', logger='mmseg')
                    except Exception:
                        print(f'IoUMetric DEBUG: data_batch keys={list(data_batch.keys())}')
                    try:
                        sample_keys = list(data_samples[0].keys()) if len(data_samples) > 0 else None
                        print_log(f'IoUMetric DEBUG: data_samples[0] keys={sample_keys}', logger='mmseg')
                        # One-time: log which GT class ids appear in the first sample
                        try:
                            if not getattr(self, '_logged_label_values', False) and len(data_samples) > 0:
                                gs = data_samples[0].get('gt_sem_seg')
                                if gs is not None:
                                    lab = gs.get('data', None)
                                    if lab is not None:
                                        try:
                                            arr = lab.cpu().numpy()
                                        except Exception:
                                            arr = np.array(lab)
                                        unique_vals = np.unique(arr)
                                        try:
                                            if getattr(self, 'dataset_meta', None) and 'classes' in self.dataset_meta:
                                                names = [self.dataset_meta['classes'][int(v)] if 0 <= int(v) < len(self.dataset_meta['classes']) else str(int(v)) for v in unique_vals]
                                                print_log(f'IoUMetric DEBUG: gt unique ids={list(unique_vals)}, names={names}', logger='mmseg')
                                            else:
                                                print_log(f'IoUMetric DEBUG: gt unique ids={list(unique_vals)}', logger='mmseg')
                                        except Exception:
                                            print(f'IoUMetric DEBUG: gt unique ids={list(unique_vals)}')
                                self._logged_label_values = True
                        except Exception:
                            pass
                    except Exception:
                        pass
                    self._logged_data_sample_structure = True
            except Exception:
                pass
        else:
            num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data']
            # Normalize prediction tensor to a 2D class map (H, W).
            # Possible shapes: (C, H, W), (1, C, H, W), (H, W) or (1, H, W).
            if hasattr(pred_label, 'ndim') and pred_label.ndim == 4:
                # (1, C, H, W) -> (C, H, W)
                pred_label = pred_label.squeeze(0)
            if hasattr(pred_label, 'ndim') and pred_label.ndim == 3:
                # (C, H, W) -> class map by argmax over channels
                pred_label = pred_label.argmax(dim=0)
            else:
                # ensure it's 2D
                pred_label = pred_label.squeeze()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze()
                # If shapes mismatch, resize prediction to match label using nearest
                if pred_label.shape != label.shape:
                    try:
                        # make float for interpolation then restore integer class labels
                        pred_label = pred_label.unsqueeze(0).unsqueeze(0).float()
                        pred_label = F.interpolate(
                            pred_label, size=tuple(label.shape), mode='nearest')
                        pred_label = pred_label.squeeze().to(label.dtype)
                    except Exception:
                        # As a fallback, attempt to transpose if dimensions are swapped
                        try:
                            pred_label = pred_label.transpose(0, 1)
                        except Exception:
                            pass
                label = label.to(pred_label)
                self.results.append(self.intersect_and_union(pred_label, label,
                                                             num_classes,
                                                             self.ignore_index))
            # format_result
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(
                    data_sample['img_path']))[0]
                png_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}.png'))
                output_mask = pred_label.cpu().numpy()
                # The index range of official ADE20k dataset is from 0 to 150.
                # But the index range of output is from 0 to 149.
                # That is because we set reduce_zero_label=True.
                if data_sample.get('reduce_zero_label', False):
                    output_mask = output_mask + 1
                output = Image.fromarray(output_mask.astype(np.uint8))
                output.save(png_filename)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)

        # One-time diagnostic: which class indices actually appear in GT
        try:
            present = np.where(np.array(total_area_label) > 0)[0]
            if len(present) == 0:
                try:
                    print_log('IoUMetric DEBUG: no ground-truth pixels found for any class in this evaluation set', logger='mmseg')
                except Exception:
                    print('IoUMetric DEBUG: no ground-truth pixels found for any class in this evaluation set')
            else:
                try:
                    if getattr(self, 'dataset_meta', None) and 'classes' in self.dataset_meta:
                        names = [self.dataset_meta['classes'][int(i)] for i in present]
                        print_log(f'IoUMetric DEBUG: gt present class indices={list(present)}, names={names}', logger='mmseg')
                    else:
                        print_log(f'IoUMetric DEBUG: gt present class indices={list(present)}', logger='mmseg')
                except Exception:
                    print(f'IoUMetric DEBUG: gt present class indices={list(present)}')
        except Exception:
            pass

        # Defensive: dataset_meta may be None for some custom datasets.
        # Fall back to generated class names based on the number of classes
        # inferred from the accumulated histogram arrays.
        if getattr(self, 'dataset_meta', None) and 'classes' in self.dataset_meta:
            class_names = self.dataset_meta['classes']
        else:
            num_classes = len(total_area_intersect)
            try:
                print_log(
                    f"IoUMetric: dataset_meta missing, using generated class names for {num_classes} classes",
                    logger='mmseg',
                    level='WARNING')
            except Exception:
                print(f"IoUMetric WARNING: dataset_meta missing, using generated class names for {num_classes} classes")
            class_names = [f'class_{i}' for i in range(num_classes)]

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                metrics[key] = val
            else:
                metrics['m' + key] = val

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        # Prepare a human-friendly per-class table. Replace NaN with '-' so
        # absent classes (no GT pixels) are clearly shown, and ensure all
        # columns have matching lengths.
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # Convert numeric arrays to lists of display strings, preserving
        # the ability to return numeric metrics separately.
        display_metrics = OrderedDict()
        for key, val in ret_metrics_class.items():
            # Class column is already strings
            if key == 'Class':
                display_metrics[key] = list(val)
                continue

            # Ensure we operate on a numpy array for isnan checks
            arr = np.array(val)
            disp_list = []
            for v in arr:
                if np.isnan(v):
                    disp_list.append('-')
                else:
                    # keep two decimal places for readability
                    try:
                        disp_list.append(f"{float(v):.2f}")
                    except Exception:
                        disp_list.append(str(v))
            display_metrics[key] = disp_list

        class_table_data = PrettyTable()
        for key, val in display_metrics.items():
            class_table_data.add_column(key, val)

        try:
            print_log('per class results:', logger)
            print_log('\n' + class_table_data.get_string(), logger=logger)
        except Exception:
            # Fallback to stdout in case the MMLogger instance isn't available.
            print('per class results:')
            print('\n' + class_table_data.get_string())

        return metrics

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta**2) * (precision * recall) / (
                (beta**2 * precision) + recall)
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'aAcc': all_acc})
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall

        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics
