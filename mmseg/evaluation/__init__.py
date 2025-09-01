# Copyright (c) OpenMMLab. All rights reserved.
from .metrics import CityscapesMetric, DepthMetric, IoUMetric

# Provide a compatibility alias `eval_metrics` used by older codepaths.
try:
	# Prefer a local implementation if available
	from tv3s_utils.datasets.custom import eval_metrics as _eval_metrics  # type: ignore
except Exception:
	# Minimal fallback implementation
	def _eval_metrics(results, gt_seg_maps, num_classes, ignore_index, metrics, **kwargs):
		# return zeros for common metrics in a compatible shape
		if 'mIoU' in metrics:
			return (0.0, [0.0] * num_classes, [0.0] * num_classes)
		return ()

eval_metrics = _eval_metrics

__all__ = ['IoUMetric', 'CityscapesMetric', 'DepthMetric', 'eval_metrics']
