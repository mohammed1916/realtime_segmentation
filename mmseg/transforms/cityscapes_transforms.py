import numpy as np
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class CityscapesLabelIdToTrainId:
    """Convert Cityscapes labelIds to trainIds.

    This transform converts the 34-class labelIds used in Cityscapes
    to the 19-class trainIds used for training.

    Args:
        labelId_to_trainId (dict): Mapping from labelId to trainId
    """

    def __init__(self, labelId_to_trainId=None):
        if labelId_to_trainId is None:
            # Standard Cityscapes mapping
            self.labelId_to_trainId = {
                0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255,
                10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255,
                19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
                29: 255, 30: 255, 31: 16, 32: 17, 33: 18
            }
        else:
            self.labelId_to_trainId = labelId_to_trainId

    def __call__(self, results):
        """Convert labelIds to trainIds in the segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results with trainIds.
        """
        if 'gt_seg_map' in results:
            gt_seg_map = results['gt_seg_map']

            # Convert to numpy array if it's a PIL image
            if hasattr(gt_seg_map, 'convert'):
                gt_seg_map = np.array(gt_seg_map)

            # Apply mapping
            trainId_map = np.full_like(gt_seg_map, 255, dtype=np.uint8)

            for labelId, trainId in self.labelId_to_trainId.items():
                if trainId != 255:  # Only map valid classes
                    trainId_map[gt_seg_map == labelId] = trainId

            results['gt_seg_map'] = trainId_map

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(labelId_to_trainId={self.labelId_to_trainId})'
