import numpy as np
from typing import List, Tuple


class NMS:
    """
    Implements Non-Maximum Suppression (NMS) to filter redundant bounding boxes 
    in object detection.

    This class takes bounding boxes, confidence scores, and class IDs and applies 
    NMS to retain only the most relevant bounding boxes based on confidence scores 
    and Intersection over Union (IoU) thresholding.
    """

    def __init__(self, score_threshold: float, nms_iou_threshold: float) -> None:
        """
        Initializes the NMS filter with confidence and IoU thresholds.

        :param score_threshold: The minimum confidence score required to retain a bounding box.
        :param nms_iou_threshold: The Intersection over Union (IoU) threshold for non-maximum suppression.

        :ivar self.score_threshold: The threshold below which detections are discarded.
        :ivar self.nms_iou_threshold: The IoU threshold that determines whether two boxes 
                                      are considered redundant.
        """
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def __init__(self, score_threshold: float, nms_iou_threshold: float) -> None:
        """
        Initializes the NMS filter with confidence and IoU thresholds.
        """
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold

    @staticmethod
    def _iou(box_a: List[int], box_b: List[int]) -> float:
        """
        Computes IoU for boxes in [x, y, w, h] format.
        """
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        ax1, ay1, ax2, ay2 = float(ax), float(ay), float(ax + aw), float(ay + ah)
        bx1, by1, bx2, by2 = float(bx), float(by), float(bx + bw), float(by + bh)

        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter_area = inter_w * inter_h

        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = area_a + area_b - inter_area

        if union <= 0.0:
            return 0.0
        return inter_area / union
   
    
    def filter(
        self,
        bboxes: List[List[int]],
        class_ids: List[int],
        scores: List[float],
        class_scores: List[float],
    ) -> Tuple[List[List[int]], List[int], List[float], List[float]]:
        """
        Applies Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.

        :param bboxes: A list of bounding boxes, where each box is represented as 
                       [x, y, width, height]. (x, y) is the top-left corner.
        :param class_ids: A list of class IDs corresponding to each bounding box.
        :param scores: A list of confidence scores for each bounding box.
        :param class_scores: A list of class-specific scores for each detection.

        :return: A tuple containing:
            - **filtered_bboxes (List[List[int]])**: The final bounding boxes after NMS.
            - **filtered_class_ids (List[int])**: The class IDs of retained bounding boxes.
            - **filtered_scores (List[float])**: The confidence scores of retained bounding boxes.
            - **filtered_class_scores (List[float])**: The class-specific scores of retained boxes.

        **How NMS Works:**
        - The function selects the bounding box with the highest confidence.
        - It suppresses any boxes that have a high IoU (overlapping area) with this selected box.
        - This process is repeated until all valid boxes are retained.

        **Example Usage:**
        ```python
        nms_processor = NMS(score_threshold=0.5, nms_iou_threshold=0.4)
        final_bboxes, final_class_ids, final_scores, final_class_scores = nms_processor.filter(
            bboxes, class_ids, scores, class_scores
        )
        ```
        """

        # TASK: Apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.
        #         DO NOT USE **cv2.dnn.NMSBoxes()** for this Assignment. For Assignment 2, you will be
        #         permitted to use this function.
        #
        # Return these variables in order as described in Line 46-50:
        # return filtered_bboxes, filtered_class_ids, filtered_scores, filtered_class_scores

        # Quick exits for empty inputs (tests expect ([], [], [], []) )
        if not bboxes or not class_ids or not scores or not class_scores:
            return [], [], [], []

        # Basic consistency check
        n = len(bboxes)
        if len(class_ids) != n or len(scores) != n or len(class_scores) != n:
            raise ValueError("Input lists (bboxes, class_ids, scores, class_scores) must have the same length.")

        # 1) Discard boxes below the score threshold
        candidate_indices = [i for i, s in enumerate(scores) if float(s) >= self.score_threshold]
        if not candidate_indices:
            return [], [], [], []

        # 2) Sort remaining indices by descending score
        candidate_indices = sorted(candidate_indices, key=lambda i: float(scores[i]), reverse=True)

        # 3) Greedily keep highest-score box, suppress others with IoU > threshold
        keep: List[int] = []
        while candidate_indices:
            best = candidate_indices.pop(0)
            keep.append(best)

            remaining: List[int] = []
            for idx in candidate_indices:
                iou_val = self._iou(bboxes[best], bboxes[idx])
                if iou_val <= self.nms_iou_threshold:
                    remaining.append(idx)
                
            candidate_indices = remaining

        # 4) Map kept indices back to outputs (in the order they were kept)
        filtered_bboxes = [bboxes[i] for i in keep]
        filtered_class_ids = [class_ids[i] for i in keep]
        filtered_scores = [scores[i] for i in keep]
        filtered_class_scores = [class_scores[i] for i in keep]

        return filtered_bboxes, filtered_class_ids, filtered_scores, filtered_class_scores
        
