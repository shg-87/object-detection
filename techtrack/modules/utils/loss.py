import numpy as np

class Loss:
    """
    *Modified* YOLO Loss for Hard Negative Mining.
    """

    def __init__(
        self,
        iou_threshold=0.5,
        lambda_coord=0.5,
        lambda_obj=0.5,
        lambda_noobj=0.5,
        lambda_cls=0.5,
        num_classes=20,
    ):
        self.num_classes = int(num_classes)
        self.lambda_coord = float(lambda_coord)
        self.lambda_cls = float(lambda_cls)
        self.lambda_obj = float(lambda_obj)
        self.lambda_noobj = float(lambda_noobj)
        self.iou_threshold = float(iou_threshold)

        self.columns = [
            "total_loss",
            "loc_loss",
            "conf_loss_obj",
            "conf_loss_noobj",
            "class_loss",
        ]

    def get_predictions(self, predictions):
        """
        Predictions format (per unit tests):
        [x1, y1, x2, y2, objectness, class_score_0, ...]
        """
        bboxes = []
        obj_scores = []
        cls_scores = []

        for out in (predictions or []):
            for det in (out or []):
                det = np.asarray(det, dtype=float)

                # [x1,y1,x2,y2]
                bboxes.append(det[:4].tolist())

                # objectness
                obj_scores.append(float(det[4]) if det.shape[0] > 4 else 0.0)

                # class scores vector
                cs = det[5 : 5 + self.num_classes] if det.shape[0] > 5 else np.zeros((0,), dtype=float)
                if cs.shape[0] < self.num_classes:
                    padded = np.zeros((self.num_classes,), dtype=float)
                    padded[: cs.shape[0]] = cs
                    cs = padded
                cls_scores.append(cs.tolist())

        pred_box = np.asarray(bboxes, dtype=float) if bboxes else np.zeros((0, 4), dtype=float)
        objectness_score = np.asarray(obj_scores, dtype=float) if obj_scores else np.zeros((0,), dtype=float)
        class_scores = (
            np.asarray(cls_scores, dtype=float)
            if cls_scores
            else np.zeros((0, self.num_classes), dtype=float)
        )

        return pred_box, objectness_score, class_scores

    def get_annotations(self, annotations):
        """
        Annotations format (per unit tests):
        [class_id, x1, y1, x2, y2]
        """
        gt_boxes = []
        gt_class_ids = []

        for ann in (annotations or []):
            ann = list(ann)
            gt_class_ids.append(int(ann[0]))
            gt_boxes.append([float(ann[1]), float(ann[2]), float(ann[3]), float(ann[4])])

        gt_box = np.asarray(gt_boxes, dtype=float) if gt_boxes else np.zeros((0, 4), dtype=float)
        gt_class_id = np.asarray(gt_class_ids, dtype=int) if gt_class_ids else np.zeros((0,), dtype=int)
        return gt_box, gt_class_id

    @staticmethod
    def _iou_xyxy(a, b):
        """IoU for boxes in [x1,y1,x2,y2]."""
        ax1, ay1, ax2, ay2 = map(float, a)
        bx1, by1, bx2, by2 = map(float, b)

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter

        return inter / union if union > 0.0 else 0.0

    def compute(self, predictions, annotations):
        """
        Uses xyxy boxes as required by the unit-test docstrings.
        """
        loc_loss = 0.0
        class_loss = 0.0
        conf_loss_obj = 0.0
        conf_loss_noobj = 0.0

        pred_box, obj_score, cls_score = self.get_predictions(predictions)
        gt_box, gt_class_id = self.get_annotations(annotations)

        N = int(pred_box.shape[0])
        M = int(gt_box.shape[0])

        # Track which GTs were matched by at least one prediction
        gt_matched = np.zeros(M, dtype=bool)

        # If no predictions: every GT is missed (hard negative mining friendly)
        if N == 0:
            if M > 0:
                conf_loss_obj = float(M) * (1.0 ** 2)  # (0 - 1)^2 per missed GT

            total_loss = (
                self.lambda_coord * loc_loss
                + self.lambda_obj * conf_loss_obj
                + self.lambda_noobj * conf_loss_noobj
                + self.lambda_cls * class_loss
            )
            return {
                "total_loss": float(total_loss),
                "loc_loss": float(loc_loss),
                "conf_loss_obj": float(conf_loss_obj),
                "conf_loss_noobj": float(conf_loss_noobj),
                "class_loss": float(class_loss),
            }

        # For each prediction: match to the best GT by IoU
        for i in range(N):
            best_j = -1
            best_iou = 0.0

            for j in range(M):
                iou = self._iou_xyxy(pred_box[i], gt_box[j])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j != -1 and best_iou >= self.iou_threshold:
                gt_matched[best_j] = True

                # Localization loss: MSE between predicted and GT xyxy
                loc_loss += float(np.sum((pred_box[i] - gt_box[best_j]) ** 2))

                # Objectness loss for matched prediction (target = 1)
                conf_loss_obj += float((obj_score[i] - 1.0) ** 2)

                # Class loss: MSE vs one-hot GT class
                one_hot = np.zeros(self.num_classes, dtype=float)
                c = int(gt_class_id[best_j])
                if 0 <= c < self.num_classes:
                    one_hot[c] = 1.0
                class_loss += float(np.sum((cls_score[i] - one_hot) ** 2))

            else:
                # No-object loss for unmatched prediction (target = 0)
                conf_loss_noobj += float((obj_score[i] - 0.0) ** 2)

        # Penalize GT boxes never matched by any prediction (false negatives)
        missed = int(np.sum(~gt_matched)) if M > 0 else 0
        if missed > 0:
            conf_loss_obj += float(missed) * (1.0 ** 2)  # (0 - 1)^2 per missed GT

        total_loss = (
            self.lambda_coord * loc_loss
            + self.lambda_obj * conf_loss_obj
            + self.lambda_noobj * conf_loss_noobj
            + self.lambda_cls * class_loss
        )

        return {
            "total_loss": float(total_loss),
            "loc_loss": float(loc_loss),
            "conf_loss_obj": float(conf_loss_obj),
            "conf_loss_noobj": float(conf_loss_noobj),
            "class_loss": float(class_loss),
        }
