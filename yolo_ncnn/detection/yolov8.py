from collections.abc import Sequence

import cv2
import numpy as np

from ..base import YOLO_NCNN
from .result import DetectionResult


class YOLOv8(YOLO_NCNN[DetectionResult]):
    def __init__(
        self,
        param_path: str,
        bin_path: str,
        meta_path: str,
        score_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        device: str = "cpu",
    ) -> None:
        """初始化YOLOv8模型.

        Args:
            param_path (str): 模型参数文件路径.
            bin_path (str): 模型二进制文件路径.
            score_threshold (float, optional): 置信度阈值. Defaults to 0.35.
            nms_threshold (float, optional): NMS阈值. Defaults to 0.45.
            device (str, optional): 设备类型. Defaults to "cpu".

        """
        super().__init__(param_path, bin_path, meta_path, device)

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

    def postprocess(
        self,
        batch_image: Sequence[np.ndarray],
        batch_result: Sequence[np.ndarray],
    ) -> Sequence[DetectionResult]:
        """对批量结果进行后处理.

        Args:
            batch_image (List[np.ndarray]): 原始图像,检测或结果可视化.
            batch_result (List[np.ndarray]): 需要进行后处理的批量结果.

        Returns:
            List[DetectionResult]: 后处理后的批量结果.

        """
        outputs = np.transpose(batch_result, (0, 2, 1))  # [1, 84, 8400] -> [1, 8400, 84]
        detections = []
        for i, batch in enumerate(outputs[: len(batch_image)]):
            # 置信度过滤
            batch = batch[batch[:, 4:].max(1) > self.score_threshold]
            if len(batch) == 0:
                detections.append(DetectionResult(batch_image[i]))
                continue

            # 检测结果切片
            boxes, classes_scores = np.split(batch, [4], axis=1)
            class_ids = np.argmax(classes_scores, axis=1)
            scores = np.amax(classes_scores, axis=1)

            # NMS
            indices = cv2.dnn.NMSBoxesBatched(boxes, scores, class_ids, self.score_threshold, self.nms_threshold)  # type: ignore
            if len(indices) == 0:
                detections.append(DetectionResult(batch_image[i]))
                continue

            # 结果切片
            boxes = boxes[indices]
            class_ids = class_ids[indices]
            scores = scores[indices]

            names = [self.names[int(class_id)] if self.names is not None else str(class_id) for class_id in class_ids]

            # 计算边界框的坐标 xywh -> xyxy
            boxes[:, :2] -= boxes[:, 2:] / 2
            boxes[:, 2:] += boxes[:, :2]

            # 修正边界框坐标
            boxes -= self.batch_paddings[i]
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, self.imgsz[1])
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, self.imgsz[0])
            boxes /= self.batch_scales[i]

            detections.append(DetectionResult(batch_image[i], boxes, scores, class_ids, names))
        return detections
