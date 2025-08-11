from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Generic, TypeVar, overload

import ncnn
import numpy as np
import yaml

from .utils.data_load import letterbox, load_image

T = TypeVar("T")


class YOLO_NCNN(ABC, Generic[T]):
    def __init__(self, param_path: str, bin_path: str, meta_path: str, device: str = "cpu"):
        self.net = ncnn.Net()
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        if device == "gpu":
            self.net.opt.use_vulkan_compute = True

        metadata = Path(meta_path)
        if metadata.exists():
            with metadata.open("r") as f:
                data = yaml.safe_load(f)
        else:
            data = {}

        self.stride = data.get("stride", 32)
        self.task = data.get("task", "detection")
        self.names = data.get("names", {})
        self.imgsz = data.get("imgsz", (640, 640))
        self.batch_size = data.get("batch", 1)
        self.kpt_shape = data.get("kpt_shape", None)

        self.batch_scales = np.ones(self.batch_size)
        self.batch_paddings = np.zeros((self.batch_size, 4))
        self.input_tensor = np.zeros((self.batch_size, 3, *self.imgsz), dtype=np.float32)

    def _forward(self, image: np.ndarray):
        mat_in = ncnn.Mat(image)
        with self.net.create_extractor() as ex:
            ex.input(self.net.input_names()[0], mat_in)
            y = [np.array(ex.extract(x)[1])[None] for x in sorted(self.net.output_names())]  # out0
        return y[0]

    def _preprocess_single_image(self, image: np.ndarray):
        resize_shape = (640, 640)
        image, scale, padding = letterbox(image, resize_shape)
        image = image.astype(np.float32) / 255.0
        image = image.transpose((2, 0, 1))
        return image, scale, padding

    def _validate_batch_size(self, batch_size: int) -> None:
        if batch_size > self.batch_size:
            raise ValueError(
                f"批次大小过大: 期望最大 {self.batch_size}，实际收到 {batch_size}。"
                f"请减少输入图像数量或使用支持更大批次的模型。"
            )
        if batch_size <= 0:
            raise ValueError("批次大小必须大于0")

    @overload
    def predict(self, images: str | np.ndarray) -> T: ...

    @overload
    def predict(self, images: Sequence[str | np.ndarray]) -> Sequence[T]: ...

    def predict(self, images: Sequence[str | np.ndarray] | str | np.ndarray) -> T | Sequence[T]:
        """对图像进行推理.

        该方法可以处理单张图像或多张图像的批量推理.

        Args:
            images: 输入图像.可以是以下形式之一：
                - 单张图像的文件路径(str)
                - 单张图像的 numpy 数组(np.ndarray)
                - 多张图像的文件路径列表(List[str])
                - 多张图像的 numpy 数组列表(List[np.ndarray])

        Returns:
            推理结果.根据输入的不同,返回类型会有所不同：
                - 对于单张图像输入,返回单个推理结果(类型 T)
                - 对于多张图像输入,返回推理结果列表(List[T])

            其中 T 是由子类定义的具体结果类型.

        Raises:
            ValueError: 如果输入的图像格式不正确或无法加载图像.

        Note:
            具体的返回值格式取决于子类中 `postprocess` 方法的实现.
        """
        if isinstance(images, (str, np.ndarray)):  # noqa: UP038
            images = [images]
            input_is_sequence = False
        else:
            input_is_sequence = True

        self._validate_batch_size(len(images))
        batch_images = [load_image(im) for im in images]

        self.preprocess(batch_images)
        batch_result = self._forward(self.input_tensor)
        batch_result = self.postprocess(batch_images, batch_result)  # pyright: ignore

        out = batch_result if input_is_sequence else batch_result[0] if len(batch_result) == 1 else batch_result

        return out

    def preprocess(self, batch_image: Sequence[np.ndarray]) -> None:
        """对一批图像进行预处理.

        Args:
            batch_image (List[np.ndarray]): 需要进行预处理的图像列表.

        """
        for i, im in enumerate(batch_image):
            image, scale, (pad_w, pad_h) = self._preprocess_single_image(im)
            self.input_tensor[i] = image
            self.batch_scales[i] = scale
            self.batch_paddings[i] = [pad_w, pad_h, pad_w, pad_h]

    @abstractmethod
    def postprocess(self, batch_image: Sequence[np.ndarray], batch_result: Sequence[np.ndarray]) -> Sequence[T]:
        """对一批图像进行后处理.

        Args:
            batch_image (List[np.ndarray]): 需要进行后处理的图像列表.
            batch_result (List[np.ndarray]): 需要进行后处理的结果列表.

        Returns:
            List[T]: 后处理后的结果列表.
        """
        pass
