from pathlib import Path

import numpy as np

from yolo_ncnn.detection import YOLO11, YOLO12, YOLOv8


def test_yolov8():
    param_path = "models/yolov8n_ncnn_model/model.ncnn.param"
    bin_path = "models/yolov8n_ncnn_model/model.ncnn.bin"
    meta_path = "models/yolov8n_ncnn_model/metadata.yaml"

    detector = YOLOv8(param_path=param_path, bin_path=bin_path, meta_path=meta_path)

    # Load test image
    image_path = "images/bus.jpg"

    # Predict
    result = detector.predict(image_path)

    # Draw result
    save_path = image_path.replace("images", "results")
    save_path = save_path.replace(".jpg", "_yolov8.jpg")
    image = result.draw(save_path=save_path)

    # Check result
    assert len(result) > 0
    assert Path(save_path).exists()

    # Load empty image
    empty_image = np.ones_like(image) * 114
    result = detector.predict(empty_image)
    # Check result
    assert len(result) == 0


def test_yolo11():
    param_path = "models/yolo11n_ncnn_model/model.ncnn.param"
    bin_path = "models/yolo11n_ncnn_model/model.ncnn.bin"
    meta_path = "models/yolo11n_ncnn_model/metadata.yaml"

    detector = YOLO11(param_path=param_path, bin_path=bin_path, meta_path=meta_path)

    # Load test image
    image_path = "images/bus.jpg"

    # Predict
    result = detector.predict(image_path)

    # Draw result
    save_path = image_path.replace("images", "results")
    save_path = save_path.replace(".jpg", "_yolo11.jpg")
    image = result.draw(save_path=save_path)

    # Check result
    assert len(result) > 0
    assert Path(save_path).exists()

    # Load empty image
    empty_image = np.ones_like(image) * 114
    result = detector.predict(empty_image)
    # Check result
    assert len(result) == 0


def test_yolo12():
    param_path = "models/yolo12n_ncnn_model/model.ncnn.param"
    bin_path = "models/yolo12n_ncnn_model/model.ncnn.bin"
    meta_path = "models/yolo12n_ncnn_model/metadata.yaml"

    detector = YOLO12(param_path=param_path, bin_path=bin_path, meta_path=meta_path)

    # Load test image
    image_path = "images/bus.jpg"

    # Predict
    result = detector.predict(image_path)

    # Draw result
    save_path = image_path.replace("images", "results")
    save_path = save_path.replace(".jpg", "_yolo12.jpg")
    image = result.draw(save_path=save_path)

    # Check result
    assert len(result) > 0
    assert Path(save_path).exists()

    # Load empty image
    empty_image = np.ones_like(image) * 114
    result = detector.predict(empty_image)
    # Check result
    assert len(result) == 0
