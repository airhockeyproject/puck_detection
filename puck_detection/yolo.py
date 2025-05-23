import cv2
from ultralytics import YOLO


def detect_single_object_center(model: YOLO, image: cv2.typing.MatLike, object_id: int):
    config = {
        "classes": [object_id],  # 指定したオブジェクトのみを検知
        "max_det": 1,  # 最大検出数
        "half": True,  # 半精度演算 (FP16)
    }

    # 推論実行
    result = model(image, **config)[0]
    x1, y1, x2, y2 = result.boxes[0].xyxy[0].tolist()
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return (cx, cy)


if __name__ == "__main__":
    image_path = "images/1.png"
    model = YOLO("yolov8n.pt")
    image = cv2.imread(image_path)
    center_pos = detect_single_object_center(model, image, 0)  # 人だけを検出

    # 表示（ウィンドウ表示）
    cv2.circle(image, center_pos, radius=5, color=(0, 0, 255), thickness=-1)
    cv2.imshow("Detection with Center", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
