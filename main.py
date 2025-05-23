from puck_detection.homography import perform_homography, load_detector
from puck_detection.yolo import detect_single_object_center

import cv2
from ultralytics import YOLO


def main(img_path, model_path="models/yolov8n.pt"):
    corner_arrange = [2, 3, 0, 1]
    plane_size = (500, 500)
    object_id = 14  # 犬のobject_idは16だが、なぜかベンチとして認識されてしまう

    image = cv2.imread(img_path)
    yolo = YOLO(model_path)
    aruco_detector = load_detector()

    image_plane = perform_homography(aruco_detector, image, corner_arrange, plane_size)
    center_pos = detect_single_object_center(yolo, image_plane, object_id)
    print(yolo(image_plane)[0].boxes)

    # 表示（ウィンドウ表示）
    cv2.circle(image_plane, center_pos, radius=5, color=(0, 0, 255), thickness=-1)
    cv2.imshow("Detection with Center", image_plane)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img_path = "images/dog.png"
    main(img_path)
