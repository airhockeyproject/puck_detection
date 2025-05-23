import cv2
from cv2 import aruco
from typing import List, Tuple
import numpy as np


def load_detector():
    # ArUcoマーカーの辞書と検出器初期化
    aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    return detector


def perform_homography(
    detector: aruco.ArucoDetector,
    image: cv2.typing.MatLike,
    corner_arrange: List[int],
    size_trans: Tuple[int],
):
    width, height = size_trans
    corners, ids, _ = detector.detectMarkers(image)

    # 特定のID順にマーカーを並べて、ID対応する特定の角を抽出
    corners_sorted = np.zeros_like(corners)
    for i, corner in zip(ids, corners):
        corners_sorted[i] = corner
    marker_coordinates = np.float32(
        [corner[0][i] for corner, i in zip(corners_sorted, corner_arrange)]
    )

    # 画像をホモグラフィー変換
    true_coordinates = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    trans_mat = cv2.getPerspectiveTransform(marker_coordinates, true_coordinates)
    image_trans = cv2.warpPerspective(image, trans_mat, (width, height))
    return image_trans


if __name__ == "__main__":
    img = cv2.imread("images/2D.png")

    detector = load_detector()
    corners, ids, _ = detector.detectMarkers(img)
    img_trans = perform_homography(detector, img, [2, 3, 0, 1], (500, 500))

    # 結果表示
    cv2.imshow("Warped Full Image", img_trans)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
