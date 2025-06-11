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
    size_trans: Tuple[int, int],
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


def perform_full_image_homography(
    detector: aruco.ArucoDetector,
    image: cv2.typing.MatLike,
    corner_arrange: List[int],
):
    corners, ids, _ = detector.detectMarkers(image)

    # 特定のID順にマーカーを並べて、ID対応する特定の角を抽出
    corners_sorted = np.zeros_like(corners)
    for i, corner in zip(ids, corners):
        corners_sorted[i] = corner
    marker_coordinates = np.float32(
        [corner[0][i] for corner, i in zip(corners_sorted, corner_arrange)]
    )

    # 基準となる正方形（単位長の平面）を作成（ここでは仮に1x1の正方形）
    dst_coords = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]) * 500
    H = cv2.getPerspectiveTransform(marker_coordinates, dst_coords)

    # 元画像の4隅をホモグラフィー変換
    h_img, w_img = image.shape[:2]
    corners_orig = np.float32([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]]).reshape(
        -1, 1, 2
    )
    corners_transformed = cv2.perspectiveTransform(corners_orig, H)

    # 変換後の座標の最小・最大を求めて新しい画像サイズを計算
    corners_transformed = corners_transformed.reshape(-1, 2)
    min_xy = np.min(corners_transformed, axis=0)
    max_xy = np.max(corners_transformed, axis=0)
    size = np.ceil(max_xy - min_xy).astype(int)

    # 平行移動（負の座標がないようにする）
    offset = -min_xy
    trans_mat = np.array(
        [[1, 0, offset[0]], [0, 1, offset[1]], [0, 0, 1]], dtype=np.float32
    )
    H_total = trans_mat @ H  # 3x3行列同士の積

    # warpPerspectiveで画像全体を変換
    warped = cv2.warpPerspective(image, H_total, tuple(size))

    return warped


if __name__ == "__main__":
    img = cv2.imread("images/2D.png")

    detector = load_detector()
    corners, ids, _ = detector.detectMarkers(img)
    img_trans = perform_homography(detector, img, [2, 3, 0, 1], (500, 500))

    # 結果表示
    cv2.imshow("Warped Full Image", img_trans)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
