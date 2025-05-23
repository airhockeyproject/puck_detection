import cv2
import numpy as np
import sys

# 入力画像の読み込み
image = cv2.imread(sys.argv[1])
h_img, w_img = image.shape[:2]

# ArUcoマーカーの辞書と検出器
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# マーカー検出
corners, ids, _ = detector.detectMarkers(image)

if ids is not None and len(corners) >= 1:
    # 最初のマーカーの4点を使用（左上、右上、右下、左下）
    pts_src = corners[-1][0].astype(np.float32)

    # 変換先座標（マーカーを正方形として見たときの座標）
    size = 300  # マーカー平面上での1辺の長さ（仮定）
    pts_dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)

    # Homography行列を計算
    H, _ = cv2.findHomography(pts_src, pts_dst)

    # 元画像全体を変換する（出力サイズは大きめにしておく）
    warp_size = (3200, 3200)  # 仮の出力解像度
    warped = cv2.warpPerspective(image, H, warp_size)

    # 結果表示
    cv2.imshow("Warped Full Image", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("マーカーが検出できませんでした。")
