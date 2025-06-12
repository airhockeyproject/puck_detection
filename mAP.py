from ultralytics import YOLO

model = YOLO("results/test1_run5/weights/best.pt")
results = model.val(data="test1-2/data.yaml", split="test")

# 指標を数値として取り出す（1クラス前提）
map_50_95 = results.box.map
map_50    = results.box.map50
precision = results.box.p[0]   # ← [0] を追加
recall    = results.box.r[0]   # ← [0] を追加

# 出力
print(f"mAP@0.5:0.95 = {map_50_95:.4f}")
print(f"mAP@0.5      = {map_50:.4f}")
print(f"Precision     = {precision:.4f}")
print(f"Recall        = {recall:.4f}")
