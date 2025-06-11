# puck_detection

適宜、自分のPCでcloneして動かしてください。

以下のフローで組んでいます

- マーカーによって、画像を面上の２D平面の画像に変換する
- それに対して、YOLOで中心位置を特定する

データは以下のコマンドでダウンロード

```bash
curl -L "https://app.roboflow.com/ds/rBJEa9VAyb?key=HV8NontPXK" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```

## 今後の計画

- YOLOをpuckを検知するように学習する
  - roboflowに上がっている、真上からみたパックのデータセットで学習する
  - 学習して、そのデータセットで精度検証
- 実機でデータセットを収集する
  - ホッケー台の四隅にarucoを貼り付けて実施
  - ある程度の数、100のオーダーは欲しい？
  - 以下を集計
    - 斜めから撮ったパックをホッケー台で撮った画像
    - テーブル上の座標寸法
- 実機データセットに対して検証する
  - 最終的な寸法があっているかどうか

参考

https://qiita.com/code0327/items/c6e468da7007734c897f#%E8%BF%BD%E5%8A%A0%E5%AE%9F%E9%A8%93
