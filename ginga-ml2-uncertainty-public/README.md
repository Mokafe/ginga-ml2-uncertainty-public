# ginga-ml2-uncertainty-public

テキスト（イベント列）を **A=接近 / B=脅威 / C=資源** の3軸に落として、  
**(3D×時間)** の「場」として復元し、**転換点候補（warp×矛盾）**を拾うPoCです。

> 重要: 作品本文そのものは同梱しません（必要なら各自で用意してください）。

---

## いちばん早い入口（Colab）

READMEに下のバッジを置くと、ワンクリックでColabが開けます。

```md
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
  https://colab.research.google.com/github/Mokafe/ginga-ml2-uncertainty-public/blob/main/notebooks/00_turning_points_colab.ipynb
)
```

---

## 置いてあるもの

- `notebooks/00_turning_points_colab.ipynb` : Colabで実行（データアップロード可）
- `src/poc_turning_points.py` : ローカル実行用PoC
- `src/kimura2.txt` : kimura2（前方モデルの実装を利用）
- `data/sample/` : 小さなサンプルデータ

---

## ローカル実行

```bash
pip install -r requirements.txt
python -m src.poc_turning_points --json data/sample/sample_g_1_reconstructed_keep_last9.json
# => assets/turning_points.png が生成されます
```
