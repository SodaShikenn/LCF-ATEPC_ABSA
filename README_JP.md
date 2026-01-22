# LCF-ATEPC-ABSA

BERTベースの**アスペクト感情分析 (ABSA: Aspect-Based Sentiment Analysis)** モデル

## これは何？

このプロジェクトは2つのタスクを同時に実行します：

1. **アスペクト抽出 (ATE: Aspect Term Extraction)**: レビュー中の製品属性を特定（例：「デザイン」「カメラ」）
2. **極性分類 (PC: Polarity Classification)**: 各アスペクトの感情を判定（ポジティブ/ネガティブ）

この2つを合わせて **ATEPC** (Aspect Term Extraction and Polarity Classification) と呼びます。

### 主要技術: LCF (Local Context Focus)

文全体を均等に分析するのではなく、**LCF**は各アスペクト周辺の単語に注目します。これにより、無関係な文脈からのノイズを減らし、精度を向上させます。

### 例

```
入力: "このスマホはデザインがおしゃれだが、カメラは残念だ。"
出力: デザイン → ポジティブ, カメラ → ネガティブ
```

## プロジェクト構成

```
code/
├── config.py       # ハイパーパラメータ
├── model.py        # BERT + LCF + CRFモデル
├── utils.py        # データセット・ユーティリティ
├── process.py      # データ前処理
├── train.py        # 学習スクリプト
├── test.py         # 評価スクリプト
└── predict.py      # 推論スクリプト
```

## セットアップ

### 1. 環境構築

```bash
# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 依存関係のインストール
pip install torch transformers pandas torchcrf
```

### 2. BERTモデル（必須）

`bert-base-chinese`をダウンロードし、親ディレクトリに配置してください：

```bash
# Hugging Face CLIを使用
huggingface-cli download bert-base-chinese --local-dir ../huggingface/bert-base-chinese

# または手動でダウンロード: https://huggingface.co/bert-base-chinese
```

期待されるディレクトリ構成：

```
parent_folder/
├── code/                        # このリポジトリ
│   ├── config.py
│   └── ...
└── huggingface/
    └── bert-base-chinese/
        ├── config.json
        ├── vocab.txt
        └── model.safetensors    # または pytorch_model.bin
```

> **注意**: 異なるパスを使用する場合は`config.py`の`BERT_MODEL_NAME`を変更してください。

### 3. 実行

```bash
# データ前処理
python process.py

# 学習
python train.py

# 評価
python test.py

# 推論
python predict.py
```

## モデルアーキテクチャ

| コンポーネント | 説明 |
|---------------|------|
| エンコーダ | BERT-base-chinese (768次元) |
| エンティティ抽出 | Linear + CRF → BIOタグ (O, B-ASP, I-ASP) |
| 感情分類 | LCF重み付け + Self-Attention → ポジティブ/ネガティブ |

**LCF (Local Context Focus)**: アスペクトからの距離に基づいてトークンを重み付けし、関連する局所文脈に注目します。

## 設定パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `SRD` | 3 | 局所文脈ウィンドウサイズ |
| `BATCH_SIZE` | 50 | バッチサイズ |
| `EPOCH` | 100 | 学習エポック数 |
| `LR` | 1e-4 | 学習率 |
| `LCF` | cdw | 戦略: cdw / cdm / fusion |

## データ形式

**入力** (1行1トークン):

```
画面  B-ASP  1
は    O      -1
綺麗  O      -1
```

**ラベル**: BIOタグ (O/B-ASP/I-ASP)、極性 (0=ネガティブ, 1=ポジティブ, -1=なし)

**ドメイン**: camera, car, laptop, notebook, phone, restaurant, twitter, mixed

## 評価指標

結合精度: エンティティ位置と感情極性の両方が正解と一致する必要があります。

```
指標: 適合率 (Precision)、再現率 (Recall)、F1スコア
```

## 参考文献

- BERT: Devlin et al.
- LCF-ATEPC: Zeng et al.
- CRF: Lafferty et al.
- [BERTベース多言語感情分析モデル: LCF-ATEPC (Qiita)](https://qiita.com/donaldchi/items/76a107fb4493490afa9e)
