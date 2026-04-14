# S.O.N.A.T.A.M.
**Symbolic Ontology & Neural Audio Transcription Architecture for Music**

### 📖 About
S.O.N.A.T.A.M. is a neuro-symbolic AI framework for music that combines a **structured Music Knowledge Graph** with **Graph Neural Networks** (GraphSAGE) for hybrid recommendation and link prediction.

---

### 🧠 Project Overview

S.O.N.A.T.A.M. builds a heterogeneous knowledge graph from large-scale symbolic music datasets and trains an **inductive GraphSAGE** model to:

* **Predict missing links** — infer genre, artist, key, and era for new pieces
* **Recommend similar music** — via learned node embeddings (content-based + collaborative)
* **Process unseen audio** — transcribe raw MP3/WAV via MT3, extract features, and predict without retraining

The key insight is that GraphSAGE learns an *aggregation function* over the graph neighbourhood, not fixed node embeddings. This makes it **inductive**: a brand-new audio file can be transcribed to MIDI, featurised, added to the graph, and scored instantly.

### ⚙️ Core Pipeline

* **Dual-Branch Feature Extraction:** jSymbolic2 (≈200 statistical features) + musif/music21 (semantic: Roman numerals, key profiles, functional harmony)
* **Heterogeneous Knowledge Graph:** RDF (rdflib) with entity nodes (Piece, Artist, Genre, Key, Era) and feature literals
* **PyTorch Geometric HeteroData:** Direct conversion from DataFrame for GNN training
* **GraphSAGE Link Prediction:** 2-layer heterogeneous message passing + MLP decoder
* **Audio Inference Pipeline:** MT3 transcription → dual-branch features → inductive GNN scoring

---

## 🗺️ Pipeline Architecture

```text
Lakh MIDI Dataset (lmd_matched/)
    │  match_scores.json (DTW quality filter)
    ▼
┌────────────────────────────────────┐
│  01. Feature Extraction            │  → curated_dataset.parquet
│  LakhMSDLinker                     │
│  ├─ jSymbolic2  (statistical)      │
│  └─ SemanticAnalyzer (semantic)    │
└────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────┐
│  02. Knowledge Graph               │  → harmonic_kg.ttl + hetero_data.pt
│  RDF (rdflib) + NetworkX           │
│  HeteroGraphConverter → PyG        │
└────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────┐
│  03. Graph Learning                │  → checkpoints/best_model.pt
│  HeteroGraphSAGE (2-layer)         │
│  LinkPredictor (MLP)               │
│  LinkPredTrainer                   │
│  Eval: AUC, AP, MRR, Hits@K       │
└────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────┐
│  04. Audio Inference               │  → predicted links + recommendations
│  MT3 audio → MIDI                  │
│  Dual-branch feature extraction    │
│  Inductive GNN scoring             │
└────────────────────────────────────┘
```

---

## 📁 Project Structure

```text
SONATAM/
├── pyproject.toml               ← installable package config
├── README.md
│
├── config/
│   ├── config.yaml              ← all paths, thresholds, hyperparameters
│   └── settings.py              ← CFG = load("config.yaml")
│
├── src/sonata/
│   ├── __init__.py
│   ├── notebook_utils.py        ← rp(), show_paths() for notebooks
│   │
│   ├── data/                    ← Feature extraction
│   │   ├── analyzer.py          ← MIDIHarmonicAnalyzer (legacy)
│   │   ├── linker.py            ← LakhMSDLinker (dual-branch pipeline)
│   │   ├── msd_reader.py        ← read_msd_metadata() for MSD HDF5
│   │   ├── jsymbolic_wrapper.py ← jSymbolic2 JAR wrapper (statistical)
│   │   └── semantic_analyzer.py ← musif / music21 (semantic features)
│   │
│   ├── kg/                      ← Knowledge graph
│   │   ├── schema.py            ← RDF ontology: classes, properties, URI factories
│   │   ├── builder.py           ← DataFrame → RDF (rdflib) + NetworkX
│   │   ├── queries.py           ← SPARQL wrappers
│   │   └── converter.py         ← DataFrame → PyTorch Geometric HeteroData
│   │
│   ├── models/                  ← Graph neural networks
│   │   ├── graph_models.py      ← HeteroGraphSAGE + LinkPredictor
│   │   ├── train.py             ← LinkPredTrainer + TrainerConfig
│   │   └── evaluate.py          ← AUC, AP, MRR, Hits@K, t-SNE plots
│   │
│   └── inference/               ← Audio inference pipeline
│       ├── audio_transcriber.py ← MT3 / basic-pitch wrapper
│       ├── feature_extractor.py ← Audio/MIDI → dual-branch features
│       └── graph_querier.py     ← Trained GNN → link predictions
│
├── notebooks/
│   ├── 01-feature-extraction.ipynb
│   ├── 02-kg-construction.ipynb
│   ├── 03-graph-learning.ipynb
│   └── 04-audio-inference.ipynb
│
├── scripts/
│   ├── download_lmd.py          ← Download Lakh MIDI Dataset assets
│   ├── build_dataset.py         ← CLI: dual-branch feature extraction
│   ├── build_kg.py              ← CLI: RDF + HeteroData construction
│   └── train_model.py           ← CLI: GraphSAGE training
│
├── tests/
│   ├── test_data/
│   ├── test_kg/
│   └── test_models/
│
├── data/
│   ├── raw/                     ← lmd_matched, h5, match_scores
│   └── processed/               ← parquet, ttl, hetero_data.pt
│
└── checkpoints/                 ← saved model weights
```

---

## 💾 Data Sources

| Dataset | Description |
|---|---|
| Lakh MIDI Dataset (LMD-matched) | 116k MIDI files matched to MSD |
| MSD HDF5 (single-song files) | Metadata + audio features |
| match_scores.json | DTW similarity for quality filtering |

**Default match-score threshold:** `≥ 0.70` (keeps ~50% of MIDI files)

---

## 🚀 Quick Start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Configure paths

Edit `config/config.yaml`:

```yaml
data:
  midi_root: "data/raw/lmd_matched"
  h5_root:   "data/raw/lmd_matched_h5"
```

### 3. Download data

```bash
python scripts/download_lmd.py
```

### 4. Build the dataset

```bash
python scripts/build_dataset.py --max-tracks 500
```

### 5. Build the KG + HeteroData

```bash
python scripts/build_kg.py
```

### 6. Train GraphSAGE

```bash
python scripts/train_model.py --data data/processed/hetero_data.pt --epochs 100
```

### 7. Inference on a new audio file

```python
from sonata.inference import FeatureExtractor, GraphQuerier

extractor = FeatureExtractor()
features = extractor.extract("my_song.mp3")

querier = GraphQuerier(model_path="checkpoints/best_model.pt",
                       hetero_data=hetero_data)
predictions = querier.predict_links(features, top_k=5)
similar = querier.recommend_similar(features, top_k=10)
```

---

## 🏗️ Architecture Details

### GraphSAGE (Inductive Heterogeneous GNN)

| Hyperparameter | Default |
|---|---|
| Hidden channels | 128 |
| GraphSAGE layers | 2 |
| Aggregator | mean |
| Link predictor MLP layers | 2 |
| Dropout | 0.3 |
| Learning rate | 1e-3 |
| Epochs | 100 |
| Early stopping patience | 15 |
| Negative sampling ratio | 1.0 |

### Node Types

| Type | Source |
|---|---|
| MusicalPiece | MSD tracks + dual-branch features |
| Artist | MSD metadata |
| Genre | MSD primary_genre |
| MusicalKey | Detected global key |
| Era | Decade from release year |

### Edge Types

| Edge | Meaning |
|---|---|
| hasArtist | Piece → Artist |
| hasGenre | Piece → Genre |
| hasGlobalKey | Piece → Key |
| hasEra | Piece → Era |

---

## 🛠️ Dependencies

See `pyproject.toml` for full list.

| Package | Purpose |
|---|---|
| `torch`, `torch-geometric` | GNN training (GraphSAGE) |
| `rdflib` | RDF knowledge graph |
| `networkx` | Graph analytics & visualisation |
| `music21` | MIDI parsing, harmonic analysis |
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Evaluation metrics |
| `torchmetrics` | AUC-ROC, Average Precision |
| `matplotlib`, `seaborn` | Visualisation |
| `h5py` | MSD HDF5 reader |
| `librosa` | Audio processing |

---

## 📄 License

This project is part of a Master's thesis. All rights reserved.
