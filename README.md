# S.O.N.A.T.A.
**Symbolic Ontology & Neural Audio Transcription Architecture**

### 📖 About
S.O.N.A.T.A. is a neuro-symbolic AI framework for music. It provides a full end-to-end pipeline from MIDI and audio metadata to a structured Music Knowledge Graph and downstream Deep Learning models. 

---

### 🧠 Project Overview
Automatic Music Transcription (AMT) models often struggle with polyphonic audio, hallucinating notes that make acoustic sense but violate fundamental music theory. S.O.N.A.T.A. solves this by bridging the gap between acoustic perception and the "Platonic rules" of music.

By mining massive symbolic datasets (like the Lakh MIDI Dataset), S.O.N.A.T.A. automatically extracts the statistical and theoretical rules of voice-leading, harmony, and key modulation. It then formalizes these rules into a highly structured Knowledge Graph. When paired with downstream Deep Learning models (like MT3), this graph acts as a mathematical prior—forcing the acoustic neural network to weigh what it *hears* against what is theoretically *probable* within a specific musical genre.

### ⚙️ Core Pipeline Features
* **Automated Rule Mining:** Parses raw, multi-track MIDI files using dynamic harmonic pooling and rolling windows to extract clean, root-position chord progressions.
* **Bilingual Ontology:** Maps absolute acoustic events (Harte notation) to their relative functional theory (Roman Numerals) based on dynamically detected local key contexts.
* **The Music Knowledge Graph:** Generates a dense, probabilistic RDF/Turtle graph mapping genres, structural sequences, and harmonic rules.
* **Neuro-Symbolic Integration:** Formats structural metadata (Keys, Chords) into discrete tokens or continuous embeddings to inject music theory priors directly into Transformer encoder-decoder architectures.

---

## 🗺️ Pipeline Architecture

```text
Lakh MIDI Dataset (lmd_matched/)
        │  match_scores.json (DTW quality filter)
        ▼
┌─────────────────────────────────┐
│  01. Dataset Curation           │  → curated_dataset.parquet
│  MIDIHarmonicAnalyzer           │
│  LakhMSDLinker + MSD HDF5       │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  02. Knowledge Graph            │  → harmonic_kg.ttl / .graphml
│  RDF (rdflib) + NetworkX        │
│  SPARQL query helpers           │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  03. Deep Learning              │  → checkpoints/best.pt
│  GenreClassifier (MLP)          │
│  ChordTransformer               │
│  Trainer + Evaluator            │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  04. Generation                 │  → output/*.mid / *.musicxml
│  Chord progression → MIDI       │
│  Chord progression → MusicXML   │
└─────────────────────────────────┘
```

---

## 📁 Project Structure

```text
SONATA/
├── README.md
├── requirements.txt
│
├── config/
│   ├── config.yaml          ← all paths, thresholds, hyperparameters
│   └── config.py            ← CFG = load("config.yaml")
│
├── 01_dataset/
│   ├── harmonic_analyzer.py ← MIDIHarmonicAnalyzer class
│   ├── msd_reader.py        ← read_msd_metadata() for MSD HDF5
│   ├── linker.py            ← LakhMSDLinker (MIDI ↔ HDF5 + match scores)
│   ├── notebooks/
│   │   └── 01_dataset_exploration.ipynb
│   └── output/              ← curated_dataset.parquet / .csv
│
├── 02_knowledge_graph/
│   ├── schema.py            ← RDF namespace + URI factory
│   ├── builder.py           ← DataFrame → rdflib Graph / NetworkX
│   ├── queries.py           ← SPARQL wrappers + graph traversal
│   ├── notebooks/
│   │   └── 02_kg_construction.ipynb
│   └── output/              ← harmonic_kg.ttl / .nt / .graphml
│
├── 03_deep_learning/
│   ├── dataset.py           ← HarmonicDataset (PyTorch Dataset)
│   ├── train.py             ← Trainer + TrainerConfig
│   ├── evaluate.py          ← classification_report, confusion matrix, t-SNE
│   ├── models/
│   │   ├── genre_classifier.py  ← MLP (feature vector → genre)
│   │   └── sequence_model.py    ← ChordTransformer (token seq → genre / next chord)
│   ├── notebooks/
│   │   └── 03_model_training.ipynb
│   └── checkpoints/
│
└── 04_generation/
    ├── midi_writer.py       ← write_chord_midi()
    ├── musicxml_writer.py   ← write_musicxml()
    ├── notebooks/
    │   └── 04_generation.ipynb
    └── output/              ← generated .mid / .musicxml files
```

---

## 💾 Data Sources & Thresholds

| Dataset | Location | Description |
|---|---|---|
| Lakh MIDI Dataset (LMD-matched) | `tegridy-tools/tegridy-tools/lmd_matched/` | 116k MIDI files matched to MSD |
| MSD HDF5 (single-song files) | `tegridy-tools/tegridy-tools/lmd_matched_h5/` | 31k HDF5 files with audio features |
| match_scores.json | `match_scores.json` | DTW similarity scores (quality filter) |

**Match Score Thresholds**

| Threshold | MIDI files kept | Tracks kept | Recommended for |
|---|---|---|---|
| ≥ 0.80 | 1.4% | 5.7% | Music theory / KG (high purity) |
| **≥ 0.70** | **50.5%** | **60.4%** | **Genre classification (default)** |
| ≥ 0.65 | 73.1% | 80.3% | Exploratory analysis |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r SONATA/requirements.txt
```

### 2. Configure paths

Edit `SONATA/config/config.yaml` to point to your local data:

```yaml
data:
  midi_root: "tegridy-tools/tegridy-tools/lmd_matched"
  h5_root:   "tegridy-tools/tegridy-tools/lmd_matched_h5"
  match_scores_path: "match_scores.json"
```

### 3. Build the curated dataset

```python
import json
from harmonic_kg_project.dataset import MIDIHarmonicAnalyzer, LakhMSDLinker
from harmonic_kg_project.config.config import CFG

with open(CFG["data"]["match_scores_path"]) as f:
    match_scores = json.load(f)

analyzer = MIDIHarmonicAnalyzer(
    key_window    = CFG["analyzer"]["key_window"],
    key_confidence= CFG["analyzer"]["key_confidence"],
)
linker = LakhMSDLinker(
    midi_root    = CFG["data"]["midi_root"],
    h5_root      = CFG["data"]["h5_root"],
    analyzer     = analyzer,
    match_scores = match_scores,
)

df = linker.build_dataset(
    min_score   = CFG["dataset"]["min_match_score"],
    pick_midi   = CFG["dataset"]["pick_midi"],
    max_tracks  = 500,   # set None for full dataset
)
df.to_parquet(CFG["dataset"]["parquet_file"], index=False)
```

### 4. Build the Knowledge Graph

```python
from harmonic_kg_project.knowledge_graph import KGBuilder, KGQueries

builder = KGBuilder()
g       = builder.from_dataframe(df)
builder.save(g, CFG["knowledge_graph"]["turtle_file"])

q   = KGQueries(g)
print(q.summary())
print(q.genre_distribution().head(20))
```

### 5. Train a genre classifier

```python
from harmonic_kg_project.deep_learning import HarmonicDataset
from harmonic_kg_project.deep_learning.models import GenreClassifier
from harmonic_kg_project.deep_learning.train import Trainer, TrainerConfig

dataset  = HarmonicDataset(df, label_col="primary_genre", mode="classification")
idx_train, idx_val, _ = dataset.split()

from torch.utils.data import Subset
train_loader = Subset(dataset, idx_train)
val_loader   = Subset(dataset, idx_val)

model   = GenreClassifier(input_dim=dataset.input_dim, num_classes=dataset.num_classes)
trainer = Trainer(model, train_loader, val_loader, TrainerConfig(epochs=30))
history = trainer.fit()
```

### 6. Export a chord progression to MIDI / MusicXML

```python
from harmonic_kg_project.generation import write_chord_midi, write_musicxml

chords = ["C:maj", "A:min", "F:maj", "G:7", "C:maj"]
write_chord_midi(chords,  "output/my_progression.mid",     tempo=120)
write_musicxml  (chords,  "output/my_progression.musicxml", tempo=120)
```

---

## 🛠️ Dependencies

See `requirements.txt` for pinned versions.

| Package | Purpose |
|---|---|
| `music21` | MIDI parsing, harmonic analysis, score export |
| `h5py` | MSD HDF5 reader |
| `pandas`, `numpy` | Data manipulation |
| `rdflib` | RDF knowledge graph |
| `networkx` | Graph analytics |
| `torch` | Deep learning |
| `scikit-learn` | Data splits, evaluation metrics |
| `matplotlib`, `seaborn` | Visualisation |
| `pyyaml` | Config loading |