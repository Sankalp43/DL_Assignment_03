# Hindi Transliteration Using Seq2Seq Models (Vanilla & Attention)

This repository provides a complete framework for transliterating romanized Hindi (Latin script) into Devanagari using two deep learning approaches:

- ✅ A **Vanilla Sequence-to-Sequence (Seq2Seq)** model
- 🔥 An **Attention-Based Seq2Seq** model (Bahdanau-style)

Both models are built with **PyTorch** and evaluated using the **Google Dakshina dataset**. The project includes visualization tools, W&B hyperparameter sweeps, and side-by-side evaluation of model performance.

---

## 📁 Directory Overview

```bash
├── Attention_Seq_to_Seq/           # Attention-based Seq2Seq model code
├── Vanilla_Seq_to_Seq/             # Vanilla Seq2Seq model code
├── attention_predictions/          # Predictions from attention model
├── predictions_vanilla/            # Predictions from vanilla model
├── data/hi/                        # Dataset (Dakshina) in Latin & Devanagari
├── IPYNB_files/                    # Jupyter notebooks
````

---

## 📌 Dataset: Google Dakshina

The **Dakshina dataset** by Google Research includes transliteration corpora for various Indic languages. In this project, only **Hindi** data is used:

* `native_script_wikipedia/` — Hindi words in Devanagari
* `romanized/` — Romanized (Latin) versions of those words
* `lexicons/` — Vocabulary and token mappings

---

## ✨ Models

### 🔹 Vanilla Seq2Seq Model (No Attention)

A classic **encoder-decoder architecture** using RNNs (LSTM/GRU) for sequence generation.

#### 📐 Architecture

* **Encoder**:

  * Character Embedding
  * RNN/GRU/LSTM
  * Final hidden state as context vector

* **Decoder**:

  * Embedding + RNN/GRU/LSTM
  * Linear output layer
  * Teacher forcing during training

#### 🧪 Training & Evaluation

```bash
# Train
python train_model.py --embed_dim 256 --hidden_size 512 --cell_type lstm --enc_layers 2 --dec_layers 2 --dropout 0.2 --data_path path/to/dakshina_dataset_v1.0

# Evaluate
python test_model.py
```

Output: `vanilla_prediction.tsv` (Latin input, predicted Devanagari, ground truth)

#### 🗂️ Files

| File               | Purpose                    |
| ------------------ | -------------------------- |
| `vanilla_model.py` | Model architecture         |
| `dataloader.py`    | Preprocessing and batching |
| `train_model.py`   | Training loop              |
| `test_model.py`    | Evaluation script          |
| `config.yaml`      | W\&B sweep config          |
| `best_model.pth`   | Best trained weights       |

---

### 🔸 Attention-Based Seq2Seq Model

An **enhanced encoder-decoder** with **Bahdanau-style attention**, allowing dynamic focus on input tokens during decoding.

#### 📐 Architecture

* **Encoder**:

  * Embedding + RNN/GRU/LSTM
  * Returns *all hidden states* for attention

* **Attention**:

  * Alignment scores: `score(h_t, s_t-1)`
  * Attention weights via softmax
  * Context vector = weighted sum of encoder states

* **Decoder**:

  * Embedding + context → RNN + Linear layer
  * Teacher forcing & dynamic attention

#### 🎯 Key Features

* Attention Heatmaps
* LSTM Unit Visualization
* Flexible RNN layers
* W\&B integration for tracking experiments

#### 🧪 Training & Evaluation

```bash
# Train with attention
python train_model.py --embed_dim 256 --hidden_size 512 --cell_type lstm --attention true

# Evaluate
python test_model.py
```

Output: `prediction_attention.tsv` (Latin input, predicted Devanagari, ground truth)

#### 📊 Visualization

```bash
# Attention visualization
python plot_attention_heatmap.py

# LSTM activations
python plot_lstm_activations.py
```

#### 🗂️ Files

| File                        | Purpose                       |
| --------------------------- | ----------------------------- |
| `attention_model.py`        | Full model with attention     |
| `attention.py`              | Bahdanau attention logic      |
| `dataloader.py`             | Preprocessing utilities       |
| `train_model.py`            | Training loop                 |
| `test_model.py`             | Evaluation and TSV generation |
| `plot_attention_heatmap.py` | Heatmap visualizer            |
| `plot_lstm_activations.py`  | LSTM unit insights            |
| `config.yaml`               | W\&B sweep configuration      |
| `best_model.pth`            | Best trained weights          |

---

## 📊 Configuration Options

Both models support hyperparameter tuning with **Weights & Biases** via `config.yaml`.

### Key Configurations:

| Parameter                   | Options              |
| --------------------------- | -------------------- |
| `embed_dim`                 | 64, 128, 256         |
| `hidden_size`               | 128, 256, 512        |
| `cell_type`                 | `rnn`, `lstm`, `gru` |
| `dropout`                   | 0.0, 0.2, 0.3        |
| `teacher_forcing_ratio`     | 0.5, 0.7             |
| `enc_layers` / `dec_layers` | 1, 2                 |
| `learning_rate`             | 0.001, 0.0005        |

### Run a sweep:

```bash
wandb sweep config.yaml
wandb agent <SWEEP_ID>
```

---

## 🧪 Evaluation Metrics

* **Character-level accuracy**
* **Word-level accuracy**
* Side-by-side comparison of transliteration outputs in `.tsv` files

---

## 🧰 Requirements

* Python ≥ 3.7
* PyTorch
* pandas
* numpy
* tqdm
* matplotlib (for visualizations)
* wandb (for tracking)

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 📓 Jupyter Notebooks

| Notebook                     | Description                         |
| ---------------------------- | ----------------------------------- |
| `Seq_to_seq_Attention.ipynb` | Full pipeline using attention model |
| `Seq_to_seq_Vanilla.ipynb`   | Full pipeline using vanilla model   |

---

## 🚀 Results Visualization

📷 *Sample attention heatmap and LSTM activation plots available in the `Attention_Seq_to_Seq/` folder for better interpretability of the model behavior.*

---

