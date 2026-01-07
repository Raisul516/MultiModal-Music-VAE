# 🎵 Multi-Modal Music Clustering using Hybrid VAE 🤖

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A PyTorch implementation of a **Hybrid Beta-Variational Autoencoder (β-VAE)** for **unsupervised music genre clustering**.  
The model fuses **Mel-Spectrogram audio features** and **TF-IDF-based lyric representations** to learn a joint disentangled latent space, achieving improved genre separation compared to unimodal baselines.

<p align="center">
  <img src="viz_tsne_top10.png" width="600" title="Latent Space Visualization (Top 10 Genres)">
</p>

---

## 🌟 Key Features

- **Multi-Modal Fusion:** Combines CNN-based encoders for audio and MLP-based encoders for lyrics.
- **Hybrid β-VAE Architecture:** Encourages a structured, continuous latent space suitable for clustering.
- **Unsupervised Learning:** No genre labels are used during training.
- **Improved Clustering Quality:** Achieves **NMI ≈ 0.139** and **high cluster purity (~0.53)** across **69 micro-genres**, outperforming PCA and standard autoencoder baselines.
- **Disentanglement:** Effectively separates overlapping genres such as *Rap*, *Pop*, and *Electronic* in latent space.

---

## 📂 Dataset & Setup (Important)

This project uses the **MERGE Dataset**, which contains paired **audio tracks and lyrics**.

⚠️ **Important:**  
Raw audio files and lyrics are **not included** in this repository due to copyright and storage limitations.

### 🔽 Dataset Download
Download the dataset from the official source:  
https://doi.org/10.5281/zenodo.13939205

### 📁 Expected Directory Structure

```text
MultiModal-Music-VAE/
├── data/
│   ├── audio/          # .wav or .mp3 files
│   ├── lyrics/         # lyric text files
│   └── metadata.csv
├── src/
├── results/
└── README.md



## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/MultiModal-Music-VAE.git](https://github.com/YOUR_USERNAME/MultiModal-Music-VAE.git)
    cd MultiModal-Music-VAE
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🧠 Usage

1. Train the Model
To train the Hybrid VAE from scratch:
```bash
python src/train.py
2. Evaluate and Visualize
To generate the clustering metrics, results table, and visualizations (t-SNE, Heatmaps):
python src/evaluate.py
3. Generate Report Plots
To reproduce the specific figures used in the final report:
python src/generate_all_report_plots.py