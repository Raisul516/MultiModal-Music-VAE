import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

from sklearn.metrics import (
    adjusted_rand_score, 
    normalized_mutual_info_score, 
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score,
    confusion_matrix
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from torch.utils.data import DataLoader
import sys
import os

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from model import HybridVAE
from dataset import HybridMusicDataset
from device import get_device, move_batch_to_device

# --- CONFIG ---
BATCH_SIZE = 32
LATENT_DIM = 32
# central device helper
device = get_device()
MODEL_PATH = "vae_final.pth"

# --- HELPER: PURITY SCORE (Required for Hard Task) ---
def purity_score(y_true, y_pred):
    # Purity = (sum of max items in each cluster) / total items
    cm = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)

# --- HELPER: SIMPLE AUTOENCODER (Baseline 2) ---
class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

def train_autoencoder_baseline(data_loader, input_dim, epochs=10):
    print(f"  > Training Baseline Autoencoder ({epochs} epochs)...")
    model = SimpleAutoencoder(input_dim, LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            audio = batch['audio'].view(batch['audio'].size(0), -1)
            lyrics = batch['lyrics']
            x = torch.cat([audio, lyrics], dim=1)
            optimizer.zero_grad()
            recon, z = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
    return model

# --- MAIN EVALUATION ---
def evaluate():
    print("="*60)
    print("STARTING FULL PROJECT EVALUATION (Easy + Medium + Hard Tasks)")
    print("="*60)
    
    # Print device information at the start
    print(f"\n[DEVICE CONFIGURATION]")
    print(f"  Device: {device}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"  ✓ Using GPU for computation")
    else:
        print(f"  ⚠ Warning: CUDA not available. Using CPU (will be slower).")
    print("="*60)
    print()
    
    # 1. LOAD DATA
    base_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(base_dir, 'data')
    dataset = HybridMusicDataset(
        metadata_file=os.path.join(data_dir, 'metadata_clean.csv'),
        audio_dir=os.path.join(data_dir, 'audio'),
        lyrics_dir=os.path.join(data_dir, 'lyrics')
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. EXTRACT RAW FEATURES & LABELS
    print("Extracting Ground Truth & Raw Features...")
    all_audio = []
    all_lyrics = []
    all_labels_raw = [] 

    for batch in dataloader:
        # FIX: Force every label to be a String. 
        # This converts empty cells (NaN) into the text "nan" so it won't crash.
        if 'label' in batch:
            clean_labels = [str(l) for l in batch['label']]
            all_labels_raw.extend(clean_labels)
        else:
            all_labels_raw.extend(["Unknown"] * batch['audio'].size(0))

        all_audio.append(batch['audio'].view(batch['audio'].size(0), -1))
        all_lyrics.append(batch['lyrics'])
        
    X_audio = torch.cat(all_audio).numpy()
    X_lyrics = torch.cat(all_lyrics).numpy()
    X_flat = np.hstack([X_audio, X_lyrics]) 
    
    # 3. ENCODE LABELS SAFELY
    # Instead of manual dictionary, use LabelEncoder which handles this automatically
    le = LabelEncoder()
    labels = le.fit_transform(all_labels_raw)
    num_classes = len(np.unique(labels))
    print(f"Detected {num_classes} unique genres: {le.classes_}")
    
    # Normalize for Baselines (Important for K-Means/PCA)
    scaler = MinMaxScaler()
    X_flat_norm = scaler.fit_transform(X_flat)

    results = []

    # --- BASELINE 1: DIRECT SPECTRAL FEATURE CLUSTERING (Hard Task) ---
    # Running K-Means directly on flattened raw data (high dim)
    print("\n[Baseline 1] Running Direct Spectral Feature Clustering...")
    kmeans_raw = KMeans(n_clusters=5, random_state=42).fit(X_flat_norm)
    results.append({
        "Method": "Direct Spectral (K-Means)",
        "Sil": silhouette_score(X_flat_norm, kmeans_raw.labels_),
        "CH": calinski_harabasz_score(X_flat_norm, kmeans_raw.labels_),
        "DB": davies_bouldin_score(X_flat_norm, kmeans_raw.labels_),
        "ARI": adjusted_rand_score(labels, kmeans_raw.labels_),
        "NMI": normalized_mutual_info_score(labels, kmeans_raw.labels_),
        "Purity": purity_score(labels, kmeans_raw.labels_)
    })

    # --- BASELINE 2: PCA + K-MEANS (Easy Task) ---
    print("[Baseline 2] Running PCA + K-Means...")
    pca = PCA(n_components=LATENT_DIM)
    z_pca = pca.fit_transform(X_flat_norm)
    kmeans_pca = KMeans(n_clusters=5, random_state=42).fit(z_pca)
    results.append({
        "Method": "PCA + K-Means",
        "Sil": silhouette_score(z_pca, kmeans_pca.labels_),
        "CH": calinski_harabasz_score(z_pca, kmeans_pca.labels_),
        "DB": davies_bouldin_score(z_pca, kmeans_pca.labels_),
        "ARI": adjusted_rand_score(labels, kmeans_pca.labels_),
        "NMI": normalized_mutual_info_score(labels, kmeans_pca.labels_),
        "Purity": purity_score(labels, kmeans_pca.labels_)
    })

    # --- BASELINE 3: AUTOENCODER + K-MEANS (Hard Task Comparison) ---
    print("[Baseline 3] Running Autoencoder + K-Means...")
    ae_model = train_autoencoder_baseline(dataloader, input_dim=X_flat.shape[1])
    ae_model.eval()
    z_ae_list = []
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            audio = batch['audio'].view(batch['audio'].size(0), -1)
            lyrics = batch['lyrics']
            x = torch.cat([audio, lyrics], dim=1)
            _, z = ae_model(x)
            z_ae_list.append(z.cpu().numpy())
    z_ae = np.vstack(z_ae_list)
    kmeans_ae = KMeans(n_clusters=5, random_state=42).fit(z_ae)
    results.append({
        "Method": "Autoencoder + K-Means",
        "Sil": silhouette_score(z_ae, kmeans_ae.labels_),
        "CH": calinski_harabasz_score(z_ae, kmeans_ae.labels_),
        "DB": davies_bouldin_score(z_ae, kmeans_ae.labels_),
        "ARI": adjusted_rand_score(labels, kmeans_ae.labels_),
        "NMI": normalized_mutual_info_score(labels, kmeans_ae.labels_),
        "Purity": purity_score(labels, kmeans_ae.labels_)
    })

    # --- PROPOSED METHOD: HYBRID VAE (The Star) ---
    print("\n[Proposed] Evaluating Hybrid VAE...")
    vae = HybridVAE(latent_dim=LATENT_DIM)
    vae.to(device)
    
    # Load Weights
    weights_path = os.path.join(base_dir, MODEL_PATH)
    if not os.path.exists(weights_path):
         weights_path = os.path.join(current_dir, "vae_final.pth")
    
    if os.path.exists(weights_path):
        vae.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"  > Loaded weights from {weights_path}")
    else:
        print("  > WARNING: Weights not found! Using random weights.")

    vae.eval()
    z_vae_list = []
    # Report runtime device status before inference
    print(f"Runtime device: {device}; cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA memory allocated (before): {torch.cuda.memory_allocated()}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = move_batch_to_device(batch, device)
            audio = batch['audio']
            lyrics = batch['lyrics']

            # Print once at start to confirm tensors and model are on GPU
            if batch_idx == 0:
                print(f"Batch 0 audio device: {audio.device}; lyrics device: {lyrics.device}")
                if torch.cuda.is_available():
                    print(f"CUDA memory allocated (after): {torch.cuda.memory_allocated()}")

            mu, _ = vae.encode(audio, lyrics)
            z_vae_list.append(mu.cpu().numpy())
    z_vae = np.vstack(z_vae_list)

    # --- MEDIUM TASK: COMPARE CLUSTERING ALGORITHMS ---
    # Algorithm A: K-Means
    kmeans_vae = KMeans(n_clusters=5, random_state=42).fit(z_vae)
    results.append({
        "Method": "Hybrid VAE + K-Means",
        "Sil": silhouette_score(z_vae, kmeans_vae.labels_),
        "CH": calinski_harabasz_score(z_vae, kmeans_vae.labels_),
        "DB": davies_bouldin_score(z_vae, kmeans_vae.labels_),
        "ARI": adjusted_rand_score(labels, kmeans_vae.labels_),
        "NMI": normalized_mutual_info_score(labels, kmeans_vae.labels_),
        "Purity": purity_score(labels, kmeans_vae.labels_)
    })

    # Algorithm B: Agglomerative Clustering
    agg_vae = AgglomerativeClustering(n_clusters=5).fit(z_vae)
    results.append({
        "Method": "Hybrid VAE + Agglomerative",
        "Sil": silhouette_score(z_vae, agg_vae.labels_),
        "CH": calinski_harabasz_score(z_vae, agg_vae.labels_),
        "DB": davies_bouldin_score(z_vae, agg_vae.labels_),
        "ARI": adjusted_rand_score(labels, agg_vae.labels_),
        "NMI": normalized_mutual_info_score(labels, agg_vae.labels_),
        "Purity": purity_score(labels, agg_vae.labels_)
    })

    # Algorithm C: DBSCAN (Density Based)
    # Note: DBSCAN often finds noise (-1). We filter it for fair metric calculation if needed, 
    # but here we leave it to show density properties.
    dbscan = DBSCAN(eps=3.0, min_samples=5).fit(z_vae)
    # DBSCAN might result in 1 cluster or noise, metrics might fail if < 2 labels
    if len(set(dbscan.labels_)) > 1:
        results.append({
            "Method": "Hybrid VAE + DBSCAN",
            "Sil": silhouette_score(z_vae, dbscan.labels_),
            "CH": calinski_harabasz_score(z_vae, dbscan.labels_),
            "DB": davies_bouldin_score(z_vae, dbscan.labels_),
            "ARI": adjusted_rand_score(labels, dbscan.labels_),
            "NMI": normalized_mutual_info_score(labels, dbscan.labels_),
            "Purity": purity_score(labels, dbscan.labels_)
        })
    else:
         results.append({
            "Method": "Hybrid VAE + DBSCAN", "Sil": 0, "CH": 0, "DB": 0, "ARI": 0, "NMI": 0, "Purity": 0
        })

    # --- PRINT FINAL TABLE ---
    df_results = pd.DataFrame(results)
    # Reorder columns for readability
    cols = ["Method", "NMI", "ARI", "Purity", "Sil", "CH", "DB"]
    print("\n" + "="*85)
    print("FINAL RESULTS TABLE (Higher is better for NMI/ARI/Purity/Sil/CH. Lower is better for DB)")
    print("="*85)
    print(df_results[cols].to_string(index=False, float_format="%.4f"))
    print("="*85)

    # --- VISUALIZATION (Hard Task: t-SNE) ---
    print("\nGenerating Visualizations (t-SNE for High Quality)...")
    plt.figure(figsize=(20, 6))
    
    # 1. PCA Baseline
    plt.subplot(1, 3, 1)
    sns.scatterplot(x=z_pca[:,0], y=z_pca[:,1], hue=labels, palette="viridis", s=15, legend='full')
    plt.title("Baseline: PCA (2D Projection)")
    
    # 2. Autoencoder Baseline (t-SNE)
    tsne = TSNE(n_components=2, random_state=42)
    z_ae_tsne = tsne.fit_transform(z_ae)
    plt.subplot(1, 3, 2)
    sns.scatterplot(x=z_ae_tsne[:,0], y=z_ae_tsne[:,1], hue=labels, palette="viridis", s=15, legend=False)
    plt.title("Baseline: Autoencoder (t-SNE)")

    # 3. Hybrid VAE (t-SNE)
    z_vae_tsne = tsne.fit_transform(z_vae)
    plt.subplot(1, 3, 3)
    sns.scatterplot(x=z_vae_tsne[:,0], y=z_vae_tsne[:,1], hue=labels, palette="viridis", s=15, legend=False)
    plt.title("Proposed: Hybrid VAE (t-SNE)")
    
    plt.savefig("viz_final_comparison.png")
    print("Saved high-res plot to 'viz_final_comparison.png'")

if __name__ == "__main__":
    evaluate()