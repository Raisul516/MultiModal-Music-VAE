import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder
import os
import sys

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from model import HybridVAE
from dataset import HybridMusicDataset
from device import get_device, move_batch_to_device

# --- CONFIG ---
BATCH_SIZE = 32
LATENT_DIM = 32
DEVICE = get_device()
MODEL_PATH = "vae_final.pth"

def generate_all_plots():
    print("="*60)
    print("GENERATING FINAL REPORT VISUALIZATIONS")
    print("="*60)
    
    # Print device information
    print(f"\n[DEVICE CONFIGURATION]")
    print(f"  Device: {DEVICE}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  [OK] Using GPU for computation")
    else:
        print(f"  [WARNING] CUDA not available. Using CPU (will be slower).")
    print("="*60)
    print()

    # 1. SETUP
    base_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(base_dir, 'data')
    
    # Load Dataset
    dataset = HybridMusicDataset(
        metadata_file=os.path.join(data_dir, 'metadata_clean.csv'),
        audio_dir=os.path.join(data_dir, 'audio'),
        lyrics_dir=os.path.join(data_dir, 'lyrics')
    )
    # Shuffle for random reconstruction example
    dataloader_shuffle = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    # No shuffle for consistent clustering analysis
    dataloader_full = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load Model
    vae = HybridVAE(latent_dim=LATENT_DIM).to(DEVICE)
    weights_path = os.path.join(base_dir, MODEL_PATH)
    if not os.path.exists(weights_path):
         weights_path = os.path.join(current_dir, "vae_final.pth")
    
    if os.path.exists(weights_path):
        vae.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        print("Model loaded successfully.")
    else:
        print("ERROR: Weights not found!")
        return
    vae.eval()

    # ============================================
    # PLOT 1: RECONSTRUCTION (Evidence of Learning)
    # ============================================
    print("\n[1/3] Generating Reconstruction Plot...")
    batch = next(iter(dataloader_shuffle))
    batch = move_batch_to_device(batch, DEVICE)
    audio_in = batch['audio']
    lyrics_in = batch['lyrics']
    
    with torch.no_grad():
        recon_audio, _, _, _ = vae(audio_in, lyrics_in)
    
    orig_spec = audio_in[0].cpu().numpy().squeeze()
    recon_spec = recon_audio[0].cpu().numpy().squeeze()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(orig_spec, aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=1)
    plt.title("Original Input")
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(recon_spec, aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=1)
    plt.title("VAE Reconstructed")
    plt.yticks([])
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig("viz_reconstruction.png", dpi=300)
    print("Saved 'viz_reconstruction.png'")

    # ============================================
    # PREPARE DATA FOR PLOTS 2 & 3
    # ============================================
    print("\nProcessing dataset for Clustering Analysis...")
    all_z = []
    all_labels_raw = []
    
    with torch.no_grad():
        for batch in dataloader_full:
            if 'label' in batch:
                clean_labels = [str(l) for l in batch['label']]
                all_labels_raw.extend(clean_labels)
            else:
                all_labels_raw.extend(["Unknown"] * batch['audio'].size(0))
            
            batch = move_batch_to_device(batch, DEVICE)
            a = batch['audio']
            l = batch['lyrics']
            mu, _ = vae.encode(a, l)
            all_z.append(mu.cpu().numpy())

    z_data = np.vstack(all_z)
    labels_data = np.array(all_labels_raw)

    # ============================================
    # PLOT 2: CLEAN TOP-10 GENRE t-SNE (The Readable One!)
    # ============================================
    print("[2/3] Generating Top-10 Genre t-SNE...")
    counts = pd.Series(labels_data).value_counts()
    top_10_genres = counts.head(10).index.tolist()
    
    # Filter dataset to only keep these 10 genres
    mask = np.isin(labels_data, top_10_genres)
    z_filtered = z_data[mask]
    labels_filtered = labels_data[mask]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_tsne = tsne.fit_transform(z_filtered)
    
    plt.figure(figsize=(10, 8))
    # Use tab10 palette for clear distinction between 10 items
    sns.scatterplot(x=z_tsne[:,0], y=z_tsne[:,1], hue=labels_filtered, palette="tab10", s=60, alpha=0.8)
    plt.title("Latent Space: Top 10 Genres Only (Clean View)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("viz_tsne_top10.png", dpi=300)
    print("Saved 'viz_tsne_top10.png'")

    # ============================================
    # PLOT 3: CONFUSION / PURITY HEATMAP
    # ============================================
    print("[3/3] Generating Confusion Heatmap...")
    
    # We cluster the filtered data into 10 clusters to see if they match the 10 genres
    kmeans = KMeans(n_clusters=10, random_state=42)
    cluster_labels = kmeans.fit_predict(z_filtered) 
    
    # Create Matrix: Rows=True Genre, Cols=Predicted Cluster
    cm = contingency_matrix(labels_filtered, cluster_labels)
    
    plt.figure(figsize=(10, 8))
    df_cm = pd.DataFrame(cm, index=np.unique(labels_filtered), columns=[f"Cluster {i}" for i in range(10)])
    
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Cluster Purity Heatmap (Top 10 Genres)")
    plt.ylabel("True Genre Label")
    plt.xlabel("Predicted Cluster Assignment")
    plt.tight_layout()
    plt.savefig("viz_confusion_heatmap.png", dpi=300)
    print("Saved 'viz_confusion_heatmap.png'")
    
    print("\nDONE. 3 New images created.")

if __name__ == "__main__":
    generate_all_plots()