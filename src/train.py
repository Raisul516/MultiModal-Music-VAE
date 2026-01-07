import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time

# Import your custom modules
# Ensure dataset.py and model.py are in the 'src' folder
from dataset import HybridMusicDataset
from model import HybridVAE

# --- HYPERPARAMETERS ---
BATCH_SIZE = 16          # Efficient for RTX 3070
LEARNING_RATE = 1e-3
EPOCHS = 50              # Standard training duration
LATENT_DIM = 32          # Size of the "brain" vector

# --- SETUP DEVICE ---
# This ensures we use your RTX 3070
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Training Configuration ---")
print(f"Device:       {device}")
print(f"Batch Size:   {BATCH_SIZE}")
print(f"Latent Dim:   {LATENT_DIM}")
print(f"------------------------------")

# --- 1. PREPARE DATA ---
# This automatically finds your 'data' folder relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

print("Loading dataset...")
dataset = HybridMusicDataset(
    metadata_file=os.path.join(DATA_DIR, 'metadata_clean.csv'),
    audio_dir=os.path.join(DATA_DIR, 'audio'),
    lyrics_dir=os.path.join(DATA_DIR, 'lyrics')
)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 2. INITIALIZE MODEL ---
model = HybridVAE(latent_dim=LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 3. DEFINE LOSS FUNCTION ---
def vae_loss_function(recon_audio, audio, recon_lyrics, lyrics, mu, logvar):
    # A. Reconstruction Loss (Audio) - MSE
    audio_loss = torch.nn.functional.mse_loss(recon_audio, audio, reduction='sum')
    
    # B. Reconstruction Loss (Lyrics) - MSE
    lyric_loss = torch.nn.functional.mse_loss(recon_lyrics, lyrics, reduction='sum')*50  # Weight lyrics more heavily
    
    # C. KL Divergence (The "Clustering" Enforcer)
    # Prevents the model from cheating; forces a smooth latent space.
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return audio_loss + lyric_loss + kld_loss

# --- 4. TRAINING LOOP ---
print("\nStarting Training...")
model.train()
start_time = time.time()

for epoch in range(EPOCHS):
    total_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        # Move data to GPU
        audio = batch['audio'].to(device)
        lyrics = batch['lyrics'].to(device)
        
        # Zero Gradients
        optimizer.zero_grad()
        
        # Forward Pass
        recon_audio, recon_lyrics, mu, logvar = model(audio, lyrics)
        
        # Calculate Loss
        loss = vae_loss_function(recon_audio, audio, recon_lyrics, lyrics, mu, logvar)
        
        # Backward Pass (Learn)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print progress every 20 batches
        if batch_idx % 20 == 0:
            print(f"  > Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item() / BATCH_SIZE:.2f}")

    # End of Epoch Report
    avg_loss = total_loss / len(dataset)
    print(f"=== Epoch {epoch+1}/{EPOCHS} Complete. Avg Loss: {avg_loss:.4f} ===")

    # Save Checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        save_path = f"vae_model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved checkpoint: {save_path}")

# Final Save
total_time = time.time() - start_time
print(f"\nTraining Complete in {total_time/60:.2f} minutes.")

# SAVE THE FILE
torch.save(model.state_dict(), "vae_final.pth")
print("Saved final model to 'vae_final.pth' in the current folder.")