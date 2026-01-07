import torch
import os, sys
from torch.utils.data import DataLoader

# ensure src is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from dataset import HybridMusicDataset
from model import HybridVAE
from device import get_device, move_batch_to_device

BASE_DIR = os.path.dirname(current_dir)
DATA_DIR = os.path.join(BASE_DIR, 'data')

device = get_device()
print('Using device:', device)

# small dataloader
ds = HybridMusicDataset(os.path.join(DATA_DIR, 'metadata_clean.csv'), os.path.join(DATA_DIR,'audio'), os.path.join(DATA_DIR,'lyrics'))
loader = DataLoader(ds, batch_size=4, shuffle=False)

model = HybridVAE(latent_dim=32)
model.to(device)
model.eval()

with torch.no_grad():
    for i, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)
        audio = batch['audio']
        lyrics = batch['lyrics']
        print('audio device:', audio.device)
        print('lyrics device:', lyrics.device)
        mu, logvar = model.encode(audio, lyrics)
        print('mu device:', mu.device)
        print('cuda memory allocated:', torch.cuda.memory_allocated() if torch.cuda.is_available() else 0)
        break

print('Done')
