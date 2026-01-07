import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridVAE(nn.Module):
    def __init__(self, input_shape=(1, 64, 1200), lyric_dim=100, latent_dim=32):
        """
        Standard Hybrid VAE that reconstructs BOTH Audio and Lyrics.
        """
        super(HybridVAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # --- 1. AUDIO ENCODER (CNN) ---
        # Input: (Batch, 1, 64, 1200)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        
        # Flatten size: 256 channels * 4 freq * 75 time = 76800
        self.flatten_size = 256 * 4 * 75 
        
        self.fc_audio = nn.Linear(self.flatten_size, 512)

        # --- 2. LYRIC ENCODER (Dense) ---
        self.fc_lyrics = nn.Linear(lyric_dim, 128)
        
        # --- 3. LATENT SPACE (Mean & Var) ---
        # Audio (512) + Lyrics (128) = 640
        self.fc_mu = nn.Linear(512 + 128, latent_dim)
        self.fc_logvar = nn.Linear(512 + 128, latent_dim)

        # --- 4. DECODER ---
        self.decoder_input = nn.Linear(latent_dim, 512 + 128)
        
        # Audio Decoder Path
        self.fc_decode_audio = nn.Linear(512, self.flatten_size)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
        
        # Lyric Decoder Path
        self.fc_decode_lyrics = nn.Linear(128, lyric_dim)

    def encode(self, audio, lyrics):
        # Audio
        x = F.relu(self.conv1(audio))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1) 
        h_audio = F.relu(self.fc_audio(x))
        
        # Lyrics
        h_lyrics = F.relu(self.fc_lyrics(lyrics))
        
        # Combine
        h_combined = torch.cat([h_audio, h_lyrics], dim=1)
        
        mu = self.fc_mu(h_combined)
        logvar = self.fc_logvar(h_combined)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Split back into Audio/Lyric paths
        h_decoded = F.relu(self.decoder_input(z))
        h_audio, h_lyrics = torch.split(h_decoded, [512, 128], dim=1)
        
        # Audio Reconstuction
        x = F.relu(self.fc_decode_audio(h_audio))
        x = x.view(x.size(0), 256, 4, 75) 
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        recon_audio = torch.sigmoid(self.deconv4(x)) # Output 0-1
        
        # Lyric Reconstruction
        recon_lyrics = self.fc_decode_lyrics(h_lyrics)
        
        return recon_audio, recon_lyrics

    def forward(self, audio, lyrics):
        mu, logvar = self.encode(audio, lyrics)
        z = self.reparameterize(mu, logvar)
        
        # This returns 2 items
        recon_audio, recon_lyrics = self.decode(z)
        
        # TOTAL RETURN: 4 ITEMS (Matches train.py)
        return recon_audio, recon_lyrics, mu, logvar