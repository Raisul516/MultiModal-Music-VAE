import os
import torch
import pandas as pd
import numpy as np
import librosa
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer # <--- NEW IMPORT

class HybridMusicDataset(Dataset):
    def __init__(self, metadata_file, audio_dir, lyrics_dir, target_shape=(64, 1200)):
        """
        Args:
            metadata_file (str): Path to metadata_clean.csv
            audio_dir (str): Path to audio folder
            lyrics_dir (str): Path to lyrics folder
        """
        # 1. Load Metadata
        try:
            self.metadata = pd.read_csv(metadata_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {metadata_file}.")

        self.audio_dir = audio_dir
        self.lyrics_dir = lyrics_dir
        self.target_shape = target_shape

        # --- NEW: PRE-PROCESS LYRICS ---
        print("Building Lyrics Vocabulary (TF-IDF)...")
        texts = []
        valid_indices = []
        
        # Read all text files once
        for idx in range(len(self.metadata)):
            row = self.metadata.iloc[idx]
            lyric_path = os.path.join(self.lyrics_dir, f"{row['Lyric_Song']}.txt")
            try:
                with open(lyric_path, 'r', encoding='utf-8', errors='ignore') as f:
                    texts.append(f.read())
                valid_indices.append(idx)
            except Exception:
                # If a lyric file is missing, we use an empty string
                texts.append("")
                valid_indices.append(idx)

        # Create Vectors (Top 100 important words)
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.lyric_vectors = self.vectorizer.fit_transform(texts).toarray()
        
        print(f"Dataset ready. {len(self.metadata)} samples.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # 1. Get IDs & Label
        audio_filename = f"{row['Audio_Song']}.mp3" 
        label_str = str(row['Genres']).split(',')[0].split('/')[0].strip()

        # 2. PROCESS AUDIO (With Normalization)
        audio_path = os.path.join(self.audio_dir, audio_filename)
        spectrogram = self.process_audio(audio_path)
        
        # 3. GET LYRICS (Fast look-up)
        lyric_vector = self.lyric_vectors[idx]

        return {
            'audio': torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0),
            'lyrics': torch.tensor(lyric_vector, dtype=torch.float32),
            'label': label_str,
            'id': row['Audio_Song']
        }

    def process_audio(self, path):
        # (Same normalization code as we agreed on - copy this block exactly)
        TARGET_SR = 22050
        TARGET_DURATION = 29.0
        
        try:
            y, sr = librosa.load(path, sr=TARGET_SR, duration=TARGET_DURATION)
            expected_samples = int(TARGET_SR * TARGET_DURATION)
            if len(y) < expected_samples:
                y = np.pad(y, (0, expected_samples - len(y)), mode='constant')
            else:
                y = y[:expected_samples]

            spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.target_shape[0])
            log_spec = librosa.power_to_db(spec, ref=np.max)
            
            # Crop/Pad
            if log_spec.shape[1] > self.target_shape[1]:
                log_spec = log_spec[:, :self.target_shape[1]]
            else:
                pad_width = self.target_shape[1] - log_spec.shape[1]
                log_spec = np.pad(log_spec, ((0,0), (0, pad_width)), mode='constant', constant_values=log_spec.min())

            # Normalize [0, 1]
            spec_min, spec_max = log_spec.min(), log_spec.max()
            if spec_max - spec_min > 1e-6:
                log_spec = (log_spec - spec_min) / (spec_max - spec_min)
            else:
                log_spec = np.zeros_like(log_spec)
                
            return log_spec
        except:
            return np.zeros(self.target_shape)

if __name__ == "__main__":
    # Test block
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    dataset = HybridMusicDataset(os.path.join(DATA_DIR, 'metadata_clean.csv'), os.path.join(DATA_DIR, 'audio'), os.path.join(DATA_DIR, 'lyrics'))
    print(f"Lyrics Vector Shape: {dataset[0]['lyrics'].shape}") # Should be (100,)