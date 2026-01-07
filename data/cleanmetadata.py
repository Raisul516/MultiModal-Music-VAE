import pandas as pd
import os

# Resolve project root safely
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_CSV = os.path.join(DATA_DIR, "metadata.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "metadata_clean.csv")

# Load metadata
df = pd.read_csv(INPUT_CSV)

# Keep only project-relevant columns
cols_to_keep = [
    "Audio_Song",
    "Lyric_Song",
    "Quadrant",
    "Artist",
    "Title",
    "Genres"
]
df = df[cols_to_keep]

# ---- CLEAN GENRES ----
def clean_genre(g):
    """
    Keeps only the first genre if multiple are comma-separated.
    DOES NOT split on '/' (e.g., Pop/Rock stays intact).
    """
    if pd.isna(g):
        return g

    # Convert to string and split ONLY on comma
    return g.split(",")[0].strip()

df["Genres"] = df["Genres"].apply(clean_genre)

# Save cleaned metadata
df.to_csv(OUTPUT_CSV, index=False)

print("✅ Cleaned metadata saved to:", OUTPUT_CSV)
print("Sample genres:", df["Genres"].unique()[:10])
