import pandas as pd

# 1. Load your current clean file
file_path = 'metadata_clean.csv'
df = pd.read_csv(file_path)

# 2. Define the function to remove everything after '/'
def clean_slashes(genre):
    # Ensure it's a string, then split by '/' and take the first part
    return str(genre).split('/')[0].strip()

# 3. Apply it to your Genre column
# Note: If your column is named 'Genres', change 'Primary_Genre' to 'Genres' below
target_column = 'Primary_Genre' if 'Primary_Genre' in df.columns else 'Genres'

print(f"Cleaning column: {target_column}...")
df[target_column] = df[target_column].apply(clean_slashes)

# 4. Check the results (Verify it worked)
print("\nNew Unique Genres:")
print(df[target_column].value_counts().head(10))

# 5. Save the final version
df.to_csv('metadata_clean.csv', index=False)
print("\nSaved updated 'metadata_clean.csv'.")