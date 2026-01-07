import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def save_table_image():
    # 1. Define Data
    data = {
        "Method": [
            "Direct Spectral (K-Means)", 
            "Baseline: PCA + K-Means", 
            "Baseline: Autoencoder", 
            "Hybrid VAE + K-Means", 
            "Hybrid VAE + Agglomerative", 
            "Hybrid VAE + DBSCAN"
        ],
        "NMI":    [0.1052, 0.1110, 0.1001, 0.1394, 0.1224, 0.0576],
        "ARI":    [0.0496, 0.0602, 0.0639, 0.0917, 0.0608, -0.0347],
        "Purity": [0.3260, 0.3290, 0.3250, 0.3480, 0.3490, 0.3215],
        "Sil":    [0.1068, 0.2264, 0.2936, 0.0407, 0.0143, -0.1214],
        "CH":     [343.04, 812.84, 1093.65, 81.78, 62.49, 23.39],
        "DB":     [2.4066, 1.5214, 1.1770, 3.5644, 4.5211, 3.0355]
    }

    df = pd.DataFrame(data)

    # 2. Setup Plot (Wider figure to fit text)
    fig, ax = plt.subplots(figsize=(16, 4)) 
    ax.axis('tight')
    ax.axis('off')

    # 3. Create Table
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')

    # 4. Styling
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8) # Scale height mostly

    # === FIX: AUTO-ADJUST COLUMN WIDTHS ===
    # This specific line fixes the "text outside box" issue
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # Custom Coloring
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        elif row > 0:
            if row % 2 == 0:
                cell.set_facecolor('#f2f2f2')
            
            # Highlight Winner Row
            if df.iloc[row-1]["Method"] == "Hybrid VAE + K-Means":
                cell.set_edgecolor('black')
                cell.set_linewidth(2)
                cell.set_facecolor('#d1e7dd')

    # 5. Save
    plt.title("Table 1: Quantitative Evaluation Metrics", fontsize=14, weight='bold', pad=20)
    # bbox_inches='tight' cuts off extra whitespace
    plt.savefig("results_table.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    print("Saved fixed 'results_table.png'")

if __name__ == "__main__":
    save_table_image()