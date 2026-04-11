from ratings import load_ratings
from recommender import compute_scatter_data
import numpy as np

df = load_ratings()
df_scatter = compute_scatter_data(df)

if df_scatter.empty:
    print("Nessun profumo matchato.")
else:
    # trasformazione logaritmica
    for col in ['freshness', 'depth', 'intensity']:
        df_scatter[col] = np.log1p(df_scatter[col])
        col_max = df_scatter[col].max()
        df_scatter[col] = df_scatter[col] / col_max if col_max > 0 else df_scatter[col]

    df_sorted = df_scatter.sort_values('freshness', ascending=False)
    for _, row in df_sorted.iterrows():
        print(f"{row['brand']:25} — {row['name']:40} "
              f"fresh: {row['freshness']:.2f}  "
              f"depth: {row['depth']:.2f}  "
              f"intensity: {row['intensity']:.2f}  "
              f"rating: {row['rating']}")