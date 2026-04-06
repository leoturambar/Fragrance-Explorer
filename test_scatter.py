from parser import load_ratings
from recommender import compute_scatter_data

df = load_ratings()
df_scatter = compute_scatter_data(df)

if df_scatter.empty:
    print("Nessun profumo matchato.")
else:
    df_sorted = df_scatter.sort_values('freshness', ascending=False)
    for _, row in df_sorted.iterrows():
        print(f"{row['brand']:25} — {row['name']:40} "
              f"fresh: {row['freshness']:.2f}  "
              f"depth: {row['depth']:.2f}  "
              f"intensity: {row['intensity']:.2f}  "
              f"rating: {row['rating']}")