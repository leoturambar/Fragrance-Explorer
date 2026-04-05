from matcher import load_dataset
from recommender import build_brand_scores, compute_quality_score
import pandas as pd
import numpy as np

df = load_dataset()
brand_scores = build_brand_scores(df)

# brand scores
print("=== BRAND SCORES ===")
for brand in ['nishane', 'diptyque', 'rasasi', 'cuba paris',
              'lattafa perfumes', 'halston', 'mugler', 'miller harris']:
    score = brand_scores.get(brand, None)
    print(f"{brand}: {score:.3f}" if score else f"{brand}: non trovato")

# quality scores per profumi specifici
print("\n=== QUALITY SCORES PROFUMI ===")
test_frags = [
    # niche/lusso
    'Philosykos', 'Hacivat', 'Irish Leather', 'Aventus',
    # mass market buoni
    'Sauvage', 'Layton',
    # low quality
    'Cuba City', 'Fattan', 'Shaheen Silver',
    # lattafa
    'Oudy Night', 'Asad',
]
for name in test_frags:
    matches = df[df['Perfume'].str.contains(name, case=False, na=False)]
    if matches.empty:
        print(f"{name}: non trovato")
        continue
    row = matches.iloc[0]
    score = compute_quality_score(row, brand_scores)
    print(f"{row['Brand']} — {row['Perfume']}: {score:.3f} "
          f"(rating: {row['Rating Value']}, reviews: {row['Rating Count']})")

test_frags2 = ['Supremacy', 'Hacivaz', 'Chyprissime', 'Halston Couture', 'Haciventus', 'Fattan', 'Gris Dior']
for name in test_frags2:
    matches = df[df['Perfume'].str.contains(name, case=False, na=False)]
    if matches.empty:
        print(f"{name}: non trovato")
        continue
    row = matches.iloc[0]
    score = compute_quality_score(row, brand_scores)
    print(f"{row['Brand']} — {row['Perfume']}: {score:.3f} "
          f"(rating: {row['Rating Value']}, reviews: {row['Rating Count']})")