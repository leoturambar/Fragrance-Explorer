from matcher import load_dataset
import pandas as pd

df = load_dataset()

FRESH_NOTES = {
    'citrus', 'bergamot', 'lemon', 'grapefruit', 'orange', 'lime',
    'mandarin', 'neroli', 'aquatic', 'marine', 'ozonic', 'water',
    'fresh', 'green', 'grass', 'fig', 'basil', 'mint', 'herbal',
    'lavender', 'rosemary', 'eucalyptus', 'petitgrain'
}
DEEP_NOTES = {
    'oud', 'agarwood', 'amber', 'ambergris', 'resin', 'resinous',
    'balsamic', 'benzoin', 'labdanum', 'incense', 'oriental',
    'leather', 'tobacco', 'smoky', 'smoke', 'tar', 'dark',
    'vanilla', 'tonka', 'caramel', 'gourmand', 'myrrh', 'frankincense'
}
INTENSE_NOTES = {
    'oud', 'musk', 'ambergris', 'civet', 'castoreum', 'animalic',
    'amber', 'resin', 'tobacco', 'leather', 'vetiver', 'patchouli',
    'incense', 'myrrh', 'frankincense', 'benzoin', 'labdanum'
}
FRESH_ACCORDS = {
    'citrus', 'fresh', 'aquatic', 'green', 'aromatic', 'herbal',
    'fougere', 'fruity'
}
DEEP_ACCORDS = {
    'oriental', 'amber', 'oud', 'resinous', 'balsamic', 'leather',
    'tobacco', 'smoky', 'woody', 'warm spicy', 'gourmand', 'vanilla',
    'mossy', 'earthy', 'animalic', 'honey'
}

WEIGHTS = {'top': 0.3, 'middle': 0.6, 'base': 1.0}

def score_notes(note_str, note_set, weight):
    tokens = set(note_str.lower().replace(',', ' ').split())
    return len(tokens & note_set) * weight

test_frags = [
    'Philosykos Eau De Parfum',
    'Terre D Hermes Eau Intense Vetiver',
    'Hacivat',
    'Chergui',
    'Irish Leather',
    'Aventus',
    'Sauvage Elixir',
    'Un Jardin Sur Le Nil',
]

for name in test_frags:
    matches = df[df['Perfume'].str.lower() == name.lower()]
    if matches.empty:
        matches = df[df['Perfume'].str.contains(name, case=False, na=False)]
    if matches.empty:
        print(f"{name}: non trovato")
        continue
    row = matches.iloc[0]

    top    = str(row.get('Top', ''))
    middle = str(row.get('Middle', ''))
    base   = str(row.get('Base', ''))

    freshness_raw = (
        score_notes(top,    FRESH_NOTES, WEIGHTS['top'])    +
        score_notes(middle, FRESH_NOTES, WEIGHTS['middle']) +
        score_notes(base,   FRESH_NOTES, WEIGHTS['base'])
    )
    depth_raw = (
        score_notes(top,    DEEP_NOTES, WEIGHTS['top'])    +
        score_notes(middle, DEEP_NOTES, WEIGHTS['middle']) +
        score_notes(base,   DEEP_NOTES, WEIGHTS['base'])
    )
    intensity_raw = (
        score_notes(top,    INTENSE_NOTES, WEIGHTS['top'])    +
        score_notes(middle, INTENSE_NOTES, WEIGHTS['middle']) +
        score_notes(base,   INTENSE_NOTES, WEIGHTS['base'])
    )

    accord_str = ' '.join(filter(None, [
        str(row.get('mainaccord1', '')), str(row.get('mainaccord2', '')),
        str(row.get('mainaccord3', '')), str(row.get('mainaccord4', '')),
        str(row.get('mainaccord5', '')),
    ])).lower()
    accord_tokens = set(accord_str.replace(',', ' ').split())

    fresh_boost = 1.0 + 0.5 * len(accord_tokens & FRESH_ACCORDS)
    depth_boost = 1.0 + 0.5 * len(accord_tokens & DEEP_ACCORDS)
    fresh_base  = 1 if accord_tokens & FRESH_ACCORDS else 0
    depth_base  = 1 if accord_tokens & DEEP_ACCORDS else 0

    freshness = (freshness_raw + fresh_base) * fresh_boost
    depth     = (depth_raw + depth_base)     * depth_boost

    print(f"\n{row['Brand']} — {row['Perfume']}")
    print(f"  Accords: {accord_str}")
    print(f"  Fresh raw (pesato): {freshness_raw:.2f}  |  "
          f"Deep raw (pesato): {depth_raw:.2f}  |  "
          f"Intense raw (pesato): {intensity_raw:.2f}")
    print(f"  Fresh boost: x{fresh_boost} (base +{fresh_base})  |  "
          f"Deep boost: x{depth_boost} (base +{depth_base})")
    print(f"  → Freshness finale: {freshness:.2f}  |  Depth finale: {depth:.2f}")