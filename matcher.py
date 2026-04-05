import pandas as pd
import os
from rapidfuzz import process, fuzz
import re

FRAGRANCES_CLEAN = 'data/fragrances_clean.csv'
FRAGRANCES_RAW   = 'data/fragrances_raw.csv'


def _parse_notes_from_description(desc: str) -> tuple[str, str, str]:
    """
    Estrae top, middle, base notes dalla descrizione testuale del dataset raw.
    Es: "Top notes are Lemon, Rose; middle notes are Jasmine; base notes are Musk"
    """
    if not isinstance(desc, str):
        return '', '', ''

    top    = re.search(r'[Tt]op notes? (?:are|is) ([^;\.]+)', desc)
    middle = re.search(r'[Mm]iddle notes? (?:are|is) ([^;\.]+)', desc)
    base   = re.search(r'[Bb]ase notes? (?:are|is) ([^;\.]+)', desc)

    return (
        top.group(1).strip()    if top    else '',
        middle.group(1).strip() if middle else '',
        base.group(1).strip()   if base   else '',
    )


def _parse_accords(accord_str: str) -> list[str]:
    """
    Converte la stringa "['citrus', 'woody', 'musky']" in lista pulita.
    """
    if not isinstance(accord_str, str):
        return []
    return re.findall(r"'([^']+)'", accord_str)


def _extract_brand_name_from_url(url: str) -> tuple[str, str]:
    """
    Estrae brand e nome dall'URL Fragrantica.
    Es: https://www.fragrantica.com/perfume/Afnan/9am-70706.html
    → brand='Afnan', name='9am'
    """
    if not isinstance(url, str):
        return '', ''
    parts = url.rstrip('/').split('/')
    if len(parts) < 2:
        return '', ''
    brand = parts[-2].replace('-', ' ').title()
    name_raw = parts[-1]
    name_raw = re.sub(r'\.html?$', '', name_raw, flags=re.IGNORECASE)  # rimuove .html
    name_raw = re.sub(r'-\d+$', '', name_raw)  # rimuove id numerico finale
    name = name_raw.replace('-', ' ').title()
    return brand, name


def _load_clean() -> pd.DataFrame:
    """Carica e normalizza il dataset clean."""
    try:
        df = pd.read_csv(FRAGRANCES_CLEAN, encoding='utf-8', sep=';')
    except UnicodeDecodeError:
        df = pd.read_csv(FRAGRANCES_CLEAN, encoding='latin-1', sep=';')
    df.columns = df.columns.str.strip()
    df['Perfume'] = df['Perfume'].str.replace('-', ' ').str.title()
    df['Brand']   = df['Brand'].str.replace('-', ' ').str.title()
    df['Rating Value'] = df['Rating Value'].astype(str).str.replace(',', '.').astype(float)
    for col in ['Top', 'Middle', 'Base']:
        df[col] = df[col].str.title()
    df['source_dataset'] = 'clean'
    return df


def _load_raw() -> pd.DataFrame:
    """Carica il dataset raw e lo normalizza nel formato del clean."""
    try:
        df = pd.read_csv(FRAGRANCES_RAW, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(FRAGRANCES_RAW, encoding='latin-1')

    rows = []
    for _, row in df.iterrows():
        brand, name = _extract_brand_name_from_url(row.get('url', ''))
        if not brand or not name:
            continue

        top, middle, base = _parse_notes_from_description(row.get('Description', ''))
        accords = _parse_accords(row.get('Main Accords', ''))

        # padding accords a 5
        accords += [''] * (5 - len(accords))

        rows.append({
            'Brand':        brand,
            'Perfume':      name,
            'Country':      '',
            'Gender':       str(row.get('Gender', '')),
            'Rating Value': float(str(row.get('Rating Value', 0)).replace(',', '.')),
            'Rating Count': row.get('Rating Count', 0),
            'Year':         '',
            'Top':    top.title()    if top    else '',
            'Middle': middle.title() if middle else '',
            'Base':   base.title()   if base   else '',
            'Perfumer1':    '',
            'Perfumer2':    '',
            'mainaccord1':  accords[0],
            'mainaccord2':  accords[1],
            'mainaccord3':  accords[2],
            'mainaccord4':  accords[3],
            'mainaccord5':  accords[4],
            'source_dataset': 'raw',
        })

    return pd.DataFrame(rows)


def _normalize_key(s: str) -> str:
    """Normalizza stringa per deduplicazione — rimuove spazi, trattini, accenti."""
    import unicodedata
    s = s.lower().strip()
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()
    s = re.sub(r'[-\s]+', '', s)
    return s


def load_dataset() -> pd.DataFrame:
    """
    Carica e fonde i due dataset.
    Priorità al clean — i duplicati (stesso brand+nome) vengono deduplati
    mantenendo la versione clean.
    """
    df_clean = _load_clean()
    df_raw   = _load_raw()

    # normalizza brand e nome per deduplicazione
    for df in [df_clean, df_raw]:
        df['brand_clean'] = df['Brand'].str.lower().str.strip()
        df['name_clean']  = df['Perfume'].str.lower().str.strip()

    # chiave di deduplicazione
    df_clean['_key'] = df_clean['Brand'].apply(_normalize_key) + '||' + df_clean['Perfume'].apply(_normalize_key)
    df_raw['_key']   = df_raw['Brand'].apply(_normalize_key)   + '||' + df_raw['Perfume'].apply(_normalize_key)

    # tieni dal raw solo quelli non presenti nel clean
    clean_keys = set(df_clean['_key'])
    df_raw_only = df_raw[~df_raw['_key'].isin(clean_keys)]

    # fondi
    df = pd.concat([df_clean, df_raw_only], ignore_index=True)
    df = df.drop(columns=['_key'])
    df = df.reset_index(drop=True)

    return df


def get_candidates(brand: str, name: str, df_dataset: pd.DataFrame,
                   n: int = 5, threshold: int = 50) -> list[dict]:
    query_brand = brand.lower().strip()
    query_name = name.lower().strip()

    brand_matches = process.extract(
        query_brand,
        df_dataset['brand_clean'],
        scorer=fuzz.ratio,
        limit=100
    )
    brand_indices = [m[2] for m in brand_matches if m[1] >= threshold]

    if not brand_indices:
        return []

    df_brand = df_dataset.iloc[brand_indices].copy().reset_index(drop=False)

    if df_brand.empty:
        return []

    name_matches = process.extract(
        query_name,
        df_brand['name_clean'],
        scorer=fuzz.token_sort_ratio,
        limit=n
    )

    if not name_matches:
        return []

    candidates = []
    for match_text, score, idx in name_matches:
        if idx >= len(df_brand):
            continue
        row = df_brand.iloc[idx]
        acc = [str(row.get(f'mainaccord{i}', '')) for i in range(1, 6)]
        accords = ', '.join(a for a in acc if a and a != 'nan')
        candidates.append({
            'dataset_idx': int(row['index']),
            'brand':       row['Brand'],
            'name':        row['Perfume'],
            'score':       round(score),
            'top':         str(row.get('Top', '')) if pd.notna(row.get('Top')) else '',
            'middle':      str(row.get('Middle', '')) if pd.notna(row.get('Middle')) else '',
            'base':        str(row.get('Base', '')) if pd.notna(row.get('Base')) else '',
            'accords':     accords,
        })

    return sorted(candidates, key=lambda x: x['score'], reverse=True)


def enrich_ratings(df_ratings: pd.DataFrame,
                   confirmed_matches: dict) -> pd.DataFrame:
    """
    Aggiunge note olfattive al DataFrame personale usando i match confermati.

    confirmed_matches: dict {rating_index: dataset_idx | None}
        None = utente ha scelto 'nessuno di questi'
    """
    df_dataset = load_dataset()
    df = df_ratings.copy()

    tops, middles, bases, accords, matched, dataset_idxs = [], [], [], [], [], []

    for i, row in df.iterrows():
        dataset_idx = confirmed_matches.get(i)
        if dataset_idx is not None:
            ds_row = df_dataset.loc[dataset_idx]
            acc = [str(ds_row.get(f'mainaccord{j}', '')) for j in range(1, 6)]
            tops.append(str(ds_row.get('Top', '')) if pd.notna(ds_row.get('Top')) else '')
            middles.append(str(ds_row.get('Middle', '')) if pd.notna(ds_row.get('Middle')) else '')
            bases.append(str(ds_row.get('Base', '')) if pd.notna(ds_row.get('Base')) else '')
            accords.append(', '.join(a for a in acc if a and a != 'nan'))
            matched.append(True)
            dataset_idxs.append(dataset_idx)
        else:
            tops.append('')
            middles.append('')
            bases.append('')
            accords.append('')
            matched.append(False)
            dataset_idxs.append(None)

    df['top_notes']    = tops
    df['middle_notes'] = middles
    df['base_notes']   = bases
    df['main_accords'] = accords
    df['matched']      = matched
    df['dataset_idx']  = dataset_idxs

    return df


def save_confirmed_matches(confirmed: dict, path: str = 'data/confirmed_matches.csv'):
    """Salva i match confermati in CSV per non rifare il processo ogni volta."""
    rows = [{'rating_idx': k, 'dataset_idx': v} for k, v in confirmed.items()]
    pd.DataFrame(rows).to_csv(path, index=False)


def load_confirmed_matches(path: str = 'data/confirmed_matches.csv') -> dict:
    """Carica i match confermati salvati."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    result = {}
    for _, row in df.iterrows():
        k = int(row['rating_idx'])
        v = None if pd.isna(row['dataset_idx']) else int(row['dataset_idx'])
        result[k] = v
    return result