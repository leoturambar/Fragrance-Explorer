from bs4 import BeautifulSoup
import pandas as pd
import os

MY_LIST_FILE = 'data/my_list.html'
MY_RATINGS_FILE = 'data/my_ratings.csv'

RATINGS_COLS = [
    'brand',
    'name',
    'form',              # EdP, EdT, Extrait, ecc.
    'status',            # Bottle | Decant | Sample | Tested | '' (nessuno)
    'watchlist',         # True/False — segnalibro indipendente dallo status
    'rating',            # 1.0 → 10.0
    'bottle_candidate',  # True/False — N/A se status == Bottle
    'comment',           # commento personale
    'source',            # html_import, manual
]

_OWNERSHIP_TO_STATUS = {
    'Full Bottle': 'Bottle',
    'Decant':      'Decant',
    'Sample':      'Sample',
    'Wishlist':    '',
}


def _migrate_ownership(df: pd.DataFrame) -> pd.DataFrame:
    """Migrates old 'ownership' column to 'status' + 'watchlist'."""
    if 'ownership' not in df.columns:
        return df
    if 'status' not in df.columns:
        df['status'] = df['ownership'].map(_OWNERSHIP_TO_STATUS).fillna('')
    if 'watchlist' not in df.columns:
        df['watchlist'] = df['ownership'].str.strip().str.lower() == 'wishlist'
    df = df.drop(columns=['ownership'])
    return df


def parse_html() -> pd.DataFrame:
    """
    Legge my_list.html ed estrae tutte le fragranze con rating e commenti.
    Restituisce un DataFrame con le colonne RATINGS_COLS.
    """
    if not os.path.exists(MY_LIST_FILE):
        return pd.DataFrame(columns=RATINGS_COLS)

    with open(MY_LIST_FILE, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    rows = []
    for section in soup.find_all('section', class_='section'):
        # estrai il rating dalla sezione
        score_el = section.find(class_='score')
        if not score_el:
            continue
        rating_str = score_el.get_text(strip=True).replace('/10', '')
        try:
            rating = float(rating_str)
        except ValueError:
            continue

        for item in section.find_all('div', class_='item'):
            # brand
            brand_el = item.find('span', class_='brand')
            brand = brand_el.get_text(strip=True) if brand_el else ''

            # name
            name_el = item.find('span', class_='name')
            name = name_el.get_text(strip=True) if name_el else ''

            # form (EdP, EdT, ecc.) — primo span.form
            form_els = item.find_all('span', class_='form')
            form = form_els[0].get_text(strip=True) if form_els else ''

            # ownership (Decant, Full Bottle) — secondo span.form se esiste
            ownership = form_els[1].get_text(strip=True) if len(form_els) > 1 else ''
            # pulizia parentesi
            ownership = ownership.strip('()')

            # bottle candidate
            bottle_candidate = bool(item.find('span', class_='bc'))

            # commento
            desc_el = item.find('div', class_='desc')
            comment = desc_el.get_text(strip=True) if desc_el else ''

            rows.append({
                'brand':            brand,
                'name':             name,
                'form':             form,
                'ownership':        ownership,
                'rating':           rating,
                'bottle_candidate': bottle_candidate,
                'comment':          comment,
                'source':           'html_import',
            })

    return pd.DataFrame(rows, columns=RATINGS_COLS)


def load_ratings() -> pd.DataFrame:
    """
    Carica il CSV dei ratings. Se non esiste, lo crea parsando l'HTML.
    Migra automaticamente il vecchio campo 'ownership' al nuovo schema.
    """
    if not os.path.exists(MY_RATINGS_FILE):
        df = parse_html()
        if not df.empty:
            df = _migrate_ownership(df)
            df.to_csv(MY_RATINGS_FILE, index=False)
        return df

    df = pd.read_csv(MY_RATINGS_FILE)
    if 'ownership' in df.columns:
        df = _migrate_ownership(df)
        df.to_csv(MY_RATINGS_FILE, index=False)
    if 'status' not in df.columns:
        df['status'] = ''
    if 'watchlist' not in df.columns:
        df['watchlist'] = False
    return df


def save_ratings(df: pd.DataFrame):
    """Salva il DataFrame dei ratings nel CSV."""
    df.to_csv(MY_RATINGS_FILE, index=False)


def save_enriched_notes(idx: int, top: str, middle: str,
                         base: str, accords: str):
    """
    Aggiorna le note olfattive di una entry esistente nel CSV.
    """
    df = load_ratings()
    df.at[idx, 'top_notes']    = top
    df.at[idx, 'middle_notes'] = middle
    df.at[idx, 'base_notes']   = base
    df.at[idx, 'main_accords'] = accords
    df.at[idx, 'matched']      = True
    df.at[idx, 'notes_source'] = 'web_enriched'
    save_ratings(df)


def add_rating(brand: str, name: str, form: str, status: str,
               rating: float, comment: str = '',
               bottle_candidate: bool = False, watchlist: bool = False) -> pd.DataFrame:
    """Aggiunge una nuova fragranza al database personale."""
    df = load_ratings()
    new_row = pd.DataFrame([{
        'brand':            brand,
        'name':             name,
        'form':             form,
        'status':           status,
        'watchlist':        watchlist,
        'rating':           rating,
        'bottle_candidate': bottle_candidate,
        'comment':          comment,
        'source':           'manual',
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    save_ratings(df)
    return df


def delete_rating(idx: int):
    """Rimuove una entry dal CSV per indice e resetta gli indici."""
    df = load_ratings()
    df = df.drop(index=idx).reset_index(drop=True)
    save_ratings(df)


def check_duplicate(brand: str, name: str) -> bool:
    """Restituisce True se esiste già una entry con stesso brand e nome."""
    df = load_ratings()
    if df.empty:
        return False
    return any(
        (df['brand'].str.lower().str.strip() == brand.lower().strip()) &
        (df['name'].str.lower().str.strip() == name.lower().strip())
    )