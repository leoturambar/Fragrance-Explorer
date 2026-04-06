import requests
import time
import re
from bs4 import BeautifulSoup
from ddgs import DDGS


HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
}


def search_fragrantica_url(brand: str, name: str) -> str | None:
    query = f"{brand} {name} site:fragrantica.com"
    print(f"  Query DDG: {query}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        print(f"  Risultati trovati: {len(results)}")
        for r in results:
            print(f"    → {r.get('href', 'no href')}")
            url = r.get('href', '')
            if 'fragrantica.com/perfume/' in url:
                return url
    except Exception as e:
        print(f"  DDG search error: {type(e).__name__}: {e}")
    return None


def scrape_fragrantica(url: str) -> dict | None:
    """
    Scarica la pagina Fragrantica ed estrae note e accords.
    Le note sono in <span class="pyramid-note-label">.
    Gli accords sono in <div> con classe rounded-br-lg, testo in <span class="truncate">.
    Se non c'è piramide separata top/middle/base, mette tutto in base_notes.
    """
    try:
        time.sleep(1)  # cortesia verso il server
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"  Fetch error: {e}")
        return None

    soup = BeautifulSoup(resp.text, 'html.parser')

    # ── Note olfattive ────────────────────────────────────────────────────
    # ogni nota è in <span class="pyramid-note-label">
    note_spans = soup.find_all('span', class_='pyramid-note-label')
    all_notes = [s.get_text(strip=True) for s in note_spans if s.get_text(strip=True)]

    # controlla se ci sono livelli separati (top/middle/base)
    # i container di livello hanno classe "pyramid-level-container"
    level_containers = soup.find_all('div', class_='pyramid-level-container')

    if len(level_containers) >= 3:
        # piramide completa — estrai per livello
        top    = ', '.join(
            s.get_text(strip=True)
            for s in level_containers[0].find_all('span', class_='pyramid-note-label')
        )
        middle = ', '.join(
            s.get_text(strip=True)
            for s in level_containers[1].find_all('span', class_='pyramid-note-label')
        )
        base   = ', '.join(
            s.get_text(strip=True)
            for s in level_containers[2].find_all('span', class_='pyramid-note-label')
        )
    else:
        # nessuna piramide separata — tutto in base_notes
        top    = ''
        middle = ', '.join(all_notes)
        base   = ''

    # ── Accords ───────────────────────────────────────────────────────────
    # il container accords ha un h6 con testo "main accords"
    # ogni accord è in <span class="truncate"> dentro un div rounded-br-lg
    accords = []
    accord_section = None

    # trova il contenitore cercando l'h6 "main accords"
    for h6 in soup.find_all('h6'):
        if 'main accords' in h6.get_text(strip=True).lower():
            accord_section = h6.find_parent('div')
            break

    if accord_section:
        for div in accord_section.find_all('div', class_=lambda c: c and 'rounded-br-lg' in c):
            label = div.find('span', class_='truncate')
            if label:
                text = label.get_text(strip=True).lower()
                if text and text not in accords:
                    accords.append(text)

    notes = {
        'top':     top,
        'middle':  middle,
        'base':    base,
        'accords': ', '.join(accords[:5]),  # max 5 come nel dataset Kaggle
    }

    # restituisce None solo se non ha trovato assolutamente nulla
    return notes if any(notes[k] for k in ['top', 'middle', 'base', 'accords']) else None


def enrich_from_web(brand: str, name: str) -> dict | None:
    """
    Pipeline completa: cerca URL → scarica → estrae note.
    Restituisce dict {top, middle, base, accords, url} o None.
    """
    url = search_fragrantica_url(brand, name)
    if not url:
        return None

    notes = scrape_fragrantica(url)
    if not notes:
        return None

    notes['url'] = url
    return notes