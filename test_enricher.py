from enricher import scrape_fragrantica

url = "https://www.fragrantica.com/perfume/Hermes/Terre-d-Hermes-Intense-102772.html"
result = scrape_fragrantica(url)
if result:
    print(f"Top:     {result['top']}")
    print(f"Middle:  {result['middle']}")
    print(f"Base:    {result['base']}")
    print(f"Accords: {result['accords']}")
else:
    print("Non trovato.")