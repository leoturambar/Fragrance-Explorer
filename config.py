# Famiglie olfattive — usate per radar chart e analisi profilo
ACCORDS = [
    # Freschi
    'citrus', 'fresh', 'aquatic', 'green', 'herbal', 'aromatic',
    # Floreali
    'floral', 'powdery', 'soft floral',
    # Legnosi
    'woody', 'earthy', 'mossy', 'smoky',
    # Orientali / caldi
    'oriental', 'amber', 'balsamic', 'resinous', 'vanilla', 'gourmand', 'sweet',
    # Speziati
    'spicy', 'warm spicy',
    # Scuri / animali
    'leather', 'tobacco', 'animalic', 'musky',
    # Frutti
    'fruity',
    # Speciali
    'oud', 'incense', 'rose', 'iris', 'vetiver',
]

# Stili esplorazione con note/parole chiave associate
EXPLORATION_STYLES = {
    # Freschi
    'Citrus':           'bergamot lemon grapefruit orange mandarin citrus neroli',
    'Aquatic':          'aquatic marine sea water fresh ozonic',
    'Green':            'green grass fig leaf basil herb tomato',
    'Aromatic':         'lavender rosemary sage thyme herbal aromatic fougere',
    'Fougère':          'lavender coumarin oakmoss woody fougere aromatic clean',

    # Floreali
    'Floral':           'rose jasmine iris violet floral ylang neroli peony',
    'Powdery':          'iris powder musk soft clean talc violet orris',
    'White Floral':     'jasmine tuberose gardenia white floral indolic',

    # Legnosi
    'Woody':            'cedar sandalwood woody vetiver guaiac patchouli',
    'Vetiver':          'vetiver earthy smoky woody dry grass roots',
    'Mossy/Chypre':     'oakmoss labdanum bergamot chypre mossy earthy cistus',
    'Smoky':            'smoky birch tar bonfire ash incense dark',

    # Orientali / caldi
    'Amber/Oriental':   'amber benzoin labdanum resinous warm balsamic tonka',
    'Vanilla/Gourmand': 'vanilla tonka caramel chocolate gourmand sweet praline',
    'Balsamic':         'balsamic benzoin styrax peru balsam resin warm',

    # Speziati
    'Spicy':            'pepper cardamom cinnamon clove nutmeg ginger spicy',
    'Warm Spicy':       'saffron cumin coriander warm spicy oriental incense',

    # Scuri
    'Leather':          'leather suede birch tar castoreum smoky dark',
    'Tobacco':          'tobacco hay honey dried fruit coumarin warm',
    'Animalic':         'musk civet castoreum animalic skin warm sensual',

    # Speciali
    'Oud':              'oud agarwood resinous smoky incense woody barnyard',
    'Incense':          'incense frankincense myrrh smoke resin church sacred',
    'Rose':             'rose damascene turkish bulgarian rosy floral honeyed',
    'Iris':             'iris orris violet carrot powdery clean cool',
    'Fruity':           'peach pear apple berry plum fruity juicy',
}

# Generi per filtro raccomandazioni
GENDER_OPTIONS = {
    'All':    'all',
    'Male':   'male',
    'Female': 'female',
    'Unisex': 'unisex',
}

# Rating scale
MIN_RATING = 1.0
MAX_RATING = 10.0
RATING_STEP = 0.5