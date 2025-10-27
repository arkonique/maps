
# sim_countries_named.py
import numpy as np
import random
from collections import defaultdict
from markov_namegen import NameGenerator

# ====== DEMOGRAPHIC STYLE CORPORA ======
RACE_CORPORA = {
    "elves": ["Aeloria", "Sylthien", "Thalorien", "Faelir", "Naeriel"],
    "drow": ["Zynvyr", "Belmara", "Viconia", "Zaknafein", "Drizzt"],
    "humans": ["Alden", "Kaelin", "Mara", "Ulric", "Wulfric"],
    "dwarves": ["Brumgar", "Kharvek", "Morgrin", "Uldrak", "Ragniar"],
    "orcs": ["Brugnak", "Ghorlak", "Ragthuk", "Ugrak", "Zhurmok"],
}

# ====== Percent to 2dp (exact 100.00%) ======
def _percent_2dp_allocation(weights):
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    tot = w.sum()
    if tot <= 0:
        return np.zeros_like(w, dtype=np.float32)
    p_bp = (w / tot) * 10000.0
    floor_bp = np.floor(p_bp).astype(np.int64)
    remainder = int(10000 - floor_bp.sum())
    if remainder != 0:
        frac = p_bp - floor_bp
        order = np.argsort(-frac)[:remainder] if remainder > 0 else np.argsort(frac)[:(-remainder)]
        floor_bp[order] += 1 if remainder > 0 else -1
    return floor_bp.astype(np.float32) / 100.0

# ====== Example simulation stub ======
def simulate_country_demographics(n_countries, demo_spec, seed=42):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    race_names = list(demo_spec.keys())
    rarity = np.array([demo_spec[r][0] for r in race_names], dtype=np.float64)
    inv_rarity = 1.0 / rarity
    demographics = {}
    naming_races = []
    for i in range(n_countries):
        weights = rng.random(len(race_names)) * inv_rarity
        pct = _percent_2dp_allocation(weights)
        mask = pct >= 0.005
        demo = {race_names[j]: float(round(pct[j], 2)) for j in range(len(race_names)) if mask[j]}
        demographics[i] = demo

        # pick naming race weighted by percentage
        races, probs = zip(*[(r, pct[j]) for j, r in enumerate(race_names) if pct[j] >= 0.005])
        naming_race = rng.choice(races, p=np.array(probs) / sum(probs))
        naming_races.append(naming_race)
    return demographics, naming_races

# ====== Generate names ======
def generate_names_for_countries(naming_races, seed=7):
    gen = NameGenerator(order=3, seed=seed, corpora={r: RACE_CORPORA[r] for r in naming_races if r in RACE_CORPORA})
    country_names = []
    capital_names = []
    for race in naming_races:
        style = race if race in RACE_CORPORA else "human"
        country = gen.generate_exact_length(length=8, styles=style)
        capital = gen.generate_exact_length(length=8, styles=style)
        country_names.append(country)
        capital_names.append(capital)
    return country_names, capital_names

# ====== Run sim ======
if __name__ == "__main__":
    demo_spec = {
        "elves":  (1,  -0.85),
        "drow":   (4,   0.76),
        "humans": (1,   0.10),
        "dwarves":(2,  -0.20),
        "orcs":   (3,   0.55),
    }
    n = 10  # number of countries
    demographics, naming_races = simulate_country_demographics(n, demo_spec)
    country_names, capital_names = generate_names_for_countries(naming_races)

    # Final Output
    for i in range(n):
        print(f"Country {i+1}: {country_names[i]}")
        print(f"  Capital: {capital_names[i]}")
        print(f"  Naming race: {naming_races[i]}")
        for k, v in demographics[i].items():
            print(f"    {k}: {v:.2f}%")
        print()
