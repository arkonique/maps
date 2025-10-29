# pop_sim_all_fast.py
import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    convolve,
    maximum_filter,
    label,  # C-optimized component labeling
)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm
import re
import random
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Sequence, Tuple, Union


RACE_SPECS = {
        # Core & common
        "humans":        (1,  0.10),
        "elves":         (1, -0.85),
        "dwarves":       (2, -0.25),
        "halflings":     (2, -0.55),
        "gnomes":        (2, -0.45),
        "half-elves":    (2, -0.40),
        "half-orcs":     (2,  0.55),
        "dragonborn":    (3,  0.60),
        "tieflings":     (3,  0.35),

        # Elf subraces
        "high elves":    (2, -0.80),
        "wood elves":    (2, -0.75),
        "drow":          (4,  0.76),
        "eladrin":       (3, -0.65),
        "sea elves":     (3, -0.70),
        "shadar-kai":    (4,  0.50),

        # Dwarves
        "hill dwarves":     (2, -0.20),
        "mountain dwarves": (2,  0.10),
        "duergar":          (4,  0.65),

        # Gnomes
        "forest gnomes": (3, -0.50),
        "rock gnomes":   (2, -0.35),
        "deep gnomes":   (4,  0.20),

        # Halflings
        "lightfoot halflings": (2, -0.60),
        "stout halflings":     (2, -0.45),

        # Planetouched / exotic
        "aasimar":       (3, -0.70),
        "genasi":        (3,  0.00),
        "firbolg":       (3, -0.80),
        "goliaths":      (3,  0.40),
        "aarakocra":     (3, -0.40),
        "kenku":         (3,  0.20),
        "tabaxi":        (3, -0.20),
        "tortles":       (4, -0.10),
        "yuan-ti purebloods": (4,  0.85),
        "lizardfolk":    (3,  0.50),
        "kobolds":       (3,  0.55),
        "goblins":       (2,  0.70),
        "hobgoblins":    (3,  0.80),
        "bugbears":      (3,  0.90),
        "orc":           (2,  0.75),
        "gnolls":        (4,  0.95),

        # Fey / nature aligned
        "satyrs":        (3, -0.30),
        "centaurs":      (3, -0.10),
        "fairies":       (4, -0.85),
        "harengon":      (4, -0.40),
        "dryads":        (4, -0.90),

        # Construct / undead / planar
        "warforged":     (3,  0.45),
        "reborn":        (4, -0.10),
        "dhampir":       (4,  0.65),
        "hexblood":      (4, -0.25),
        "shardmind":     (5,  0.00),
        "changeling":    (3, -0.30),
        "kalashtar":     (3, -0.75),
        "githyanki":     (4,  0.80),
        "githzerai":     (4, -0.70),
        "autognomes":    (4, -0.15),
        "plasmoids":     (4,  0.10),
        "thri-kreen":    (4,  0.50),
        "locathah":      (4, -0.20),
        "merfolk":       (3, -0.40),
        "tritons":       (3, -0.30),

        # Celestial / infernal extremes
        "angelic celestials": (5, -1.00),
        "devils":             (5,  1.00),
        "demons":             (5,  0.95),
        "archfey":            (5, -0.90),
}

# =========================================================
# Neighbor utilities
# =========================================================
NEIGH = np.array([
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1)
], dtype=np.int8)

KERNEL_8 = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.float32)

def roll2(a, dy, dx):
    # wraparound roll
    return np.roll(np.roll(a, dy, axis=0), dx, axis=1)

def one_step_route(amount, attractiveness, water_mask):
    """
    Move 'amount' one neighbor step toward argmax-attractiveness.
    Vectorized: stack 8 rolled attractiveness maps → argmax → scatter via rolls.
    """
    Astack = np.stack([roll2(attractiveness, -dy, -dx) for dy, dx in NEIGH], axis=0)  # [8,H,W]
    water_penalty = -1e20
    for k, (dy, dx) in enumerate(NEIGH):
        dest_is_water = roll2(water_mask, dy, dx)
        Astack[k][dest_is_water] = water_penalty
    best_dir = np.argmax(Astack, axis=0)
    inflow = np.zeros_like(amount, dtype=np.float32)
    for k, (dy, dx) in enumerate(NEIGH):
        take = amount * (best_dir == k)
        if np.any(take):
            inflow += roll2(take, dy, dx)
    return inflow

# =========================================================
# Biomes & weights
# IDs: 0=water, 1=sand, 2=grass, 3=forest, 4=rock, 5=snow
# =========================================================
def biome_id_from_scaled(land_scaled, water_mask, thresholds=(0.10, 0.40, 0.65, 0.85)):
    biome = np.zeros_like(land_scaled, dtype=np.uint8)
    biome[water_mask] = 0
    t1, t2, t3, t4 = thresholds
    s  = (~water_mask) & (land_scaled <  t1)
    g  = (~water_mask) & (land_scaled >= t1) & (land_scaled < t2)
    f  = (~water_mask) & (land_scaled >= t2) & (land_scaled < t3)
    r  = (~water_mask) & (land_scaled >= t3) & (land_scaled < t4)
    sn = (~water_mask) & (land_scaled >= t4)
    biome[s]  = 1; biome[g] = 2; biome[f] = 3; biome[r] = 4; biome[sn] = 5
    return biome

def biome_migration_weight(biome):
    # Migration attractiveness (destination preference).
    return np.take(np.array([0.00, 0.90, 1.00, 0.85, 0.45, 0.10], np.float32), biome)

def biome_capacity_factor(biome):
    # Carrying capacity scaling
    return np.take(np.array([0.00, 0.60, 1.00, 0.80, 0.40, 0.15], np.float32), biome)

def biome_death_multiplier(biome):
    # Mortality multiplier
    return np.take(np.array([1.00, 1.02, 1.00, 1.05, 1.20, 1.40], np.float32), biome)

# =========================================================
# Conway-like Life (vectorized, probabilistic)
# =========================================================
def parse_life_rule(rule_str="B3/S23"):
    bpart, spart = rule_str.upper().split("/")
    B = set(int(ch) for ch in bpart[1:] if ch.isdigit())
    S = set(int(ch) for ch in spart[1:] if ch.isdigit())
    return B, S

def conway_prob_step(N, K_eff, rule_str="B3/S23",
                     pop_threshold=1200.0, density_norm=6000.0,
                     p_birth=0.9, p_survive=0.95, noise_sigma=0.05, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    B, S = parse_life_rule(rule_str)
    denom = pop_threshold + 0.25 * (K_eff + 1e-9)
    score = N / (denom + 1e-9)
    soft_alive = 1.0 / (1.0 + np.exp(-score))
    alive = soft_alive > (pop_threshold / (pop_threshold + density_norm))
    neigh = convolve(alive.astype(np.float32), KERNEL_8, mode="reflect")
    if noise_sigma > 0:
        pb = np.clip(p_birth   * np.exp(rng.normal(0.0, noise_sigma, size=N.shape)), 0.0, 1.0)
        ps = np.clip(p_survive * np.exp(rng.normal(0.0, noise_sigma, size=N.shape)), 0.0, 1.0)
    else:
        pb, ps = p_birth, p_survive
    dead = ~alive
    birth_mask   = dead  & np.isin(neigh, list(B))
    survive_mask = alive & np.isin(neigh, list(S))
    births_real  = birth_mask   & (rng.random(N.shape) < pb)
    survive_real = survive_mask & (rng.random(N.shape) < ps)
    alive_next = survive_real | births_real
    if len(B | S) > 0:
        ok = np.array(sorted(list(B | S)), dtype=np.float32)
        prox = np.min(np.stack([np.abs(neigh - k) for k in ok], axis=0), axis=0)
        targetish = 1.0 - (prox / 8.0)
    else:
        targetish = np.zeros_like(N, dtype=np.float32)
    L = np.clip(0.6 * soft_alive.astype(np.float32) + 0.4 * targetish, 0.0, 1.0)
    return alive_next, L

# =========================================================
# Heightmap utilities
# =========================================================
def heightmap_xyz_to_grid(heightmap_xyz, H=None, W=None, dtype=np.float32):
    """
    heightmap_xyz: array-like (N,3) with integer x,y and float z. If H,W not provided, infer from max x/y + 1.
    Returns h_raw[ y, x ] = z
    """
    xyz = np.asarray(heightmap_xyz)
    if H is None: H = int(xyz[:,1].max()) + 1
    if W is None: W = int(xyz[:,0].max()) + 1
    h_raw = np.zeros((H, W), dtype=dtype)
    x = xyz[:,0].astype(int); y = xyz[:,1].astype(int); z = xyz[:,2].astype(dtype)
    h_raw[y, x] = z
    return h_raw

# =========================================================
# Markov name generator (backoff, exact length)
# =========================================================
SANITIZE_RE = re.compile(r"[^a-zA-Z'\-]")

def _sanitize(name: str, allow_punct: bool) -> str:
    s = SANITIZE_RE.sub("", name.strip())
    if not allow_punct:
        s = re.sub(r"[-']", "", s)
    s = re.sub(r"([-'])\1+", r"\1", s)
    return s

class BackoffMarkov:
    """
    Multi-order backoff Markov model over characters.
    Trains graphs for k = 0..order-1 (unigram..(order-1)-gram).
    '$' end token is learned but *never emitted* during length-constrained generation.
    """
    def __init__(self, order: int = 3):
        if order < 2:
            raise ValueError("order must be >= 2")
        self.order = order
        self.graphs: List[Dict[str, Counter]] = [defaultdict(Counter) for _ in range(order)]
        self.alphabet: Counter = Counter()  # letter frequencies (no '$')

    def train(self, words: Sequence[str], allow_punct: bool = True) -> None:
        o = self.order
        for w in words:
            w = _sanitize(w, allow_punct).lower()
            if not w:
                continue
            s = ("^" * (o - 1)) + w + "$"
            self.alphabet.update([ch for ch in w if ch != "$"])
            # fill graphs for all k
            for i in range(len(s) - 1):
                for k in range(0, o):
                    if i - (o - 1 - k) < 0:
                        continue
                    ctx = s[i - (o - 1 - k): i] if k > 0 else ""
                    self.graphs[k][ctx].update(s[i])
            # transition to '$'
            last_context_full = s[-(o - 1 + 1):-1]
            for k in range(0, o):
                ctx = last_context_full[-k:] if k > 0 else ""
                self.graphs[k][ctx].update("$")
        if "^" in self.alphabet:
            del self.alphabet["^"]

    def _sample_from_counter(self, ctr: Counter, rng: random.Random, forbidden: Optional[set] = None) -> Optional[str]:
        if not ctr:
            return None
        items, weights = [], []
        forb = forbidden or set()
        for ch, w in ctr.items():
            if ch in forb:
                continue
            items.append(ch)
            weights.append(w)
        if not items:
            return None
        return rng.choices(items, weights=weights, k=1)[0]

    def next_char(self, context: str, rng: random.Random, allow_end: bool = False) -> Optional[str]:
        forbidden = set() if allow_end else {"$"}
        o = self.order
        kmax = min(o - 1, len(context))
        for k in range(kmax, -1, -1):
            ctx = context[-k:] if k > 0 else ""
            ctr = self.graphs[k].get(ctx)
            if ctr:
                ch = self._sample_from_counter(ctr, rng, forbidden=forbidden)
                if ch is not None:
                    return ch
        return self._sample_from_counter(self.alphabet, rng, forbidden=forbidden)

class NameGenerator:
    def __init__(self, order: int = 3, seed: Optional[int] = None, corpora: Optional[Dict[str, List[str]]] = None):
        self.order = order
        self.rng = random.Random(seed)
        self.corpora = dict({} if corpora is None else corpora)

    def _collect_examples(self, styles: Union[str, Sequence[str]], weights: Optional[Sequence[float]] = None) -> List[str]:
        if isinstance(styles, str):
            styles = [styles]
        styles = [s.lower() for s in styles]
        if weights is None:
            weights = [1.0] * len(styles)
        if len(weights) != len(styles):
            raise ValueError("weights must match styles length")
        data: List[str] = []
        for s, w in zip(styles, weights):
            if s not in self.corpora:
                raise KeyError(f"Unknown style '{s}'. Known: {list(self.corpora)[:10]}...")
            reps = max(1, int(round(w * 10)))
            data.extend(self.corpora[s] * reps)
        return data

    def add_style(self, name: str, examples: Sequence[str], overwrite: bool = False) -> None:
        name = name.lower()
        if name in self.corpora and not overwrite:
            self.corpora[name] = list(set(self.corpora[name]) | set(examples))
        else:
            self.corpora[name] = list(examples)

    def generate_exact_length(
        self,
        length: int,
        styles: Union[str, Sequence[str]],
        weights: Optional[Sequence[float]] = None,
        starts_with: Optional[str] = None,
        allow_punct: bool = True,
        enforce_shape: bool = True,
        max_resamples_per_pos: int = 6,
    ) -> str:
        if length <= 0:
            raise ValueError("length must be positive")
        data = self._collect_examples(styles, weights)
        model = BackoffMarkov(order=self.order)
        model.train(data, allow_punct=allow_punct)

        out: List[str] = []
        prefix = _sanitize(starts_with or "", allow_punct=allow_punct).lower()
        for ch in prefix:
            out.append(ch)

        vowels = set("aeiouy")
        def bad_shape(s: List[str]) -> bool:
            if not enforce_shape:
                return False
            st = "".join(s)
            if re.search(r"(.)\1\1", st):
                return True
            runs = re.findall(r"[aeiouy]+|[^aeiouy]+", st)
            if any(len(run) >= 6 for run in runs):
                return True
            if st and st[0] in "-'":
                return True
            if st and st[-1] in "-'":
                return True
            return False

        while len(out) < length:
            accepted = None
            for _ in range(max_resamples_per_pos):
                ctx = "".join(out)[-(self.order - 1):]
                ch = model.next_char(ctx, self.rng, allow_end=False)
                if ch is None or ch == "$":
                    continue
                candidate = out + [ch]
                if not bad_shape(candidate):
                    accepted = ch
                    break
            if accepted is None:
                last = out[-1] if out else ""
                want_vowel = (last and last not in vowels)
                pool = [c for c in (model.alphabet.keys()) if (c in vowels) == want_vowel and c not in {"$", "^"}]
                if not pool:
                    pool = [c for c in model.alphabet.keys() if c not in {"$", "^"}]
                accepted = self.rng.choice(list(pool))
            out.append(accepted)

        name = "".join(out)
        return name[:1].upper() + name[1:]

    def generate_batch(
        self,
        k: int,
        length: int,
        styles: Union[str, Sequence[str]],
        weights: Optional[Sequence[float]] = None,
        starts_with: Optional[str] = None,
        allow_punct: bool = True,
        unique: bool = True,
    ) -> List[str]:
        results: List[str] = []
        seen = set()
        attempts = 0
        budget = max(k * 5, k + 20)
        while len(results) < k and attempts < budget:
            attempts += 1
            nm = self.generate_exact_length(length=length, styles=styles, weights=weights,
                                            starts_with=starts_with, allow_punct=allow_punct)
            if unique and nm in seen:
                continue
            results.append(nm)
            seen.add(nm)
        while len(results) < k:
            results.append(self.generate_exact_length(length, styles=styles, weights=weights,
                                                     starts_with=starts_with, allow_punct=allow_punct))
        return results

# =========================================================
# Country/City corpora by race-family (programmatic coverage for all races)
# =========================================================
def race_family(race: str) -> str:
    r = race.lower()
    if any(k in r for k in ["elf", "eladrin", "shadar-kai"]):
        return "elven"
    if any(k in r for k in ["dwarf", "duergar"]):
        return "dwarven"
    if any(k in r for k in ["orc", "goblin", "hobgob", "bugbear", "gnoll", "kobold"]):
        return "orcish"
    if any(k in r for k in ["human"]):
        return "human"
    if any(k in r for k in ["tiefl", "devil", "demon", "fiend", "yuan-ti", "serpent"]):
        return "fiendish"
    if any(k in r for k in ["aasimar", "celest"]):
        return "celestial"
    if any(k in r for k in ["fairy", "satyr", "dryad", "harengon", "fey"]):
        return "fey"
    if any(k in r for k in ["triton", "merfolk", "locathah", "sea"]):
        return "aquatic"
    if any(k in r for k in ["lizard", "tabaxi", "kenku", "tortle", "aarakocra", "beast"]):
        return "beastfolk"
    if any(k in r for k in ["warforged", "autognome", "construct"]):
        return "construct"
    if any(k in r for k in ["gith"]):
        return "gith"
    if any(k in r for k in ["kalashtar", "psion"]):
        return "psionic"
    if any(k in r for k in ["genasi", "element"]):
        return "elemental"
    if any(k in r for k in ["plasmoid", "thri-kreen", "thri", "kreen"]):
        return "alien"
    if any(k in r for k in ["firbolg", "goliath", "giant"]):
        return "giantkin"
    if any(k in r for k in ["reborn", "dhampir", "hexblood", "undead"]):
        return "undead"
    return "human"

# Base family corpora (seed words). Lightly flavored & fictional.
FAMILY_COUNTRY_CORPORA = {
    "elven": [
        "Aelvarion","Quelanore","Vael'tharis","Elarion","Nyelithar","Thalorien","Syltharis","Lethariel",
        "Caelivor","Evendriel","Faeloria","Vaenlith","Saelithir","Ilvanor","Selvaris","Yavandor"
    ],
    "dwarven": [
        "Khazgrund","Barak-Drom","Stonehallow","Durkharaz","Karag-Thrun","Grumhold","Khuldarim",
        "Bharazdum","Drakzakar","Morn-Uzdir","Uldrunir","Brundrakk","KragVorn","Hammerdeep"
    ],
    "orcish": [
        "Gor'kul","Nargoth","Urzhakaar","Grashnak","Mograth","Krathgor","Uzgul","Thrukmaar","Vargor",
        "Zhurmok","Ragthuk","Skarnash","Brugra'Dar","Druzhmaar"
    ],
    "human": [
        "Ardenia","Valoria","Cendria","Westmarch","Norhaven","Tarsia","Eldoria","Hawksreach","Silbury",
        "Redwyn","Highmere","Thornwell","Greywatch","Kingsvale","Ashbourne","Varynthal"
    ],
    "fiendish": [
        "Vhalgor","Malzor","Kharzun","Xerthaal","Nerazoth","Balzaryn","Zhak'mor","Inferenia","Cindercrux",
        "Oblivionis","Skornexus","Duskmaul","Hellsreach","Nethrazar"
    ],
    "celestial": [
        "Aurelia","Empyros","Solareth","Azurantia","Hallowspire","Seraphel","Luminaris","Elyndorion",
        "Caelestia","Radiant Vale","Sanctaris","Vaelorion","Ardentia"
    ],
    "fey": [
        "Evergloam","Thistledown","Midsummer Vale","Gloamwild","Lorienn","Bramblewynd","Moonhollow",
        "Everspring","Starmeadow","Whisperbough","Faerwyn","Glimmerfen"
    ],
    "aquatic": [
        "Pelagia","Thalassor","Nerithis","Azuraquor","Seabright","Corallene","Okeanos","Aqualeth",
        "Tridentis","Marithal","Deepmere","Abylon","Brinecrown"
    ],
    "beastfolk": [
        "Skarrak","Zan'Xotl","T'zinkal","Clawhaven","Feathercrest","Sunscale","Shellgrove","Shadowfen",
        "Rockroost","Windperch","Savannah's Rest","Scalehold"
    ],
    "construct": [
        "Brassforge","Oxiron","Gearhold","Clockhaven","Mechadia","Steelspire","Ironmarch","Brassward",
        "Cobrevia","Galvanum","Anvilgate","Arcplinth"
    ],
    "gith": [
        "Rrak'mar","Zerkrith","Xam'athar","Vorkith","Sha'shar","Thramaz","Kith'ra","Zaarith","Githanor"
    ],
    "psionic": [
        "Sarashan","Tarashir","Quorath","Mindora","Sorashtal","Kalethir","Dreamspire","Reverion","Syllash"
    ],
    "elemental": [
        "Pyraxis","Aerithon","Hydrassa","Terrakhan","Volcaryn","Zephyria","Tempestar","Cinderstone",
        "Tidehome","Earthenhall"
    ],
    "alien": [
        "K'thak'ra","Zzirith","Ch'tok","Ool'var","Plasmor","Kreenath","Xil'khet","Tkhazz","Zi'vaag"
    ],
    "giantkin": [
        "Grimholm","Stonefell","Skylance Plateau","Bouldercrest","Thrymreach","Glacierguard","Stormpeak"
    ],
    "undead": [
        "Nocturne","Gravesend","Mortavia","Nightfall Dominion","Duskmere","Cairnshroud","Umbershade",
        "Necropolis of Varr"
    ],
}

FAMILY_CITY_CORPORA = {
    "elven": [
        "Lethariel","Elyndor","Quelith","Vaelara","Thalanor","Nyelith","Selvari","Aelion","Yavandar",
        "Faelir","Caelora","Saelith"
    ],
    "dwarven": [
        "Khuldun","Brundrak","Bargrum's Gate","Anvildeep","Morgrin Hold","Uldrak","Kharvek","Durzak",
        "Grunyar","Stonehearth"
    ],
    "orcish": [
        "Gharzug","Brugnak","Urmash","Throkk","Grashnak","Kragtor","Grothul","Nargul","Ragthuk"
    ],
    "human": [
        "Highmere","Kingsport","Riverwatch","Brightgate","Eldham","Westrun","Redwyn","Tamsin's Rest",
        "Cedric's Cross","Rowanfield"
    ],
    "fiendish": [
        "Cinderfall","Ashspire","Skorn Gate","Balzar","Nethra","Gloomarch","Hellscar","Vhalrim"
    ],
    "celestial": [
        "Aurelion","Seraph's Rise","Sunspire","Hallowreach","Lumina","Empyrean","Sanctum Vale"
    ],
    "fey": [
        "Glimmergrove","Moonhollow","Mistfen","Starmead","Thistlebrook","Evermere","Gloamsend"
    ],
    "aquatic": [
        "Thalassa","Pearlspire","Coralhaven","Nerissa","Deepcrest","Tidecall","Seaflower"
    ],
    "beastfolk": [
        "Skarr's Den","Featherhome","Shellbay","Sunscale Town","Windperch","Shadowbloom","Rockroost"
    ],
    "construct": [
        "Gearfall","Copper Row","Anvilcourt","Brasshaven","Cogspire","Steel Quay","Rivet Wharf"
    ],
    "gith": [
        "Rrak'eth","Zerith","Kithmar","Thramar","Shar'ka","Xameth","Zaara"
    ],
    "psionic": [
        "Dream's End","Quiet Harbor","Reverie","Mindspire","Kalastar","Sora's Gate","Tranceport"
    ],
    "elemental": [
        "Cindershade","Zephyr's Rest","Tidewall","Quakeshield","Fumarole","Airstair","Floodmarch"
    ],
    "alien": [
        "K'thak","Zzir","Ch'taa","Ool","Plasmor Minor","Xilk","Tkhaz"
    ],
    "giantkin": [
        "Stonepost","Cloudcroft","Peakwatch","Thrym's Step","Skyhold","Boulderbar"
    ],
    "undead": [
        "Gravemarch","Skullport","Duskhaven","Cairnreach","Nightward","Mortis Gate","Umbershade"
    ],
}

def build_per_race_corpora(race_names: Sequence[str]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Create per-race corpora by mapping each race to a family seed list."""
    country_corpora: Dict[str, List[str]] = {}
    city_corpora: Dict[str, List[str]] = {}
    for race in race_names:
        fam = race_family(race)
        country_corpora[race.lower()] = FAMILY_COUNTRY_CORPORA.get(fam, FAMILY_COUNTRY_CORPORA["human"])
        city_corpora[race.lower()]    = FAMILY_CITY_CORPORA.get(fam,    FAMILY_CITY_CORPORA["human"])
    return country_corpora, city_corpora

# =========================================================
# Full simulation — fast & feature-complete (+ Exodus + Military + Countries/Capitals + Demographics + Naming)
# =========================================================
def simulate_population_from_heightmap(
    heightmap_xyz,
    # grid override (optional)
    H=None, W=None,
    # sea & biome mapping
    sea_level=0.0, Hmax=1.0, biome_thresholds=(0.10,0.40,0.65,0.85),
    # seeding / initialization
    initial_total_population=1_000_000, expected_settlements=250, spawn_gamma=1.3,
    # demography & capacity
    r_birth=0.022, r_death=0.014, K0=3000.0,
    kappa_water_capacity=1.0, lambda_water_capacity=8.0,  # capacity uplift decay from water
    # attractiveness (directed flow) — includes sand corridor & stronger water pull
    alpha_w=1.6, alpha_p=1.0, alpha_b=0.5, lambda_mig=8.0,
    water_affinity_gain=0.35, sand_corridor_boost=0.50, sand_mask_weight=None,
    density_sigma=1.8, rho_half=1800.0, pref_noise_sigma=0.12,
    # migration & flows (one-step each)
    m_dir=0.05,                      # directed (A_att)
    m_gravity=0.03, grav_sigma=7.0,  # gravity toward smoothed mass
    m_sprawl=0.02, urban_k_frac=0.60,# sprawl toward urban perimeter
    m_explore=0.004, m_diff=0.008,   # exploration + diffusion
    # infra & memory
    infra_sigma=2.0, infra_threshold=8000.0, infra_strength=0.20,
    memory_decay=0.98, memory_gain_scale=0.02, crowding_penalty=0.65,
    # Life coupling
    life_enabled=True, life_rule="B3/S23",
    life_pop_threshold=1200.0, life_density_norm=6000.0,
    life_p_birth=0.9, life_p_survive=0.95, life_noise_sigma=0.05,
    life_birth_boost=0.25, life_death_suppression=0.15,
    life_capacity_boost=0.20, life_migration_bias=0.25,
    # catastrophes (regional & persistent)
    catastrophe_prob=0.03, catastrophe_events_mean=1.0,
    region_shape_probs=(0.6, 0.4), disk_radius_range=(3, 8),
    box_hw_range=((5, 14), (5, 14)),
    catastrophe_types=None,
    # congestion
    delta_death=0.35, delta_forced_move=0.25,
    # churn (fast normal approx)
    lam_out=4e-4, lam_in=4e-4,
    # CULTURE factor (persistent towns)
    culture_enabled=True,
    town_threshold=2500.0, culture_radius=1, culture_persist_years=25,
    culture_min_abs=800.0, culture_min_frac=0.06,

    # ------------------ EXODUS ------------------
    exodus_enabled=True,
    exodus_prob=0.06, exodus_events_mean=1.0,
    exodus_min_pop=None, exodus_group_frac=0.25,
    exodus_min_dist=20, exodus_max_dist=None,
    exodus_target_bias_water=1.0, exodus_target_bias_empty=0.7,
    exodus_culture_factor=0.5,
    # ---------------------------------------------------

    # ------------------ MILITARY (sim-owned) ------------------
    military_enabled=True,                  # maintain per-cell stance in sim
    record_military=False,                  # store snapshots (aligned with save_every)
    military_init="random",                 # "random" init in [-1,1]
    military_forcing_enabled=False,         # periodic anti-neutral push
    military_neutral_band=0.08,             # |stance| <= band is "neutral"
    military_period=24,                     # frames/years per cycle
    military_strength=0.5,                  # max push per step (scaled by envelope & room)
    military_shape="sin",                   # "sin" | "saw" | "linear"
    # ---------------------------------------------------

    # ------------------ COUNTRIES & CAPITALS (computed inside sim) ------------------
    record_countries=False,            # store per-frame country labels and capitals
    overlay_cluster_threshold=2500.0,  # settlements >= threshold become seeds
    overlay_use_culture=False,         # use culture anchors instead of pop threshold
    country_base_radius=2.0,           # growth radius base
    country_scale=12.0,                # growth per |stance|
    country_gamma=1.0,                 # nonlinearity for |stance|
    country_min_stance=0.05,           # ignore near-neutral clusters
    # ---------------------------------------------------

    # ------------------ DEMOGRAPHICS (per country) ------------------
    demographics_spec=None,            # dict: name -> (rarity_score:int>=1, stance_preference in [-1,1])
    record_demographics=True,          # store per-frame demographics per country
    rarity_exponent=1.0,               # effective availability ~ 1 / rarity^rarity_exponent
    demographics_sigma=0.5,            # stance affinity width (bigger = laxer matching)
    demographics_power=2.0,            # p in exp( - (|Δstance|/sigma)^p )
    # ---------------------------------------------------

    # ------------------ NAMING (per country & capital) ------------------
    naming_enabled=True,
    markov_order=3,
    country_name_len_range=(7, 14),
    city_name_len_range=(5, 12),
    naming_seed_offset=101,
    # ---------------------------------------------------

    # run control
    years=300, seed=42, clip_negative=True, dtype=np.float32,
    save_every=1, return_meta=False, record_attractiveness=False,
):
    """
    Feature-complete, vectorized population simulation (heightmap-in).
    """
    rng = np.random.default_rng(seed)

    # ----- local helpers for countries/capitals/demographics (sim-owned) -----
    def _stance_map_local(pop, military_cell, water_mask, thr, use_culture, culture_anchor=None):
        """Return (stance_map[-1..1], lbl, nlab, mask) with pop-weighted cluster means."""
        if use_culture and (culture_anchor is not None):
            mask = (culture_anchor.astype(bool)) & (~water_mask)
        else:
            mask = (pop >= float(thr)) & (~water_mask)

        lbl_, nlab_ = label(mask.astype(np.int8),
                            structure=np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.int8))
        if nlab_ == 0:
            out = np.zeros_like(pop, dtype=np.float32)
            out[water_mask] = 0.0
            return out, lbl_, nlab_, mask

        lab_flat = lbl_.ravel(); valid = lab_flat > 0; idx = lab_flat[valid].astype(np.int64)
        pop_flat = pop.ravel()[valid].astype(np.float64)
        m_flat   = military_cell.ravel()[valid].astype(np.float64)

        sum_pop = np.bincount(idx, weights=pop_flat, minlength=nlab_ + 1)
        sum_m   = np.bincount(idx, weights=pop_flat * m_flat, minlength=nlab_ + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            means = sum_m[1:] / np.maximum(sum_pop[1:], 1e-12)
        means = np.clip(means, -1.0, 1.0).astype(np.float32)

        means_pad = np.zeros(nlab_ + 1, dtype=np.float32); means_pad[1:] = means
        stance_map = means_pad[lbl_]
        stance_map[water_mask] = 0.0
        return stance_map, lbl_, nlab_, mask

    def _country_labels_clip_land_local(stance_map, lbl_, nlab_, mask, water_mask,
                                        base_radius, scale, gamma, min_stance):
        """Grow each settlement into a 'country' over LAND only, radius ~ |stance|."""
        if nlab_ == 0:
            return np.zeros_like(lbl_, dtype=np.int32)

        lab_flat = lbl_.ravel(); valid = lab_flat > 0; idx = lab_flat[valid].astype(np.int64)
        s_flat = stance_map.ravel()[valid].astype(np.float64)

        sum_abs = np.bincount(idx, weights=np.abs(s_flat), minlength=nlab_ + 1)
        cnt     = np.bincount(idx, minlength=nlab_ + 1).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_abs = sum_abs[1:] / np.maximum(cnt[1:], 1e-12)
        mean_abs = np.nan_to_num(mean_abs, nan=0.0).astype(np.float32)
        mean_abs[mean_abs < float(min_stance)] = 0.0

        radii = country_base_radius + country_scale * (mean_abs ** country_gamma)
        radii = np.maximum(0.0, radii).astype(np.float32)

        inv = ~mask
        dist, ind = distance_transform_edt(inv, return_indices=True)
        ny, nx = ind[0], ind[1]
        nearest_lbl = lbl_[ny, nx]

        radii_pad = np.zeros(nlab_ + 1, dtype=np.float32); radii_pad[1:] = radii
        R = radii_pad[nearest_lbl]

        inside = (nearest_lbl > 0) & (dist <= R) & (~water_mask)
        return np.where(inside, nearest_lbl, 0).astype(np.int32)

    def _country_capitals_local(pop, country_lbl):
        """
        Population-weighted center of mass per country label.
        Returns cy, cx, labels (1D float32/float32/int32).
        """
        Hh, Ww = pop.shape
        L = country_lbl
        labels = np.unique(L[L > 0])
        if labels.size == 0:
            return (np.empty(0, np.float32), np.empty(0, np.float32), np.empty(0, np.int32))

        yy_, xx_ = np.indices((Hh, Ww))
        Lf = L.ravel()
        Pf = pop.ravel().astype(np.float64)
        yf = yy_.ravel().astype(np.float64)
        xf = xx_.ravel().astype(np.float64)

        max_lab = int(labels.max())
        sumP   = np.bincount(Lf, weights=Pf, minlength=max_lab+1)
        sumPy  = np.bincount(Lf, weights=Pf * yf, minlength=max_lab+1)
        sumPx  = np.bincount(Lf, weights=Pf * xf, minlength=max_lab+1)

        valid = labels[sumP[labels] > 0]
        if valid.size == 0:
            return (np.empty(0, np.float32), np.empty(0, np.float32), np.empty(0, np.int32))

        cy = (sumPy[valid] / sumP[valid]).astype(np.float32)
        cx = (sumPx[valid] / sumP[valid]).astype(np.float32)
        return cy, cx, valid.astype(np.int32)

    def _percent_2dp_allocation(weights):
        """
        Turn nonnegative weights into EXACT 2dp percentages that sum to 100.00.
        Uses Largest Remainder Method at 2dp (basis points).
        """
        w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
        tot = w.sum()
        if tot <= 0:
            return np.zeros_like(w, dtype=np.float32)
        p_bp = (w / tot) * 10000.0
        floor_bp = np.floor(p_bp).astype(np.int64)
        remainder = int(10000 - floor_bp.sum())
        if remainder != 0:
            frac = p_bp - floor_bp
            if remainder > 0:
                order = np.argsort(-frac)[:remainder]
                floor_bp[order] += 1
            else:
                order = np.argsort(frac)[:(-remainder)]
                floor_bp[order] -= 1
        return (floor_bp.astype(np.float32) / 100.0)

    # -------- Terrain from heightmap --------
    h_raw = heightmap_xyz_to_grid(heightmap_xyz, H=H, W=W, dtype=dtype)
    H, W = h_raw.shape
    water_mask = h_raw <= sea_level
    land_mask  = ~water_mask

    land_scaled = np.zeros_like(h_raw, dtype=dtype)
    if np.any(land_mask):
        lo, hi = h_raw[land_mask].min(), h_raw[land_mask].max()
        denom = (hi - lo) if hi > lo else 1.0
        land_scaled[land_mask] = ((h_raw[land_mask] - lo) / denom) * Hmax

    biome = biome_id_from_scaled(land_scaled, water_mask, thresholds=biome_thresholds)
    D = distance_transform_edt(~water_mask.astype(bool)).astype(dtype)

    # -------- Static terrain fields --------
    W_bio_mig = biome_migration_weight(biome).astype(dtype)
    k_bio     = biome_capacity_factor(biome).astype(dtype)
    death_mult = biome_death_multiplier(biome).astype(dtype)
    sand_mask = (biome == 1)

    # Distance-to-water preference component for migration
    S_w = np.zeros((H, W), dtype=dtype)
    S_w[~water_mask] = np.exp(-D[~water_mask] / float(lambda_mig)).astype(dtype)

    # Carrying capacity uplift near water (separate decay)
    water_uplift = np.zeros((H, W), dtype=dtype)
    water_uplift[~water_mask] = np.exp(-D[~water_mask] / float(lambda_water_capacity)).astype(dtype)
    K_base = (K0 * k_bio * (1.0 + kappa_water_capacity * water_uplift)).astype(dtype)
    K_base[water_mask] = 0.0

    # -------- Sparse seeding --------
    pref = W_bio_mig * (1.0 + 0.5 * water_uplift)
    pref[water_mask] = 0.0
    probs = pref.ravel(); total = probs.sum()
    probs = (probs / total) if total > 0 else np.full(H*W, 1.0/(H*W), dtype=dtype)
    seeds = rng.choice(H*W, size=int(max(1, expected_settlements)), replace=False, p=probs)
    sy, sx = np.divmod(seeds, W)
    weights = np.clip(pref[sy, sx] ** spawn_gamma, 1e-9, None).astype(dtype)
    weights /= weights.sum()
    N = np.zeros((H, W), dtype=dtype)
    N[sy, sx] += (initial_total_population * weights).astype(dtype)

    # -------- State fields --------
    cum_mem = np.zeros((H, W), dtype=dtype)
    infra_field = np.zeros((H, W), dtype=dtype)

    # CULTURE state
    if culture_enabled:
        culture_counter = np.zeros((H, W), dtype=np.int32)
        culture_anchor  = np.zeros((H, W), dtype=bool)
    else:
        culture_counter = culture_anchor = None

    # ---- MILITARY per-cell state (sim-owned) ----
    if military_enabled:
        if military_init == "random":
            M_cell = rng.uniform(-1.0, 1.0, size=(H, W)).astype(np.float32)
        else:
            M_cell = rng.uniform(-1.0, 1.0, size=(H, W)).astype(np.float32)
        M_cell[water_mask] = 0.0
        M_runlen = np.zeros((H, W), dtype=np.int32)
        M_seed_sign = np.where(rng.integers(0, 2, size=(H, W), endpoint=False) == 0, -1, 1).astype(np.int8)
        M_seed_sign[water_mask] = 0
    else:
        M_cell = M_runlen = M_seed_sign = None

    # ---- histories ----
    pop_history = []
    att_history = [] if record_attractiveness else None
    mil_history = [] if (military_enabled and record_military) else None
    countries_history = [] if record_countries else None
    capitals_history  = [] if record_countries else None  # list of dicts {"cy","cx","label"}
    demographics_history = [] if (record_countries and record_demographics) else None

    def maybe_store(t, arr_pop, arr_att=None, arr_mil=None):
        if t % save_every == 0:
            pop_history.append(arr_pop.copy())
            if record_attractiveness and (arr_att is not None):
                att_history.append(arr_att.copy())
            if mil_history is not None and (arr_mil is not None):
                mil_history.append(arr_mil.copy())

    # ---- compute baseline A_att for t=0 (no infra, no shocks, no life; memory=0) ----
    S_w_eff0 = np.clip(S_w * (1.0 + water_affinity_gain), 0.0, None) ** dtype(alpha_w)
    rho0     = gaussian_filter(np.zeros_like(N, dtype=np.float32), sigma=density_sigma, mode="reflect").astype(dtype)
    S_pop0   = rho0 / (rho0 + dtype(rho_half))
    crowd0   = np.clip(N / (K_base + 1e-9), 0.0, 2.0)
    crowd_t0 = np.exp(-crowding_penalty * crowd0).astype(dtype)
    mem_t0   = np.zeros_like(N, dtype=dtype)
    A_base0  = (S_w_eff0 * (S_pop0 ** dtype(alpha_p)) * (W_bio_mig ** dtype(alpha_b)) * crowd_t0 * (1.0 + 0.5 * mem_t0))
    A_sand_b0= (biome == 1).astype(dtype) * S_w_eff0 * dtype(sand_corridor_boost)
    A_att0   = A_base0 * (1.0 + A_sand_b0)

    maybe_store(0, N, A_att0, (M_cell if military_enabled and record_military else None))

    # -------- Catastrophes defaults --------
    if catastrophe_types is None:
        catastrophe_types = {
            "war":     {"prob":0.40,"duration_range":(5,15),"mortality":0.04,"forced_migration":0.15,
                        "r_birth_mult":0.65,"r_death_add":0.006,"K_mult":0.92,"m_dir_mult":1.25},
            "famine":  {"prob":0.35,"duration_range":(3, 8),"mortality":0.02,"forced_migration":0.10,
                        "r_birth_mult":0.70,"r_death_add":0.008,"K_mult":0.65,"m_dir_mult":1.10},
            "epidemic":{"prob":0.15,"duration_range":(2, 5),"mortality":0.03,"forced_migration":0.05,
                        "r_birth_mult":0.60,"r_death_add":0.010,"K_mult":1.00,"m_dir_mult":0.85},
            "disaster":{"prob":0.10,"duration_range":(1, 3),"mortality":0.10,"forced_migration":0.20,
                        "r_birth_mult":1.00,"r_death_add":0.000,"K_mult":0.90,"m_dir_mult":1.30},
        }
    c_types = list(catastrophe_types.keys())
    c_probs = np.array([catastrophe_types[k]["prob"] for k in c_types], dtype=np.float64); c_probs /= c_probs.sum()
    active_shocks = []

    # Precompute coordinate grids for exodus distance checks
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    def sample_region_mask():
        flat = N.ravel(); tot = flat.sum()
        if tot <= 0:
            cy, cx = rng.integers(0, H), rng.integers(0, W)
        else:
            idx = rng.choice(H*W, p=(flat / tot)); cy, cx = divmod(idx, W)
        mask = np.zeros((H, W), dtype=bool)
        if rng.random() < region_shape_probs[0]:
            r = int(rng.integers(disk_radius_range[0], disk_radius_range[1] + 1))
            y0, x0 = cy, cx
            mask = (yy - y0)**2 + (xx - x0)**2 <= r**2
        else:
            (hmin,hmax),(wmin,wmax) = box_hw_range
            hh = int(rng.integers(hmin, hmax+1)); ww = int(rng.integers(wmin, wmax+1))
            y0 = np.clip(cy - hh//2, 0, H-1); x0 = np.clip(cx - ww//2, 0, W-1)
            y1 = np.clip(y0 + hh, 0, H);     x1 = np.clip(x0 + ww, 0, W)
            mask[y0:y1, x0:x1] = True
        mask &= (~water_mask)
        return mask

    def maybe_add_shocks():
        if rng.random() >= catastrophe_prob:
            return
        k = max(1, rng.poisson(catastrophe_events_mean))
        for _ in range(k):
            tname = rng.choice(c_types, p=c_probs); p = catastrophe_types[tname]
            dur = int(rng.integers(p["duration_range"][0], p["duration_range"][1] + 1))
            if dur <= 0: continue
            mask = sample_region_mask()
            if not mask.any(): continue
            active_shocks.append({
                "mask": mask, "years_left": dur,
                "mortality": float(p["mortality"]),
                "forced_migration": float(p["forced_migration"]),
                "r_birth_mult": float(p["r_birth_mult"]),
                "r_death_add": float(p["r_death_add"]),
                "K_mult": float(p["K_mult"]),
                "m_dir_mult": float(p["m_dir_mult"]),
            })

    def _military_envelope(runlen):
        if not military_forcing_enabled:
            return 0.0
        if military_period <= 1:
            phase = 1.0
        else:
            phase = (runlen % military_period).astype(np.float32) / float(military_period)
        if military_shape == "sin":
            env = np.sin(np.pi * phase)
        elif military_shape == "saw":
            env = phase
        else:
            env = np.where(phase <= 0.5, 2.0*phase, 2.0*(1.0 - phase)).astype(np.float32)
        return np.clip(env, 0.0, 1.0)

    # =========================
    # Yearly loop (vectorized)
    # =========================
    for t in range(1, years + 1):
        maybe_add_shocks()

        # Infra spillover
        hubs = (N >= infra_threshold) & (~water_mask)
        if hubs.any():
            infra_field = gaussian_filter(hubs.astype(dtype), sigma=infra_sigma, mode="reflect")
            m = infra_field.max(); infra_field = (infra_field / m) if m > 0 else infra_field*0
        else:
            infra_field *= 0

        # Shocks fields + annual shock mortality
        r_birth_mult_field = np.ones((H, W), dtype=dtype)
        r_death_add_field  = np.zeros((H, W), dtype=dtype)
        K_mult_field       = np.ones((H, W), dtype=dtype)
        m_dir_mult_field   = np.ones((H, W), dtype=dtype)
        if active_shocks:
            for sh in active_shocks:
                M = sh["mask"]
                mort = sh["mortality"]
                if mort > 0: N[M] *= (1.0 - mort)
                r_birth_mult_field[M] *= sh["r_birth_mult"]
                r_death_add_field[M]  += sh["r_death_add"]
                K_mult_field[M]       *= sh["K_mult"]
                m_dir_mult_field[M]   *= sh["m_dir_mult"]

        # Effective capacity
        K_eff = (K_base * (1.0 + infra_strength * infra_field) * K_mult_field).astype(dtype)
        K_eff[water_mask] = 0.0

        # Life coupling
        if life_enabled:
            _, L = conway_prob_step(
                N=N, K_eff=K_eff, rule_str=life_rule,
                pop_threshold=life_pop_threshold, density_norm=life_density_norm,
                p_birth=life_p_birth, p_survive=life_p_survive, noise_sigma=life_noise_sigma, rng=rng
            )
            K_eff = K_eff * (1.0 + life_capacity_boost * L)
        else:
            L = np.zeros_like(N, dtype=dtype)

        # Demography (logistic)
        r_birth_eff = r_birth * r_birth_mult_field * (1.0 + life_birth_boost * L)
        r_death_eff = (r_death + r_death_add_field) * death_mult * (1.0 - life_death_suppression * L)
        r_net_local = r_birth_eff - r_death_eff
        growth = r_net_local * N * (1.0 - np.divide(N, K_eff, out=np.zeros_like(N), where=(K_eff > 0)))
        N_prime = N + growth
        N_prime[water_mask] = 0

        # Memory update
        cum_mem = memory_decay * cum_mem + memory_gain_scale * N_prime

        # ---------- Attractiveness for standard directed flow ----------
        S_w_eff = np.clip(S_w * (1.0 + water_affinity_gain), 0.0, None) ** dtype(alpha_w)
        rho   = gaussian_filter(N.astype(np.float32), sigma=density_sigma, mode="reflect").astype(dtype)
        S_pop = rho / (rho + dtype(rho_half))
        crowd = np.clip(N / (K_eff + 1e-9), 0.0, 2.0)
        crowd_term = np.exp(-crowding_penalty * crowd).astype(dtype)
        mem_term   = np.clip(cum_mem / (cum_mem.max() + 1e-9), 0, 1).astype(dtype) ** dtype(0.8)

        A_base = (S_w_eff * (S_pop ** dtype(alpha_p)) * (W_bio_mig ** dtype(alpha_b)) * crowd_term * (1.0 + 0.5 * mem_term))
        A_sand_bonus = sand_mask.astype(dtype) * S_w_eff * dtype(sand_corridor_boost)
        A_att = A_base * (1.0 + A_sand_bonus)
        if life_enabled and life_migration_bias != 0:
            A_att = A_att * (1.0 + life_migration_bias * L)
        if pref_noise_sigma > 0:
            A_att = A_att * np.exp(rng.normal(0.0, pref_noise_sigma, size=A_att.shape)).astype(dtype)

        # -------- Flows (one-step, vectorized) --------
        # (a) Directed flow
        M_dir_out = (m_dir * m_dir_mult_field) * N_prime
        N_after = N_prime - M_dir_out
        inflow_dir = one_step_route(M_dir_out, A_att, water_mask)

        # (b) Gravity flow — toward smoothed mass field
        if m_gravity > 0:
            Pgrav = gaussian_filter(N_after.astype(np.float32), sigma=grav_sigma, mode="reflect").astype(dtype)
            M_grav_out = m_gravity * N_after
            N_after -= M_grav_out
            inflow_grav = one_step_route(M_grav_out, Pgrav, water_mask)
        else:
            inflow_grav = 0.0

        # (c) Urban sprawl — to urban perimeter
        if m_sprawl > 0:
            U = (N >= urban_k_frac * K_eff) & (~water_mask)
            U_neighbors = convolve(U.astype(np.float32), KERNEL_8, mode="reflect")
            A_sprawl = U_neighbors * (~U)
            M_sprawl_out = m_sprawl * N_after
            N_after -= M_sprawl_out
            inflow_sprawl = one_step_route(M_sprawl_out, A_sprawl, water_mask)
        else:
            inflow_sprawl = 0.0

        # (d) Exploration (small random neighbor diffusion)
        if m_explore > 0:
            avg_nbr = convolve(N_after, KERNEL_8, mode="wrap") / 8.0
            N_after = N_after + m_explore * (avg_nbr - N_after)

        # (e) Diffusion (isotropic)
        if m_diff > 0:
            avg_nbr2 = convolve(N_after, KERNEL_8, mode="wrap") / 8.0
            N_after = N_after + m_diff * (avg_nbr2 - N_after)

        # Aggregate flows
        N_temp = N_after + inflow_dir + (inflow_grav if isinstance(inflow_grav, np.ndarray) else 0.0) + (inflow_sprawl if isinstance(inflow_sprawl, np.ndarray) else 0.0)
        N_temp[water_mask] = 0

        # Congestion
        over = np.maximum(0.0, N_temp - K_eff)
        extra_death = delta_death * over
        forced_out = delta_forced_move * over
        N_temp = N_temp - (extra_death + forced_out)
        N_temp[N_temp < 0] = 0
        N_temp += one_step_route(forced_out, A_att, water_mask)
        N_temp[water_mask] = 0

        # Persistent shock-driven outflow
        if active_shocks:
            total_forced = np.zeros_like(N_temp, dtype=dtype)
            for sh in active_shocks:
                if sh["forced_migration"] <= 0: continue
                total_forced += sh["forced_migration"] * (sh["mask"].astype(dtype)) * N_temp
            N_temp -= total_forced
            N_temp += one_step_route(total_forced, A_att, water_mask)
            N_temp[water_mask] = 0

        # Churn
        if lam_in > 0 or lam_out > 0:
            mean_out = lam_out * N_temp
            mean_in  = lam_in  * N_temp
            Xi_out = np.maximum(0.0, rng.normal(mean_out, np.sqrt(mean_out + 1e-9), size=N_temp.shape)).astype(dtype)
            Xi_in  = np.maximum(0.0, rng.normal(mean_in,  np.sqrt(mean_in  + 1e-9), size=N_temp.shape)).astype(dtype)
            N_next = N_temp - Xi_out + Xi_in
        else:
            N_next = N_temp

        if clip_negative: N_next[N_next < 0] = 0
        N_next[water_mask] = 0

        # ------------------ EXODUS ------------------
        if exodus_enabled and rng.random() < exodus_prob:
            k_events = max(1, rng.poisson(exodus_events_mean))
            min_pop = float(town_threshold if exodus_min_pop is None else exodus_min_pop)
            rho_next = gaussian_filter(N_next.astype(np.float32), sigma=density_sigma, mode="reflect").astype(dtype)
            S_pop_next = rho_next / (rho_next + dtype(rho_half))
            cand = (N_next >= min_pop) & (~water_mask)
            if np.any(cand):
                src_probs = (N_next * cand).ravel()
                src_probs_sum = src_probs.sum()
                if src_probs_sum > 0:
                    src_probs = src_probs / src_probs_sum
                    for _ in range(k_events):
                        sidx = rng.choice(H*W, p=src_probs)
                        sy, sx = divmod(sidx, W)
                        src_pop = float(N_next[sy, sx])
                        if src_pop < min_pop:
                            continue
                        group = max(1.0, exodus_group_frac * src_pop)
                        N_next[sy, sx] = max(0.0, src_pop - group)

                        dist = np.maximum(np.abs(yy - sy), np.abs(xx - sx))
                        target_mask = (~water_mask) & (dist >= int(exodus_min_dist))
                        if exodus_max_dist is not None:
                            target_mask &= (dist <= int(exodus_max_dist))

                        if exodus_target_bias_water > 0 or exodus_target_bias_empty > 0:
                            w = np.ones((H, W), dtype=np.float64)
                            if exodus_target_bias_water > 0:
                                w *= (1e-6 + S_w)
                            if exodus_target_bias_empty > 0:
                                w *= (1.0 - S_pop_next + 1e-6)
                        else:
                            w = np.ones((H, W), dtype=np.float64)

                        w *= target_mask.astype(np.float64)
                        wsum = w.sum()
                        if wsum <= 0:
                            fallback = np.where(target_mask.ravel())[0]
                            if fallback.size == 0:
                                N_next[sy, sx] += group
                                continue
                            tidx = int(rng.choice(fallback))
                        else:
                            w = (w / wsum).ravel()
                            tidx = int(rng.choice(H*W, p=w))

                        ty, tx = divmod(tidx, W)
                        N_next[ty, tx] += group

                        if culture_enabled:
                            boost = int(max(0, round(exodus_culture_factor * culture_persist_years)))
                            culture_counter[ty, tx] = max(culture_counter[ty, tx], boost)

        # ===== CULTURE =====
        if culture_enabled:
            size = 2 * int(culture_radius) + 1
            local_max = maximum_filter(N, size=size, mode="reflect")
            stable = local_max >= float(town_threshold)
            culture_counter = np.where(stable, culture_counter + 1, 0)
            culture_anchor = culture_anchor | (culture_counter >= int(culture_persist_years))
            floor = np.maximum(culture_min_abs, culture_min_frac * K_eff)
            N_next = np.where(culture_anchor & (~water_mask), np.maximum(N_next, floor), N_next)

        # ---- MILITARY periodic forcing ----
        if military_enabled and military_forcing_enabled:
            neutral = (np.abs(M_cell) <= float(military_neutral_band))
            M_runlen = np.where(neutral, M_runlen + 1, 0)
            env = _military_envelope(M_runlen).astype(np.float32)
            sgn = np.sign(M_cell).astype(np.int8)
            sgn = np.where(sgn == 0, M_seed_sign, sgn).astype(np.int8)
            room = (1.0 - np.clip(np.abs(M_cell), 0.0, 1.0)).astype(np.float32)
            delta = (military_strength * env * room).astype(np.float32)
            M_cell = np.clip(M_cell + (sgn.astype(np.float32) * delta), -1.0, 1.0)
            M_cell[water_mask] = 0.0

        # ----- Countries, Capitals, and Demographics (optional recording) -----
        if record_countries:
            P_for_labels = N_next.astype(np.float32)
            M_for_labels = (M_cell.astype(np.float32) if military_enabled else np.zeros_like(P_for_labels, dtype=np.float32))

            stance_map, lbl_, nlab_, mask_ = _stance_map_local(
                pop=P_for_labels,
                military_cell=M_for_labels,
                water_mask=water_mask,
                thr=overlay_cluster_threshold,
                use_culture=overlay_use_culture,
                culture_anchor=(culture_anchor if culture_enabled else None)
            )

            c_lbl = _country_labels_clip_land_local(
                stance_map, lbl_, nlab_, mask_, water_mask,
                base_radius=country_base_radius,
                scale=country_scale,
                gamma=country_gamma,
                min_stance=country_min_stance
            )

            cy, cx, lab_ids = _country_capitals_local(P_for_labels, c_lbl)
            countries_history.append(c_lbl.astype(np.int32))
            capitals_history.append({"cy": cy, "cx": cx, "label": lab_ids})

            # Demographics per label
            if record_demographics and (demographics_spec is not None) and len(demographics_spec) > 0:
                lab_flat = lbl_.ravel()
                valid = lab_flat > 0
                idx = lab_flat[valid].astype(np.int64)
                pop_flat = P_for_labels.ravel()[valid].astype(np.float64)
                stance_flat = stance_map.ravel()[valid].astype(np.float64)
                sum_pop = np.bincount(idx, weights=pop_flat, minlength=nlab_ + 1)
                sum_ms  = np.bincount(idx, weights=pop_flat * stance_flat, minlength=nlab_ + 1)
                stance_means = np.zeros(nlab_ + 1, dtype=np.float32)
                with np.errstate(divide="ignore", invalid="ignore"):
                    stance_means[1:] = (sum_ms[1:] / np.maximum(sum_pop[1:], 1e-12)).astype(np.float32)

                labels_present = np.unique(c_lbl[c_lbl > 0]).astype(np.int32)
                demo_for_frame = {}

                race_names = list(demographics_spec.keys())
                rarity = np.array([max(1, int(demographics_spec[n][0])) for n in race_names], dtype=np.float32)
                stance_pref = np.array([float(demographics_spec[n][1]) for n in race_names], dtype=np.float32)
                availability = 1.0 / np.power(rarity, float(rarity_exponent))

                sigma = float(demographics_sigma)
                pwr   = float(demographics_power)
                eps   = 1e-12

                for lab in labels_present:
                    mean_stance = float(stance_means[lab]) if lab < stance_means.size else 0.0
                    d = np.abs(mean_stance - stance_pref)
                    affinity = np.exp(-np.power(d / max(sigma, eps), pwr))
                    weights_local = availability * affinity
                    if not np.isfinite(weights_local).all() or weights_local.sum() <= 0:
                        weights_local = np.ones_like(weights_local)
                    pct = _percent_2dp_allocation(weights_local)
                    mask_nonzero = pct > 0.0
                    pct = np.round(pct, 2)
                    demo_for_frame[int(lab)] = {
                        race_names[i]: float(pct[i])
                        for i in range(len(race_names))
                        if mask_nonzero[i]
                    }
                demographics_history.append(demo_for_frame)

        # Expire shocks
        if active_shocks:
            for sh in active_shocks: sh["years_left"] -= 1
            active_shocks = [sh for sh in active_shocks if sh["years_left"] > 0]

        # advance state
        N = N_next.astype(dtype, copy=False)
        maybe_store(t, N, A_att, (M_cell if (mil_history is not None) else None))

    # <<< END OF LOOP >>>
    military_trait = (M_cell.astype(np.float32) if military_enabled else None)

    # Prepare meta
    countries_last = (countries_history[-1] if (record_countries and countries_history) else None)
    capitals_last  = (capitals_history[-1]  if (record_countries and capitals_history)  else None)
    demographics_last = (demographics_history[-1] if (record_countries and record_demographics and demographics_history) else None)

    # ------------------ NAMING (use demographics distribution) ------------------
    country_names_last = None
    capital_names_last = None
    country_race_last  = None
    if naming_enabled and (countries_last is not None) and (capitals_last is not None) and (demographics_last is not None):
        # Build corpora per race based on the races present in demographics_spec
        race_list = list(demographics_spec.keys())
        country_corpora, city_corpora = build_per_race_corpora(race_list)
        # Two generators with deterministic seeds
        country_gen = NameGenerator(order=int(markov_order), seed=int(seed) + int(naming_seed_offset) + 7, corpora=country_corpora)
        city_gen    = NameGenerator(order=int(markov_order), seed=int(seed) + int(naming_seed_offset) + 19, corpora=city_corpora)

        labels = capitals_last["label"].astype(int)
        cy_arr = capitals_last["cy"]; cx_arr = capitals_last["cx"]
        # map label -> coord
        cap_coord = {int(l): (float(cy_arr[i]), float(cx_arr[i])) for i, l in enumerate(labels)}
        country_names_last = {}
        capital_names_last = {}
        country_race_last  = {}
        used_country = set()
        used_city    = set()

        for lab in labels:
            demo = demographics_last.get(int(lab), {})
            if not demo:
                # fallback if no demographics: pick most common by availability (rarity=1 first)
                demo = {}
                for r, (rar, _stance) in demographics_spec.items():
                    if int(rar) == 1:
                        demo[r] = 1.0
                if not demo:
                    demo = {"humans": 1.0}

            races = np.array(list(demo.keys()))
            weights = np.array([max(0.0, float(v)) for v in demo.values()], dtype=np.float64)
            if weights.sum() <= 0:
                races = np.array(["humans"])
                weights = np.array([1.0], dtype=np.float64)
            probs = weights / weights.sum()
            # sample one race by demographics distribution
            idx = np.argmax(probs) if len(probs) == 1 else np.random.default_rng(int(seed) + int(lab) + 313).choice(len(races), p=probs)
            chosen_race = races[idx].lower()

            # Generate country and city names
            clen = int(np.random.default_rng(int(seed) + int(lab) + 911).integers(country_name_len_range[0], country_name_len_range[1] + 1))
            cstart = None
            # enforce uniqueness lightly
            tries = 0
            while True:
                cname = country_gen.generate_exact_length(clen, styles=chosen_race, starts_with=cstart, allow_punct=True)
                tries += 1
                if (cname not in used_country) or tries > 10:
                    used_country.add(cname)
                    break

            tlen = int(np.random.default_rng(int(seed) + int(lab) + 137).integers(city_name_len_range[0], city_name_len_range[1] + 1))
            tries = 0
            while True:
                capname = city_gen.generate_exact_length(tlen, styles=chosen_race, allow_punct=True)
                tries += 1
                if (capname not in used_city) or tries > 10:
                    used_city.add(capname)
                    break

            country_names_last[int(lab)] = cname
            capital_names_last[int(lab)] = capname
            country_race_last[int(lab)]  = chosen_race

    # ------------------ COUNTRY STATS (Total pop + military metrics) ------------------
    country_stats_last = None
    if countries_last is not None:
        L = countries_last.astype(np.int32)
        labels_present = np.unique(L[L > 0]).astype(int)
        stats = {}
        for lab in labels_present:
            mask = (L == lab)
            tot_pop = float(N[mask].sum())
            if military_enabled and (military_trait is not None):
                mvals = military_trait[mask].astype(np.float64)
                weights = N[mask].astype(np.float64)
                wsum = weights.sum()
                if wsum > 0:
                    mean_stance = float((mvals * weights).sum() / wsum)         # signed mean in [-1,1]
                    strength    = float((np.abs(mvals) * weights).sum() / wsum) # 0..1
                else:
                    mean_stance = 0.0
                    strength = 0.0
            else:
                mean_stance = 0.0
                strength = 0.0
            stats[int(lab)] = {
                "total_population": tot_pop,
                "mean_stance": mean_stance,
                "strength": strength
            }
        country_stats_last = stats

    meta = dict(
        biome=biome, water_mask=water_mask, D=D, land_scaled=land_scaled, h_raw=h_raw,
        culture_anchor=(culture_anchor if culture_enabled else None),
        K_base=K_base,
        military_trait=military_trait,
        countries_history=(countries_history if record_countries else None),
        capitals_history=(capitals_history  if record_countries else None),
        countries_last=countries_last,
        capitals_last=capitals_last,
        demographics_history=(demographics_history if (record_countries and record_demographics) else None),
        demographics_last=demographics_last,
        country_names_last=country_names_last,
        capital_names_last=capital_names_last,
        country_race_last=country_race_last,
        country_stats_last=country_stats_last,   # <<< NEW
    )

    if return_meta:
        if record_attractiveness and (att_history is not None) and (mil_history is not None):
            return ((pop_history, att_history), mil_history), meta
        elif record_attractiveness and (att_history is not None):
            return (pop_history, att_history), meta
        elif mil_history is not None:
            return (pop_history, mil_history), meta
        else:
            return pop_history, meta

    if record_attractiveness and (att_history is not None):
        return (pop_history, att_history)
    if mil_history is not None:
        return (pop_history, mil_history)
    return pop_history

# =========================================================
# Visualizer (biome base + population/attractiveness/military + totals)
# =========================================================
def visualize_population_with_totals(
    pop_history,
    meta=None,
    # ---- view mode knob ----
    view="biome_overlay",            # "biome_overlay" | "population" | "attractiveness" | "military"

    # ---- overlay knobs (biome_overlay) ----
    overlay_metric="population",     # "population" or "military"
    highlight_frac=0.50,
    highlight_basis="start_max",     # "start_max" | "global_max" | "frame_max"
    overlay_alpha_gamma=0.6,
    overlay_alpha_min=0.0,
    overlay_alpha_max=1.0,
    overlay_cmap="magma",
    overlay_military_cmap="coolwarm",
    overlay_cluster_threshold=2500.0,
    overlay_use_culture=False,
    overlay_military_color_window=(-0.5, 0.5),

    # population-only knobs
    pop_cmap="magma",
    pop_log=False,
    pop_vmax=None,
    pop_vmax_mode="global",

    # attractiveness view knobs
    att_history=None,
    att_cmap="viridis",
    att_log=False,
    att_vmax=None,
    att_vmax_mode="global",

    # military full-view knobs
    military_cmap="coolwarm",
    military_color_window=(-0.5, 0.5),
    cluster_threshold=2500.0,
    cluster_use_culture=False,

    # inputs from sim
    military_cell_history=None,

    # COUNTRY BOUNDARIES
    country_overlay=False,
    country_base_radius=2.0,
    country_scale=12.0,
    country_gamma=1.0,
    country_min_stance=0.05,
    country_edge_alpha=0.95,
    country_edge_color=None,

    # animation & layout
    fps=12,
    title_prefix="Population — Year ",
    save_path=None,
    show=True,
    first_year=0,
    step=1,
    last_frame_png_path=None,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from scipy.ndimage import label, distance_transform_edt

    frames = len(pop_history)
    H, W = pop_history[0].shape
    totals = np.array([f.sum() for f in pop_history], dtype=np.float64)

    fig = plt.figure(figsize=(9, 7))
    gs  = fig.add_gridspec(3, 1, height_ratios=[3.0, 0.15, 1.0])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[2, 0])
    ttl = ax_top.set_title(f"{title_prefix}{first_year}")
    ax_top.set_xticks([]); ax_top.set_yticks([])

    def to_log(arr, flag):
        return np.log1p(arr) if flag else arr

    def _stance_map(pop, military_cell, water_mask, thr, use_culture, culture_anchor=None):
        if use_culture and (culture_anchor is not None):
            mask = (culture_anchor.astype(bool)) & (~water_mask)
        else:
            mask = (pop >= float(thr)) & (~water_mask)
        lbl, nlab = label(mask.astype(np.int8), structure=np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.int8))
        if nlab == 0:
            out = np.zeros_like(pop, dtype=np.float32)
            out[water_mask] = 0.0
            return out, lbl, nlab, mask
        lab_flat = lbl.ravel(); valid = lab_flat > 0; idx = lab_flat[valid].astype(np.int64)
        pop_flat = pop.ravel()[valid].astype(np.float64)
        m_flat   = military_cell.ravel()[valid].astype(np.float64)
        sum_pop = np.bincount(idx, weights=pop_flat, minlength=nlab + 1)
        sum_m   = np.bincount(idx, weights=pop_flat * m_flat, minlength=nlab + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            means = sum_m[1:] / np.maximum(sum_pop[1:], 1e-12)
        means = np.clip(means, -1.0, 1.0).astype(np.float32)
        means_pad = np.zeros(nlab + 1, dtype=np.float32); means_pad[1:] = means
        stance_map = means_pad[lbl]
        stance_map[water_mask] = 0.0
        return stance_map, lbl, nlab, mask

    def _alpha_from_strength(strength):
        strength = np.clip(strength, 0.0, 1.0)
        return overlay_alpha_min + (overlay_alpha_max - overlay_alpha_min) * (strength ** overlay_alpha_gamma)

    def _apply_color_window(values, window):
        vmin, vmax = window
        if vmax <= vmin: vmax = vmin + 1e-6
        normed = (values - vmin) / (vmax - vmin)
        return np.clip(normed, 0.0, 1.0)

    def _military_cell(i):
        if military_cell_history is not None:
            return military_cell_history[i].astype(np.float32)
        if (meta is not None) and ("military_trait" in meta) and (meta["military_trait"] is not None):
            return meta["military_trait"].astype(np.float32)
        return np.zeros_like(pop_history[0], dtype=np.float32)

    def _country_labels(stance_map, lbl, nlab, mask, water_mask):
        if nlab == 0:
            return np.zeros_like(lbl, dtype=np.int32)
        lab_flat = lbl.ravel(); valid = lab_flat > 0; idx = lab_flat[valid].astype(np.int64)
        s_flat = stance_map.ravel()[valid].astype(np.float64)
        sum_abs = np.bincount(idx, weights=np.abs(s_flat), minlength=nlab + 1)
        cnt     = np.bincount(idx, minlength=nlab + 1).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_abs = sum_abs[1:] / np.maximum(cnt[1:], 1e-12)
        mean_abs = np.nan_to_num(mean_abs, nan=0.0).astype(np.float32)
        mean_abs[mean_abs < float(country_min_stance)] = 0.0
        radii = country_base_radius + country_scale * (mean_abs ** country_gamma)
        radii = np.maximum(0.0, radii).astype(np.float32)
        inv = ~mask
        dist, ind = distance_transform_edt(inv, return_indices=True)
        ny, nx = ind[0], ind[1]
        nearest_lbl = lbl[ny, nx]
        radii_pad = np.zeros(nlab + 1, dtype=np.float32)
        radii_pad[1:] = radii
        R = radii_pad[nearest_lbl]
        inside = (nearest_lbl > 0) & (dist <= R) & (~water_mask)
        country_lbl = np.where(inside, nearest_lbl, 0).astype(np.int32)
        return country_lbl

    culture_anchor = meta.get("culture_anchor") if meta is not None else None
    water_mask = meta.get("water_mask") if meta is not None else np.zeros_like(pop_history[0], dtype=bool)

    if view == "biome_overlay":
        if meta is None or "biome" not in meta:
            raise ValueError("meta with 'biome' is required for biome-colored base when view='biome_overlay'.")
        biome = meta["biome"].astype(np.int16)
        biome_colors = np.array([
            [ 54,  90, 154],
            [237, 201, 175],
            [120, 182,  96],
            [ 52, 120,  73],
            [130, 130, 130],
            [240, 245, 250],
        ], dtype=np.float32) / 255.0
        biome_cmap = ListedColormap(biome_colors)
        biome_norm = BoundaryNorm(boundaries=[-0.5,0.5,1.5,2.5,3.5,4.5,5.5], ncolors=biome_cmap.N)
        ax_top.imshow(biome, cmap=biome_cmap, norm=biome_norm, origin="upper")

        def make_overlay(i):
            pop = pop_history[i].astype(np.float32)
            if overlay_metric == "population":
                basis = float(max(np.max(f) for f in pop_history))
                thr = highlight_frac * (basis if basis > 0 else 1.0)
                rngv = max(1e-9, (basis - thr))
                strength = np.clip((pop - thr) / rngv, 0.0, 1.0)
                alpha = _alpha_from_strength(strength)
                cmap = plt.get_cmap(overlay_cmap)
                rgba = cmap(np.clip(pop / max(1e-9, basis), 0, 1))
                rgba[..., 3] = alpha
                return rgba
            else:
                mcell = _military_cell(i)
                stance, lbl, nlab, mask = _stance_map(pop, mcell, water_mask,
                                                      overlay_cluster_threshold, overlay_use_culture, culture_anchor)
                basis = float(np.max(np.abs(stance)))
                basis = basis if basis > 0 else 1.0
                thr = highlight_frac * basis
                rngv = max(1e-9, (basis - thr))
                strength = np.clip((np.abs(stance) - thr) / rngv, 0.0, 1.0)
                alpha = _alpha_from_strength(strength)
                cmap = plt.get_cmap(overlay_military_cmap)
                normed = _apply_color_window(stance, overlay_military_color_window)
                rgba = cmap(normed)
                rgba[..., 3] = alpha
                if country_overlay:
                    country_lbl = _country_labels(stance, lbl, nlab, mask, water_mask)
                    L = country_lbl
                    land = ~water_mask
                    nz = (L > 0) & land
                    neighbor_diff = (
                        (L != np.roll(L, 1, axis=0)) |
                        (L != np.roll(L,-1, axis=0)) |
                        (L != np.roll(L, 1, axis=1)) |
                        (L != np.roll(L,-1, axis=1))
                    )
                    coast_touch = (
                        np.roll(water_mask, 1, axis=0) |
                        np.roll(water_mask,-1, axis=0) |
                        np.roll(water_mask, 1, axis=1) |
                        np.roll(water_mask,-1, axis=1)
                    )
                    edge = nz & (neighbor_diff | coast_touch)
                    rgba[edge, :3] = 0.0
                    rgba[edge,  3] = country_edge_alpha
                return rgba

        im_top = ax_top.imshow(make_overlay(0), origin="upper")
        def update_top(i):
            im_top.set_data(make_overlay(i))
            return (im_top,)
        update_top(0)

    elif view == "population":
        pop0 = to_log(pop_history[0].astype(np.float32), pop_log)
        vmax0 = pop_vmax if pop_vmax is not None else (
            float(max(np.max(to_log(f.astype(np.float32), pop_log)) for f in pop_history))
            if pop_vmax_mode == "global" else float(pop0.max())
        )
        im_top = ax_top.imshow(pop0, cmap=pop_cmap, vmin=0.0, vmax=max(1e-9, vmax0), origin="upper")
        cb = plt.colorbar(im_top, ax=ax_top, fraction=0.046, pad=0.02)
        cb.set_label("log1p(population)" if pop_log else "population per cell")
        def update_top(i):
            fld = to_log(pop_history[i].astype(np.float32), pop_log)
            if pop_vmax is None and pop_vmax_mode == "frame":
                im_top.set_clim(0.0, max(1e-9, float(fld.max())))
            im_top.set_data(fld)
            return (im_top,)
        update_top(0)

    elif view == "attractiveness":
        if att_history is None:
            raise ValueError("att_history is required when view='attractiveness'.")
        att0 = to_log(att_history[0].astype(np.float32), att_log)
        vmax0 = att_vmax if att_vmax is not None else (
            float(max(np.max(to_log(a.astype(np.float32), att_log)) for a in att_history))
            if att_vmax_mode == "global" else float(att0.max())
        )
        im_top = ax_top.imshow(att0, cmap=att_cmap, vmin=0.0, vmax=max(1e-9, vmax0), origin="upper")
        cb = plt.colorbar(im_top, ax=ax_top, fraction=0.046, pad=0.02)
        cb.set_label("log1p(attractiveness)" if att_log else "attractiveness")
        def update_top(i):
            fld = to_log(att_history[i].astype(np.float32), att_log)
            if att_vmax is None and att_vmax_mode == "frame":
                im_top.set_clim(0.0, max(1e-9, float(fld.max())))
            im_top.set_data(fld)
            return (im_top,)
        update_top(0)

    else:
        raise ValueError("view must be 'biome_overlay' or 'population' or 'attractiveness' or 'military'.")

    x = np.arange(frames) * step + first_year
    (line,) = ax_bot.plot(x[:1], totals[:1])
    ax_bot.set_xlim(x[0], x[-1])
    ax_bot.set_ylim(0, totals.max() * 1.05 if totals.max() > 0 else 1.0)
    ax_bot.set_xlabel("Year"); ax_bot.set_ylabel("Total population")
    vline = ax_bot.axvline(x[0], linestyle="--", alpha=0.7)

    def update(i):
        artists = update_top(i)
        ttl.set_text(f"{title_prefix}{x[i]}")
        line.set_data(x[:i+1], totals[:i+1])
        vline.set_xdata([x[i], x[i]])
        return (*artists, ttl, line, vline)

    anim = FuncAnimation(fig, update, frames=frames, interval=1000/max(1, fps), blit=False)
    if last_frame_png_path is not None:
        update(frames - 1)
        fig.canvas.draw_idle()
        fig.savefig(last_frame_png_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    if save_path is not None:
        anim.save(save_path, dpi=140)
    plt.close(fig)
    return anim


# PLOTLY VIZ (unchanged helpers keep working with meta artifacts)
from scipy.ndimage import label, distance_transform_edt

def _stance_map(pop, military_cell, water_mask, thr, use_culture, culture_anchor=None):
    if use_culture and (culture_anchor is not None):
        mask = (culture_anchor.astype(bool)) & (~water_mask)
    else:
        mask = (pop >= float(thr)) & (~water_mask)
    lbl, nlab = label(mask.astype(np.int8),
                      structure=np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.int8))
    if nlab == 0:
        out = np.zeros_like(pop, dtype=np.float32)
        out[water_mask] = 0.0
        return out, lbl, nlab, mask
    lab_flat = lbl.ravel()
    valid = lab_flat > 0
    idx = lab_flat[valid].astype(np.int64)
    pop_flat = pop.ravel()[valid].astype(np.float64)
    m_flat   = military_cell.ravel()[valid].astype(np.float64)
    sum_pop = np.bincount(idx, weights=pop_flat, minlength=nlab + 1)
    sum_m   = np.bincount(idx, weights=pop_flat * m_flat, minlength=nlab + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = sum_m[1:] / np.maximum(sum_pop[1:], 1e-12)
    means = np.clip(means, -1.0, 1.0).astype(np.float32)
    means_pad = np.zeros(nlab + 1, dtype=np.float32); means_pad[1:] = means
    stance_map = means_pad[lbl]
    stance_map[water_mask] = 0.0
    return stance_map, lbl, nlab, mask

def _country_labels_clip_land(stance_map, lbl, nlab, mask, water_mask,
                              country_base_radius=2.0, country_scale=12.0,
                              country_gamma=1.0, country_min_stance=0.05):
    if nlab == 0:
        return np.zeros_like(lbl, dtype=np.int32)
    lab_flat = lbl.ravel()
    valid = lab_flat > 0
    idx = lab_flat[valid].astype(np.int64)
    s_flat = stance_map.ravel()[valid].astype(np.float64)
    sum_abs = np.bincount(idx, weights=np.abs(s_flat), minlength=nlab + 1)
    cnt     = np.bincount(idx, minlength=nlab + 1).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_abs = sum_abs[1:] / np.maximum(cnt[1:], 1e-12)
    mean_abs = np.nan_to_num(mean_abs, nan=0.0).astype(np.float32)
    mean_abs[mean_abs < float(country_min_stance)] = 0.0
    radii = country_base_radius + country_scale * (mean_abs ** country_gamma)
    radii = np.maximum(0.0, radii).astype(np.float32)
    inv = ~mask
    dist, ind = distance_transform_edt(inv, return_indices=True)
    ny, nx = ind[0], ind[1]
    nearest_lbl = lbl[ny, nx]
    radii_pad = np.zeros(nlab + 1, dtype=np.float32); radii_pad[1:] = radii
    R = radii_pad[nearest_lbl]
    inside = (nearest_lbl > 0) & (dist <= R) & (~water_mask)
    country_lbl = np.where(inside, nearest_lbl, 0).astype(np.int32)
    return country_lbl

def _country_edges_black(country_lbl, water_mask):
    L = country_lbl
    land = ~water_mask
    nz = (L > 0) & land
    neighbor_diff = (
        (L != np.roll(L, 1, axis=0)) |
        (L != np.roll(L,-1, axis=0)) |
        (L != np.roll(L, 1, axis=1)) |
        (L != np.roll(L,-1, axis=1))
    )
    coast_touch = (
        np.roll(water_mask, 1, axis=0) |
        np.roll(water_mask,-1, axis=0) |
        np.roll(water_mask, 1, axis=1) |
        np.roll(water_mask,-1, axis=1)
    )
    edge = nz & (neighbor_diff | coast_touch)
    return edge

def _country_capitals(pop, country_lbl):
    H, W = pop.shape
    L = country_lbl
    labels = np.unique(L[L > 0])
    if labels.size == 0:
        return np.array([]), np.array([]), np.array([])
    yy, xx = np.indices((H, W))
    Lf = L.ravel()
    Pf = pop.ravel().astype(np.float64)
    yf = yy.ravel().astype(np.float64)
    xf = xx.ravel().astype(np.float64)
    max_lab = int(labels.max())
    sumP   = np.bincount(Lf, weights=Pf, minlength=max_lab+1)
    sumPy  = np.bincount(Lf, weights=Pf * yf, minlength=max_lab+1)
    sumPx  = np.bincount(Lf, weights=Pf * xf, minlength=max_lab+1)
    valid = (labels[sumP[labels] > 0])
    if valid.size == 0:
        return np.array([]), np.array([]), np.array([])
    cy = (sumPy[valid] / sumP[valid])
    cx = (sumPx[valid] / sumP[valid])
    return cy.astype(np.float32), cx.astype(np.float32), valid.astype(np.int32)

def visualize_plotly_countries(
    pop_history,
    meta,
    att_history=None,
    military_cell_history=None,
    pop_highlight_frac=0.10,
    att_highlight_frac=0.10,
    mil_highlight_frac=0.10,
    overlay_cluster_threshold=2500.0,
    overlay_use_culture=False,
    overlay_military_color_window=(-0.5, 0.5),
    country_overlay=True,
    country_base_radius=2.0,
    country_scale=12.0,
    country_gamma=1.0,
    country_min_stance=0.05,
    country_edge_alpha=0.95,
    show_population=True,
    show_attractiveness=False,
    show_military=True,
    show_borders=True,
    show_capitals=True,
    title="Biomes + Overlays (Plotly)",
    height=700,
):
    import numpy as np
    import plotly.graph_objects as go

    def _stance_map_local(pop, military_cell, water_mask, thr, use_culture, culture_anchor=None):
        if use_culture and (culture_anchor is not None):
            mask = (culture_anchor.astype(bool)) & (~water_mask)
        else:
            mask = (pop >= float(thr)) & (~water_mask)
        lbl_, nlab_ = label(mask.astype(np.int8),
                            structure=np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.int8))
        if nlab_ == 0:
            out = np.zeros_like(pop, dtype=np.float32)
            out[water_mask] = 0.0
            return out, lbl_, nlab_, mask
        lab_flat = lbl_.ravel(); valid = lab_flat > 0; idx = lab_flat[valid].astype(np.int64)
        pop_flat = pop.ravel()[valid].astype(np.float64)
        m_flat   = military_cell.ravel()[valid].astype(np.float64)
        sum_pop = np.bincount(idx, weights=pop_flat, minlength=nlab_ + 1)
        sum_m   = np.bincount(idx, weights=pop_flat * m_flat, minlength=nlab_ + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            means = sum_m[1:] / np.maximum(sum_pop[1:], 1e-12)
        means = np.clip(means, -1.0, 1.0).astype(np.float32)
        means_pad = np.zeros(nlab_ + 1, dtype=np.float32); means_pad[1:] = means
        stance_map = means_pad[lbl_]
        stance_map[water_mask] = 0.0
        return stance_map, lbl_, nlab_, mask

    def _country_labels_clip_land_local(stance_map, lbl_, nlab_, mask, water_mask,
                                        base_radius, scale, gamma, min_stance):
        if nlab_ == 0: return np.zeros_like(lbl_, dtype=np.int32)
        lab_flat = lbl_.ravel(); valid = lab_flat > 0; idx = lab_flat[valid].astype(np.int64)
        s_flat = stance_map.ravel()[valid].astype(np.float64)
        sum_abs = np.bincount(idx, weights=np.abs(s_flat), minlength=nlab_ + 1)
        cnt     = np.bincount(idx, minlength=nlab_ + 1).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_abs = sum_abs[1:] / np.maximum(cnt[1:], 1e-12)
        mean_abs = np.nan_to_num(mean_abs, nan=0.0).astype(np.float32)
        mean_abs[mean_abs < float(min_stance)] = 0.0
        radii = base_radius + scale * (mean_abs ** gamma)
        radii = np.maximum(0.0, radii).astype(np.float32)
        inv = ~mask
        dist, ind = distance_transform_edt(inv, return_indices=True)
        ny, nx = ind[0], ind[1]
        nearest_lbl = lbl_[ny, nx]
        radii_pad = np.zeros(nlab_ + 1, dtype=np.float32); radii_pad[1:] = radii
        R = radii_pad[nearest_lbl]
        return np.where((nearest_lbl > 0) & (dist <= R) & (~water_mask), nearest_lbl, 0).astype(np.int32)

    def _country_edges_black_local(country_lbl, water_mask):
        L = country_lbl; land = ~water_mask; nz = (L > 0) & land
        neighbor_diff = (
            (L != np.roll(L, 1, axis=0)) |
            (L != np.roll(L,-1, axis=0)) |
            (L != np.roll(L, 1, axis=1)) |
            (L != np.roll(L,-1, axis=1))
        )
        coast_touch = (
            np.roll(water_mask, 1, axis=0) |
            np.roll(water_mask,-1, axis=0) |
            np.roll(water_mask, 1, axis=1) |
            np.roll(water_mask,-1, axis=1)
        )
        return nz & (neighbor_diff | coast_touch)

    def _country_capitals_local(pop, country_lbl):
        H, W = pop.shape; L = country_lbl
        labels = np.unique(L[L > 0])
        if labels.size == 0: return np.array([]), np.array([]), np.array([])
        yy, xx = np.indices((H, W))
        Lf = L.ravel(); Pf = pop.ravel().astype(np.float64)
        yf = yy.ravel().astype(np.float64); xf = xx.ravel().astype(np.float64)
        max_lab = int(labels.max())
        sumP  = np.bincount(Lf, weights=Pf, minlength=max_lab+1)
        sumPy = np.bincount(Lf, weights=Pf * yf, minlength=max_lab+1)
        sumPx = np.bincount(Lf, weights=Pf * xf, minlength=max_lab+1)
        valid_l = labels[sumP[labels] > 0]
        if valid_l.size == 0: return np.array([]), np.array([]), np.array([])
        cy = (sumPy[valid_l] / sumP[valid_l]); cx = (sumPx[valid_l] / sumP[valid_l])
        return cy.astype(np.float32), cx.astype(np.float32), valid_l.astype(np.int32)

    H, W = pop_history[0].shape
    T = len(pop_history)
    biome = meta["biome"].astype(np.int16)
    water_mask = meta["water_mask"].astype(bool)
    culture_anchor = meta.get("culture_anchor")

    biome_colors = np.array([
        [ 54,  90, 154],
        [237, 201, 175],
        [120, 182,  96],
        [ 52, 120,  73],
        [130, 130, 130],
        [240, 245, 250],
    ], dtype=np.float32) / 255.0
    biome_img = biome_colors[biome]
    pop_colorscale = "Magma"
    att_colorscale = "Viridis"
    mil_colorscale = "RdBu"

    pop_global_max = float(max(np.max(f) for f in pop_history))
    att_global_max = float(max(np.max(a) for a in att_history)) if (att_history is not None) else 1.0

    def _mcell(i):
        if military_cell_history is not None:
            return military_cell_history[i].astype(np.float32)
        if ("military_trait" in meta) and (meta["military_trait"] is not None):
            return meta["military_trait"].astype(np.float32)
        return np.zeros_like(pop_history[0], dtype=np.float32)

    sim_countries = meta.get("countries_history")
    sim_capitals  = meta.get("capitals_history")

    pop_zs, att_zs, mil_zs, borders, capitals = [], [], [], [], []
    mil_vmin, mil_vmax = overlay_military_color_window

    for i in range(T):
        P = pop_history[i].astype(np.float32)
        p_basis = pop_global_max if pop_global_max > 0 else float(P.max())
        p_thr = pop_highlight_frac * (p_basis if p_basis > 0 else 1.0)
        P_masked = P.copy().astype(np.float32); P_masked[P_masked < p_thr] = np.nan
        pop_zs.append(np.asarray(P_masked))

        if att_history is not None:
            A = att_history[i].astype(np.float32)
            a_basis = att_global_max if att_global_max > 0 else float(A.max())
            a_thr = att_highlight_frac * (a_basis if a_basis > 0 else 1.0)
            A_masked = A.copy().astype(np.float32); A_masked[A_masked < a_thr] = np.nan
            att_zs.append(np.asarray(A_masked))
        else:
            att_zs.append(np.full((H, W), np.nan, dtype=float))

        M = _mcell(i)
        stance_map, lbl, nlab, mask = _stance_map_local(
            pop=P, military_cell=M, water_mask=water_mask,
            thr=overlay_cluster_threshold, use_culture=overlay_use_culture,
            culture_anchor=culture_anchor
        )
        m_abs_max = float(np.max(np.abs(stance_map))); m_abs_max = (m_abs_max if m_abs_max > 0 else 1.0)
        m_thr = mil_highlight_frac * m_abs_max
        M_masked = stance_map.copy().astype(np.float32); M_masked[np.abs(M_masked) < m_thr] = np.nan
        M_clip = np.clip(M_masked, mil_vmin, mil_vmax)
        mil_zs.append(np.asarray(M_clip))

        if sim_countries is not None and sim_capitals is not None and i < len(sim_countries) and i < len(sim_capitals):
            c_lbl = sim_countries[i].astype(np.int32)
            caps_i = sim_capitals[i]
            cy, cx = caps_i["cy"], caps_i["cx"]
        else:
            c_lbl = _country_labels_clip_land_local(
                stance_map, lbl, nlab, mask, water_mask,
                base_radius=country_base_radius,
                scale=country_scale,
                gamma=country_gamma,
                min_stance=country_min_stance
            )
            cy, cx, _ = _country_capitals_local(P, c_lbl)

        edge = _country_edges_black_local(c_lbl, water_mask)
        ey, ex = np.nonzero(edge)
        borders.append((ey.astype(np.float32), ex.astype(np.float32)))
        if cy.size == 0:
            capitals.append((np.array([np.nan], dtype=float), np.array([np.nan], dtype=float)))
        else:
            capitals.append((cy.astype(float), cx.astype(float)))

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Image(z=(biome_img * 255).astype(np.uint8), name="Biomes", visible=True))
    fig.add_trace(go.Heatmap(z=pop_zs[0], colorscale=pop_colorscale, zmin=0, zmax=max(1e-9, pop_global_max),
                             showscale=False, name="Population", visible=bool(show_population),
                             hoverinfo="skip", opacity=1.0))
    fig.add_trace(go.Heatmap(z=att_zs[0], colorscale=att_colorscale, zmin=0, zmax=max(1e-9, att_global_max),
                             showscale=False, name="Attractiveness", visible=bool(show_attractiveness),
                             hoverinfo="skip", opacity=1.0))
    fig.add_trace(go.Heatmap(z=mil_zs[0], colorscale=mil_colorscale, zmin=mil_vmin, zmax=mil_vmax, zmid=0.0,
                             showscale=False, name="Military", visible=bool(show_military),
                             hoverinfo="skip", opacity=1.0))
    by0, bx0 = borders[0]
    fig.add_trace(go.Scatter(x=bx0, y=by0, mode="markers", name="Borders",
                             marker=dict(color="black", size=2, opacity=float(country_edge_alpha)),
                             visible=bool(show_borders and country_overlay), hoverinfo="skip"))
    cy0, cx0 = capitals[0]
    fig.add_trace(go.Scatter(x=cx0, y=cy0, mode="markers", name="Capitals",
                             marker=dict(symbol="circle", size=14, color="red", line=dict(width=1, color="black")),
                             visible=bool(show_capitals), hovertemplate="Capital<extra></extra>"))

    frames_list = []
    for i in range(T):
        by, bx = borders[i]
        cy, cx = capitals[i]
        frames_list.append(go.Frame(
            name=str(i),
            data=[
                go.Heatmap(z=pop_zs[i], colorscale=pop_colorscale, zmin=0, zmax=max(1e-9, pop_global_max),
                           showscale=False, hoverinfo="skip", opacity=1.0),
                go.Heatmap(z=att_zs[i], colorscale=att_colorscale, zmin=0, zmax=max(1e-9, att_global_max),
                           showscale=False, hoverinfo="skip", opacity=1.0),
                go.Heatmap(z=mil_zs[i], colorscale=mil_colorscale, zmin=mil_vmin, zmax=mil_vmax, zmid=0.0,
                           showscale=False, hoverinfo="skip", opacity=1.0),
                go.Scatter(x=bx, y=by, mode="markers",
                           marker=dict(color="black", size=2, opacity=float(country_edge_alpha)),
                           hoverinfo="skip", name="Borders"),
                go.Scatter(x=cx, y=cy, mode="markers",
                           marker=dict(symbol="circle", size=14, color="red", line=dict(width=1, color="black")),
                           hovertemplate="Capital<extra></extra>", name="Capitals"),
            ],
            traces=[1, 2, 3, 4, 5]
        ))
    fig.frames = frames_list

    steps = [dict(method="animate",
                  args=[[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
                  label=str(i)) for i in range(T)]
    sliders = [dict(active=0, steps=steps, x=0.05, y=0.02, xanchor="left", yanchor="bottom",
                    currentvalue=dict(prefix="Year: ", visible=True))]

    def _menu(label, trace_idx, x, y):
        return dict(type="buttons", direction="right", x=x, y=y, xanchor="left", yanchor="top", showactive=True,
                    buttons=[dict(label=f"{label}: On", method="restyle", args=[{"visible": True}, [trace_idx]]),
                             dict(label=f"{label}: Off", method="restyle", args=[{"visible": False}, [trace_idx]])])

    updatemenus = [
        _menu("Population",     1, 0.01, 1.04),
        _menu("Attractiveness", 2, 0.27, 1.04),
        _menu("Military",       3, 0.57, 1.04),
        _menu("Borders",        4, 0.77, 1.04),
        _menu("Capitals",       5, 0.90, 1.04),
    ]

    fig.update_xaxes(showticklabels=False, visible=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(showticklabels=False, visible=False, autorange="reversed")
    fig.update_layout(title=title, height=700, margin=dict(l=10, r=10, t=90, b=30),
                      showlegend=False, sliders=sliders, updatemenus=updatemenus)
    fig.data[0].visible = True
    fig.data[1].visible = bool(show_population)
    fig.data[2].visible = bool(show_attractiveness)
    fig.data[3].visible = bool(show_military)
    fig.data[4].visible = bool(show_borders and country_overlay)
    fig.data[5].visible = bool(show_capitals)
    return fig

# Function to display pice charts of demographics in an html report using plotly
def generate_demographics_report(meta, output_html_path="demographics_report.html"):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    demographics_last = meta.get("demographics_last")
    country_names_last = meta.get("country_names_last") or {}
    capital_names_last = meta.get("capital_names_last") or {}

    if demographics_last is None:
        raise ValueError("No demographics data found in meta.")

    countries = list(demographics_last.keys())
    n_countries = len(countries)
    n_cols = 3
    n_rows = (n_countries + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=[f"{country_names_last.get(c, 'Country '+str(c))} (Capital: {capital_names_last.get(c, 'N/A')})"
                                        for c in countries])

    for idx, country in enumerate(countries):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1
        demo_data = demographics_last[country]
        labels = list(demo_data.keys())
        values = [demo_data[label] for label in labels]

        fig.add_trace(go.Pie(labels=labels, values=values, hole=0.3), row=row, col=col)

    fig.update_layout(height=300 * n_rows, width=900, title_text="Demographics Distribution by Country")

    fig.write_html(output_html_path)


# =========================================================
# Demo (includes demographics spec, naming, and final printed report)
# =========================================================
if __name__ == "__main__":
    H, W = 128, 128
    rng = np.random.default_rng()
    z = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), sigma=12.0, mode="reflect")
    z = (z - z.mean()) / (z.std() + 1e-6)
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    heightmap_xyz = np.column_stack([xs.ravel(), ys.ravel(), z.ravel()])


    # --- run sim; record attractiveness, military, countries/capitals, demographics, and generate names
    ((pops, atts), mils), meta = simulate_population_from_heightmap(
        heightmap_xyz,
        sea_level=0.0,
        years=500,
        initial_total_population=1_600_000, expected_settlements=180,
        # flows
        m_dir=0.05,
        m_gravity=0.035, grav_sigma=7.0,
        m_sprawl=0.022, urban_k_frac=0.58,
        m_explore=0.004, m_diff=0.009,
        # infra & robustness
        infra_sigma=2.0, infra_threshold=9000.0, infra_strength=0.25,
        pref_noise_sigma=0.10, crowding_penalty=0.70,
        # Life
        life_enabled=True, life_rule="B3/S23",
        life_birth_boost=0.26, life_death_suppression=0.15, life_capacity_boost=0.22, life_migration_bias=0.22,
        # Catastrophes
        catastrophe_prob=0.03, catastrophe_events_mean=1.1, disk_radius_range=(3,7),
        # Culture
        culture_enabled=True, town_threshold=2600.0, culture_radius=1, culture_persist_years=20,
        culture_min_abs=900.0, culture_min_frac=0.06,
        # Sand corridor & stronger water pull
        alpha_w=1.6, water_affinity_gain=0.35, sand_corridor_boost=0.6,
        # Exodus
        exodus_enabled=True, exodus_prob=0.07, exodus_events_mean=1.2,
        exodus_group_frac=0.22, exodus_min_dist=50,
        exodus_max_dist=None, exodus_target_bias_water=0.0,
        exodus_target_bias_empty=0.8, exodus_culture_factor=0.5,
        # Military forcing
        military_enabled=True,
        record_military=True,
        military_forcing_enabled=True,
        military_neutral_band=0.10,
        military_period=24,
        military_strength=0.6,
        military_shape="sin",
        # Countries & capitals
        record_countries=True,
        overlay_cluster_threshold=2600.0,
        overlay_use_culture=False,
        country_base_radius=2.0,
        country_scale=14.0,
        country_gamma=1.25,
        country_min_stance=0.08,
        # DEMOGRAPHICS
        demographics_spec=RACE_SPECS,
        record_demographics=True,
        rarity_exponent=1.0,
        demographics_sigma=0.45,
        demographics_power=2.0,
        # NAMING
        naming_enabled=True,
        markov_order=3,
        country_name_len_range=(7, 14),
        city_name_len_range=(5, 12),
        naming_seed_offset=101,
        # recording & performance
        save_every=1, dtype=np.float32, return_meta=True, record_attractiveness=True
    )

    print("Frames:", len(pops), "Grid:", pops[0].shape, "Start:", int(pops[0].sum()), "End:", int(pops[-1].sum()))

    # 1) Biome + population overlay with last-frame png
    visualize_population_with_totals(
        pops, meta,
        view="biome_overlay",
        overlay_metric="population",
        highlight_frac=0.1,
        fps=12,
        last_frame_png_path="population_last.png"
    )

    # 2) Plotly viz with toggles
    #fig = visualize_plotly_countries(
    #    pop_history=pops,
    #    meta=meta,
    #    att_history=atts,
    #    military_cell_history=mils,
    #    overlay_cluster_threshold=2600.0,
    #    overlay_military_color_window=(-0.5, 0.5),
    #    country_overlay=True,
    #    country_base_radius=2.0,
    #    country_scale=14.0,
    #    country_gamma=1.25,
    #    country_min_stance=0.08,
    #    title="Biomes + Overlays (Plotly)"
    #)
    #fig.write_html("plotly_countries.html")

    # ===== Pretty print: final named countries, capitals, demographics (≥0.01%), pop and military =====
    caps = meta.get("capitals_last")
    demos = meta.get("demographics_last")
    ctry_names = meta.get("country_names_last") or {}
    cap_names  = meta.get("capital_names_last") or {}
    ctry_races = meta.get("country_race_last") or {}
    stats      = meta.get("country_stats_last") or {}

    if caps is not None and demos is not None:
        cy, cx, labels = caps["cy"], caps["cx"], caps["label"]
        print("\nFinal Countries & Capital Cities")
        print("--------------------------------")
        for i in range(len(labels)):
            lab = int(labels[i])
            y = float(cy[i]); x = float(cx[i])
            cname = ctry_names.get(lab, f"Country-{lab}")
            kname = cap_names.get(lab,  f"Capital-{lab}")
            race  = ctry_races.get(lab, "unknown")
            st = stats.get(lab, {"total_population": 0.0, "mean_stance": 0.0, "strength": 0.0})
            pop_total = int(round(st["total_population"]))
            mean_stance = st["mean_stance"]
            strength = st["strength"]

            demo = demos.get(lab, {})
            shown_pairs = sorted([(k, v) for k, v in demo.items() if v > 0.0], key=lambda kv: -kv[1])
            demo_str = ", ".join([f"{k}: {v:.2f}%" for k, v in shown_pairs]) if shown_pairs else "(no groups ≥ 0.01%)"

            sign = "+" if mean_stance >= 0 else ""
            print(f" - [{lab:3d}] {cname} (race: {race}) — Capital: {kname} at (y={y:.1f}, x={x:.1f})")
            print(f"     Pop: {pop_total:,} | Military stance: {sign}{mean_stance:.2f} | Strength: {int(0.04*pop_total*strength):,}")
            print(f"     Demographics: {demo_str}")

            generate_demographics_report(meta)
    else:
        print("\n(No capitals or demographics recorded.)")
