"""
Hierarchy metrics for the curvature study (Paper 2).

Defines taxonomy trees for each dataset and computes:
  - Tree depth (longest root-to-leaf path)
  - Mean branching factor (avg children per internal node)
  - Number of internal nodes
  - Gromov delta-hyperbolicity (sampling-based)
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ===================================================================
# Taxonomy trees — {parent: [children]} adjacency lists
# ===================================================================

# CIFAR-100: exact 2-level hierarchy (20 superclasses × 5 classes)
CIFAR100_TREE: Dict[str, List[str]] = {
    "root": [
        "aquatic_mammals", "fish", "flowers", "food_containers",
        "fruit_and_vegetables", "household_electrical_devices",
        "household_furniture", "insects", "large_carnivores",
        "large_man-made_outdoor_things", "large_natural_outdoor_scenes",
        "large_omnivores_and_herbivores", "medium_mammals",
        "non-insect_invertebrates", "people", "reptiles",
        "small_mammals", "trees", "vehicles_1", "vehicles_2",
    ],
    "aquatic_mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food_containers": ["bottle", "bowl", "can", "cup", "plate"],
    "fruit_and_vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
    "household_electrical_devices": ["clock", "keyboard", "lamp", "telephone", "television"],
    "household_furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large_carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large_man-made_outdoor_things": ["bridge", "castle", "house", "road", "skyscraper"],
    "large_natural_outdoor_scenes": ["cloud", "forest", "mountain", "plain", "sea"],
    "large_omnivores_and_herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium_mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non-insect_invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
    "people": ["baby", "boy", "girl", "man", "woman"],
    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small_mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
    "vehicles_1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "vehicles_2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
}

# AwA2: animal taxonomy (Order → Family → Genus/common)
AWA2_TREE: Dict[str, List[str]] = {
    "root": ["carnivora", "artiodactyla", "perissodactyla", "cetacea",
             "rodentia", "chiroptera", "primates", "proboscidea",
             "lagomorpha", "soricomorpha"],
    "carnivora": ["canidae", "felidae", "ursidae", "mustelidae",
                  "mephitidae", "phocidae", "procyonidae"],
    "canidae": ["wolf", "fox", "german+shepherd", "chihuahua", "collie", "dalmatian"],
    "felidae": ["tiger", "leopard", "lion", "persian+cat", "siamese+cat", "bobcat"],
    "ursidae": ["grizzly+bear", "polar+bear", "giant+panda"],
    "mustelidae": ["weasel", "otter"],
    "mephitidae": ["skunk"],
    "phocidae": ["seal", "walrus"],
    "procyonidae": ["raccoon"],
    "artiodactyla": ["bovidae", "cervidae", "suidae", "giraffidae", "hippopotamidae"],
    "bovidae": ["antelope", "sheep", "ox", "buffalo", "cow"],
    "cervidae": ["deer", "moose"],
    "suidae": ["pig"],
    "giraffidae": ["giraffe"],
    "hippopotamidae": ["hippopotamus"],
    "perissodactyla": ["equidae", "rhinocerotidae"],
    "equidae": ["horse", "zebra"],
    "rhinocerotidae": ["rhinoceros"],
    "cetacea": ["blue+whale", "humpback+whale", "killer+whale", "dolphin"],
    "rodentia": ["sciuridae", "muridae", "cricetidae", "caviidae"],
    "sciuridae": ["squirrel"],
    "muridae": ["rat", "mouse"],
    "cricetidae": ["hamster"],
    "caviidae": ["beaver"],
    "chiroptera": ["bat"],
    "primates": ["gorilla", "chimpanzee", "spider+monkey"],
    "proboscidea": ["elephant"],
    "lagomorpha": ["rabbit"],
    "soricomorpha": ["mole"],
}

# CUB-200: ornithological hierarchy (Order → Family → Genus → Species)
# Approximate grouping of the 200 CUB classes into major bird orders/families
CUB200_TREE: Dict[str, List[str]] = {
    "root": ["passeriformes", "piciformes", "accipitriformes",
             "pelecaniformes", "charadriiformes", "anseriformes",
             "galliformes", "procellariiformes", "suliformes",
             "cuculiformes", "columbiformes", "apodiformes",
             "coraciiformes", "gruiformes", "podicipediformes"],
    "passeriformes": [  # ~120 species in CUB
        "corvidae", "parulidae", "fringillidae", "icteridae",
        "tyrannidae", "vireonidae", "troglodytidae", "paridae",
        "turdidae", "mimidae", "cardinalidae", "emberizidae",
        "hirundinidae", "muscicapidae",
    ],
    "corvidae": ["blue_jay", "florida_jay", "green_jay", "clark_nutcracker",
                 "common_raven", "fish_crow", "white_necked_raven"],
    "parulidae": ["black_and_white_warbler", "prothonotary_warbler",
                  "worm_eating_warbler", "orange_crowned_warbler",
                  "yellow_warbler", "wilson_warbler", "canada_warbler",
                  "kentucky_warbler", "mourning_warbler", "hooded_warbler",
                  "cape_may_warbler", "cerulean_warbler", "myrtle_warbler",
                  "bay_breasted_warbler", "blackburnian_warbler",
                  "pine_warbler", "prairie_warbler", "palm_warbler",
                  "tennessee_warbler", "nashville_warbler"],
    "fringillidae": ["house_finch", "purple_finch", "evening_grosbeak",
                     "pine_grosbeak", "american_goldfinch"],
    "icteridae": ["baltimore_oriole", "orchard_oriole", "scott_oriole",
                  "hooded_oriole", "bobolink", "red_winged_blackbird",
                  "brewer_blackbird", "rusty_blackbird", "western_meadowlark",
                  "shiny_cowbird", "bronzed_cowbird"],
    "tyrannidae": ["eastern_kingbird", "western_kingbird", "scissor_tailed_flycatcher",
                   "acadian_flycatcher", "great_crested_flycatcher",
                   "olive_sided_flycatcher", "vermilion_flycatcher",
                   "least_flycatcher", "sayornis"],
    "vireonidae": ["white_eyed_vireo", "red_eyed_vireo", "philadelphia_vireo",
                   "warbling_vireo", "yellow_throated_vireo"],
    "troglodytidae": ["cactus_wren", "carolina_wren", "house_wren",
                      "marsh_wren", "bewick_wren", "winter_wren", "rock_wren"],
    "paridae": ["black_capped_chickadee", "carolina_chickadee", "tufted_titmouse"],
    "turdidae": ["american_robin", "hermit_thrush", "wood_thrush",
                 "veery", "swainson_thrush"],
    "mimidae": ["mockingbird", "gray_catbird", "brown_thrasher", "sage_thrasher"],
    "cardinalidae": ["northern_cardinal", "rose_breasted_grosbeak",
                     "blue_grosbeak", "indigo_bunting", "lazuli_bunting",
                     "painted_bunting", "summer_tanager", "scarlet_tanager"],
    "emberizidae": ["chipping_sparrow", "field_sparrow", "fox_sparrow",
                    "harris_sparrow", "henslow_sparrow", "le_conte_sparrow",
                    "lincoln_sparrow", "nelson_sparrow", "savannah_sparrow",
                    "seaside_sparrow", "song_sparrow", "tree_sparrow",
                    "vesper_sparrow", "white_crowned_sparrow",
                    "white_throated_sparrow", "grasshopper_sparrow",
                    "clay_colored_sparrow", "dark_eyed_junco",
                    "eastern_towhee", "spotted_towhee", "green_tailed_towhee"],
    "hirundinidae": ["tree_swallow", "barn_swallow", "cliff_swallow",
                     "bank_swallow", "purple_martin"],
    "muscicapidae": ["european_goldfinch", "common_yellowthroat"],
    "piciformes": ["downy_woodpecker", "red_bellied_woodpecker",
                   "red_headed_woodpecker", "pileated_woodpecker",
                   "red_cockaded_woodpecker", "yellow_bellied_sapsucker"],
    "accipitriformes": ["bald_eagle", "red_tailed_hawk", "broad_winged_hawk",
                        "sharp_shinned_hawk", "coopers_hawk"],
    "pelecaniformes": ["white_pelican", "brown_pelican", "great_blue_heron",
                       "great_egret", "snowy_egret", "green_heron",
                       "yellow_crowned_night_heron", "least_bittern",
                       "american_bittern", "roseate_spoonbill"],
    "charadriiformes": ["herring_gull", "ring_billed_gull", "ivory_gull",
                        "slaty_backed_gull", "western_gull", "glaucous_winged_gull",
                        "heermann_gull", "california_gull", "bonaparte_gull",
                        "black_tern", "caspian_tern", "common_tern",
                        "elegant_tern", "forsters_tern", "least_tern",
                        "arctic_tern", "sooty_tern", "royal_tern",
                        "american_avocet", "black_necked_stilt",
                        "piping_plover", "killdeer", "spotted_sandpiper",
                        "whimbrel", "long_billed_curlew"],
    "anseriformes": ["mallard", "gadwall", "american_wigeon", "canvasback",
                     "ring_necked_duck", "red_breasted_merganser",
                     "hooded_merganser", "wood_duck", "green_winged_teal"],
    "galliformes": ["prairie_chicken", "ruffed_grouse", "sage_grouse"],
    "procellariiformes": ["black_footed_albatross", "laysan_albatross",
                          "sooty_albatross", "sooty_shearwater"],
    "suliformes": ["pelagic_cormorant", "brandt_cormorant",
                   "double_crested_cormorant", "magnificent_frigatebird",
                   "brown_booby", "masked_booby", "red_footed_booby"],
    "cuculiformes": ["yellow_billed_cuckoo", "mangrove_cuckoo",
                     "smooth_billed_ani", "groove_billed_ani"],
    "columbiformes": ["mourning_dove", "white_crowned_pigeon"],
    "apodiformes": ["ruby_throated_hummingbird", "anna_hummingbird",
                    "costa_hummingbird", "rufous_hummingbird",
                    "allen_hummingbird", "calliope_hummingbird",
                    "green_violetear", "rivoli_hummingbird"],
    "coraciiformes": ["belted_kingfisher", "green_kingfisher",
                      "ringed_kingfisher"],
    "gruiformes": ["american_coot", "common_gallinule", "purple_gallinule"],
    "podicipediformes": ["horned_grebe", "eared_grebe", "pied_billed_grebe",
                         "western_grebe", "clark_grebe"],
}

# SUN397: flat/shallow hierarchy (2 levels: domain → category)
SUN397_TREE: Dict[str, List[str]] = {
    "root": ["indoor", "outdoor_natural", "outdoor_man_made"],
    "indoor": ["residential", "commercial", "public", "workplace", "transportation"],
    "outdoor_natural": ["water", "land", "vegetation", "geological"],
    "outdoor_man_made": ["urban", "rural", "sports", "cultural"],
    # Each subcategory contains ~30 scene types (simplified)
    "residential": ["bedroom", "bathroom", "kitchen", "living_room", "dining_room",
                    "closet", "garage", "basement", "attic", "nursery"],
    "commercial": ["restaurant", "bar", "cafe", "supermarket", "shop",
                   "mall", "hotel_room", "lobby", "office", "bank"],
    "public": ["hospital", "church", "library", "museum", "courthouse",
               "prison", "school", "theater", "stadium", "arena"],
    "workplace": ["office", "laboratory", "factory", "warehouse", "workshop",
                  "studio", "server_room", "conference_room"],
    "transportation": ["airport", "train_station", "bus_station", "subway",
                       "parking_lot", "gas_station", "highway"],
    "water": ["beach", "ocean", "lake", "river", "waterfall", "pond",
              "swamp", "bay", "harbor"],
    "land": ["field", "meadow", "desert", "tundra", "prairie", "valley",
             "hill", "canyon", "cliff"],
    "vegetation": ["forest", "jungle", "garden", "park", "orchard",
                   "vineyard", "bamboo_forest"],
    "geological": ["mountain", "volcano", "cave", "glacier", "rock_arch",
                   "badlands", "butte"],
    "urban": ["street", "alley", "bridge", "plaza", "skyscraper",
              "apartment_building", "construction_site", "crosswalk"],
    "rural": ["farm", "barn", "cottage", "windmill", "silo",
              "hayfield", "pasture"],
    "sports": ["baseball_field", "basketball_court", "tennis_court",
               "golf_course", "ski_slope", "swimming_pool", "gymnasium"],
    "cultural": ["cemetery", "castle", "palace", "temple", "mosque",
                 "synagogue", "monument", "fountain"],
}

# Stanford Cars: shallow hierarchy (make → model)
STANFORD_CARS_TREE: Dict[str, List[str]] = {
    "root": ["american", "european", "asian"],
    "american": ["chrysler", "ford", "gmc", "chevrolet", "dodge",
                 "jeep", "lincoln", "cadillac", "buick", "tesla"],
    "european": ["bmw", "mercedes", "audi", "volkswagen", "porsche",
                 "ferrari", "lamborghini", "maserati", "bentley",
                 "aston_martin", "jaguar", "land_rover", "mini",
                 "fiat", "volvo", "saab", "rolls_royce"],
    "asian": ["toyota", "honda", "nissan", "mazda", "subaru",
              "mitsubishi", "suzuki", "hyundai", "kia", "infiniti",
              "acura", "lexus"],
    # Each make has multiple models (the 196 leaf classes)
}

# tiered-ImageNet: WordNet-based hierarchy (deep, ~8 levels)
# Simplified top-level structure
TIERED_IMAGENET_TREE: Dict[str, List[str]] = {
    "root": ["artifact", "living_thing", "natural_object", "substance"],
    "artifact": ["instrumentality", "structure", "covering", "container", "commodity"],
    "living_thing": ["animal", "plant", "fungus"],
    "animal": ["vertebrate", "invertebrate"],
    "vertebrate": ["mammal", "bird", "reptile", "amphibian", "fish"],
    "mammal": ["carnivore", "primate", "rodent", "ungulate", "cetacean"],
    "bird": ["passerine", "raptor", "waterfowl", "gamebird"],
    "instrumentality": ["vehicle", "tool", "device", "equipment", "weapon"],
    "structure": ["building", "bridge", "barrier", "establishment"],
    "plant": ["tree", "flower", "grass", "fern"],
    # Each subcategory branches further to reach ~608 leaf classes
}

# iNaturalist: biological taxonomy (Kingdom → Phylum → Class → Order → Family → Genus → Species)
INATURALIST_TREE: Dict[str, List[str]] = {
    "root": ["animalia", "plantae", "fungi"],
    "animalia": ["arthropoda", "chordata", "mollusca", "cnidaria", "annelida"],
    "arthropoda": ["insecta", "arachnida", "crustacea", "myriapoda"],
    "insecta": ["lepidoptera", "coleoptera", "hymenoptera", "diptera",
                "hemiptera", "odonata", "orthoptera"],
    "chordata": ["mammalia", "aves", "reptilia", "amphibia", "actinopterygii"],
    "mammalia": ["carnivora_m", "rodentia_m", "chiroptera_m",
                 "artiodactyla_m", "primates_m", "lagomorpha_m"],
    "aves": ["passeriformes_i", "accipitriformes_i", "anseriformes_i",
             "charadriiformes_i", "piciformes_i", "strigiformes_i",
             "falconiformes_i", "pelecaniformes_i"],
    "reptilia": ["squamata", "testudines", "crocodylia"],
    "amphibia": ["anura", "caudata"],
    "actinopterygii": ["perciformes", "cypriniformes", "siluriformes"],
    "plantae": ["magnoliopsida", "liliopsida", "pinopsida", "polypodiopsida"],
    "magnoliopsida": ["asterales", "rosales", "fabales", "lamiales",
                      "caryophyllales", "brassicales", "ericales"],
    "fungi": ["agaricomycetes", "lecanoromycetes", "pezizomycetes"],
    # Deep biological hierarchy continues for ~7 levels to species
}


HIERARCHY_TREES = {
    "cifar100": CIFAR100_TREE,
    "awa2": AWA2_TREE,
    "cub200": CUB200_TREE,
    "sun397": SUN397_TREE,
    "stanford_cars": STANFORD_CARS_TREE,
    "tiered_imagenet": TIERED_IMAGENET_TREE,
    "inaturalist": INATURALIST_TREE,
}


# ===================================================================
# Metric computation
# ===================================================================

def _tree_depth_recursive(tree: Dict[str, List[str]], node: str) -> int:
    """Compute depth of tree rooted at `node`."""
    if node not in tree or not tree[node]:
        return 0
    return 1 + max(_tree_depth_recursive(tree, child) for child in tree[node])


def tree_depth(tree: Dict[str, List[str]], root: str = "root") -> int:
    """Longest root-to-leaf path length."""
    return _tree_depth_recursive(tree, root)


def count_internal_nodes(tree: Dict[str, List[str]]) -> int:
    """Count nodes that have children (including root)."""
    return sum(1 for node, children in tree.items() if children)


def count_leaves(tree: Dict[str, List[str]]) -> int:
    """Count nodes that appear as children but not as parents."""
    all_children = set()
    for children in tree.values():
        all_children.update(children)
    parents = set(tree.keys())
    return len(all_children - parents)


def mean_branching_factor(tree: Dict[str, List[str]]) -> float:
    """Average number of children per internal node."""
    internal = [(node, children) for node, children in tree.items() if children]
    if not internal:
        return 0.0
    return sum(len(children) for _, children in internal) / len(internal)


def total_nodes(tree: Dict[str, List[str]]) -> int:
    """Total number of unique nodes in the tree."""
    all_nodes = set(tree.keys())
    for children in tree.values():
        all_nodes.update(children)
    return len(all_nodes)


# ===================================================================
# Gromov delta-hyperbolicity
# ===================================================================

def gromov_delta_hyperbolicity(
    distance_matrix: np.ndarray,
    n_samples: int = 5000,
    seed: int = 42,
) -> float:
    """Estimate Gromov delta-hyperbolicity via random quadruple sampling.

    For each quadruple (x, y, z, w), compute the three sums:
      S1 = d(x,y) + d(z,w)
      S2 = d(x,z) + d(y,w)
      S3 = d(x,w) + d(y,z)
    Sort so S1 >= S2 >= S3.
    delta = (S1 - S2) / 2.
    Return the maximum delta over all sampled quadruples.
    """
    n = distance_matrix.shape[0]
    if n < 4:
        return 0.0

    rng = np.random.RandomState(seed)
    max_delta = 0.0

    # If n is small enough, check all quadruples
    n_quads = n * (n - 1) * (n - 2) * (n - 3) // 24
    if n_quads <= n_samples:
        for quad in combinations(range(n), 4):
            x, y, z, w = quad
            s1 = distance_matrix[x, y] + distance_matrix[z, w]
            s2 = distance_matrix[x, z] + distance_matrix[y, w]
            s3 = distance_matrix[x, w] + distance_matrix[y, z]
            sums = sorted([s1, s2, s3], reverse=True)
            delta = (sums[0] - sums[1]) / 2.0
            max_delta = max(max_delta, delta)
    else:
        for _ in range(n_samples):
            indices = rng.choice(n, size=4, replace=False)
            x, y, z, w = indices
            s1 = distance_matrix[x, y] + distance_matrix[z, w]
            s2 = distance_matrix[x, z] + distance_matrix[y, w]
            s3 = distance_matrix[x, w] + distance_matrix[y, z]
            sums = sorted([s1, s2, s3], reverse=True)
            delta = (sums[0] - sums[1]) / 2.0
            max_delta = max(max_delta, delta)

    return max_delta


def compute_delta_from_features(
    features: np.ndarray,
    labels: np.ndarray,
    n_samples: int = 5000,
    seed: int = 42,
) -> float:
    """Compute delta-hyperbolicity from class-level feature centroids."""
    unique_labels = np.unique(labels)
    centroids = np.stack([
        features[labels == lbl].mean(axis=0) for lbl in unique_labels
    ])
    # Pairwise Euclidean distances between centroids
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(centroids, metric="euclidean"))
    return gromov_delta_hyperbolicity(dist_matrix, n_samples, seed)


# ===================================================================
# All-in-one computation
# ===================================================================

def compute_hierarchy_metrics(dataset_name: str) -> Dict[str, float]:
    """Compute all hierarchy metrics for a dataset from its taxonomy tree."""
    tree = HIERARCHY_TREES.get(dataset_name)
    if tree is None:
        return {"depth": 0, "branching_factor": 0, "internal_nodes": 0,
                "n_leaves": 0, "n_total": 0}

    return {
        "depth": tree_depth(tree),
        "branching_factor": round(mean_branching_factor(tree), 2),
        "internal_nodes": count_internal_nodes(tree),
        "n_leaves": count_leaves(tree),
        "n_total": total_nodes(tree),
    }


def compute_all_metrics(
    dataset_name: str,
    features: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    data_root: str = "data",
) -> Dict[str, float]:
    """Compute hierarchy metrics + delta-hyperbolicity for a dataset.

    If features/labels not provided, loads them from disk.
    """
    metrics = compute_hierarchy_metrics(dataset_name)

    # Delta-hyperbolicity from feature centroids
    if features is None or labels is None:
        base = Path(data_root) / dataset_name
        if (base / "features.npy").exists():
            features = np.load(base / "features.npy")
            labels = np.load(base / "labels.npy")

    if features is not None and labels is not None:
        delta = compute_delta_from_features(features, labels)
        metrics["delta_hyperbolicity"] = round(delta, 4)
    else:
        metrics["delta_hyperbolicity"] = None

    return metrics


def save_hierarchy_stats(dataset_name: str, data_root: str = "data") -> None:
    """Compute and save hierarchy stats to JSON."""
    metrics = compute_all_metrics(dataset_name, data_root=data_root)
    out_path = Path(data_root) / dataset_name / "hierarchy_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{dataset_name}] Hierarchy stats saved to {out_path}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


# ===================================================================
# Summary table for all datasets
# ===================================================================

def print_hierarchy_summary() -> None:
    """Print a summary table of hierarchy metrics for all datasets."""
    print(f"\n{'Dataset':<20} {'Depth':>6} {'Branch':>8} {'Internal':>9} "
          f"{'Leaves':>8} {'Total':>7}")
    print("-" * 65)
    for name in HIERARCHY_TREES:
        m = compute_hierarchy_metrics(name)
        print(f"{name:<20} {m['depth']:>6} {m['branching_factor']:>8.2f} "
              f"{m['internal_nodes']:>9} {m['n_leaves']:>8} {m['n_total']:>7}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        help="Compute for a specific dataset (default: all)")
    parser.add_argument("--data_root", type=str, default="data")
    args = parser.parse_args()

    if args.dataset:
        save_hierarchy_stats(args.dataset, args.data_root)
    else:
        print_hierarchy_summary()
