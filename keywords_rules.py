
# keywords_rules.py

# Category keyword map (no overlaps)
keyword_map = {
    "accident": [
        "crash", "collision", "overturned", "skid", "hit", "mishap", "derailment", "wreck", "pileup", "smashed", "run over",
        "slipped", "knocked down", "rammed", "turtle", "flip", "toppled", "bump", "impact", "head on", "fell", "incident",
        "side swipe", "vehicle accident", "car crash", "bike accident", "bus accident", "motorcycle crashed", "truck rolled"
    ],
    "water": [
        "leakage", "pipe burst", "drainage","flood","flooding","flooded", "low water supply", "stagnant water", "drinking water shortage", "pipeline broken", "burst pipeline",
        "water seepage", "contaminated water", "muddy water", "flooded street", "wet floor", "pipe cracked", "tap leaking", "basement flooded",
        "drain water", "sewer water", "leaking valve", "broken reservoir", "water tank overflow", "water level drop", "short water supply",
        "manhole overflow", "damaged pipeline", "supply interruption"
    ],
    "traffic": [
        "traffic jam", "signal jam", "congestion", "traffic congestion", "standstill", "gridlock", "logjam", "queue of vehicles",
        "bumper to bumper", "heavy traffic", "long wait", "diverted traffic", "blocked crossing", "traffic snarl", "lane jam",
        "vehicle queue", "road block", "intersection jam", "rush hour jam", "delayed traffic", "vehicle buildup", "slow lane", "detour"
    ],
    "roads": [
        "pothole", "damaged road", "broken bridge", "collapsed pavement", "cracked surface", "uneven road", "sunken street",
        "eroded surface", "road washed away", "roadside collapse", "unmarked speed breaker", "broken railing", "subsidized ground",
        "sinkhole", "fractured curb", "deteriorated lane", "unpaved road", "slippery surface", "structural crack", "hole in road",
        "shoulder breakdown", "manhole exposed", "crosswalk faded", "loose gravel"
    ],
    "nature": [
        "tree fallen", "branch fallen", "animal sighting", "wild animal", "landslide", "tsunami", "earthquake", "cyclone",
        "storm hit", "thunderstorm", "hurricane", "heavy rainfall", "river overflow", "flood plain", "soil erosion", "hailstorm",
        "bats infestation", "owl trapped", "peacock stuck", "bear wandering", "snake found", "monsoon damage", "natural disaster",
        "damaged by wind", "branches blocking"
    ],
    "fire": [
        "fire outbreak", "raging fire", "blaze", "smouldering", "burning", "ember", "caught fire", "big fire", "flames rising",
        "explosion fire", "smoke alarm", "wildfire", "forest blaze", "property fire", "car on fire", "house fire", "kitchen fire",
        "burning debris", "short spark caused fire", "incident of fire", "fire detected", "ignition", "heat wave", "inferno"
    ],
    "electricity": [
        "power outage", "power failure", "electric shock", "no electricity", "electric wire down", "short circuit",
        "blown fuse", "transformer blast", "current leak", "street light out", "pole down", "supply interruption", "tripped breaker",
        "blackout", "high voltage", "low voltage", "power trip", "supply fluctuation", "disconnect", "burnt cable", "generator down",
        "electrical sparks", "switchboard burnt", "lighting fault"
    ],
    "sanitation": [
        "garbage dump", "overflowing dustbin", "unclean street", "dirty smell", "rotten trash", "foul odor", "dustbin full",
        "heap of garbage", "clogged drain", "open gutter", "blocked sewage", "waste pile", "dead animal", "fly infestation",
        "mosquito breeding", "littered alley", "public toilet dirty", "trash not picked", "manhole open", "unsanitary", 
        "sampling garbage", "filthy condition", "smelly street", "unclean bin", "waste not cleared"
    ]
}

# Severity rules for fixed categories
fixed_severity_map = {
    "accident": "emergency",
    "fire": "emergency",
    "traffic": "urgent"
}

# Severity keywords (for non-emergency/urgent categories)
severity_keyword_map = {
    "critical": [
        "failure","potholes","pothole", "collapse", "danger", "high risk", "landslide", "severely damaged", "major issue",
        "flooding", "immediate help", "hazard", "flood", "tsunami", "life threatening", "earthquake"
    ],
    "needs attention": [
        "issue", "needs cleaning", "attention needed", "problem reported", "not working", "maintenance required",
        "should be fixed", "requires fix", "cleaning pending", "inspection needed", "not operational", "regular problem"
    ],
    "minor": [
        "minor", "small problem", "slightly damaged", "cosmetic issue", "some inconvenience", "slow functioning",
        "temporary", "slightly blocked", "intermittent", "rarely happens"
    ]
}

def get_category_by_keyword(text):
    text_lower = text.lower()
    for category, keywords in keyword_map.items():
        for kw in keywords:
            if kw in text_lower:
                return category
    return None

def get_severity_by_keywords(text, category):
    # Only for non-emergency/urgent categories
    if category in fixed_severity_map:
        return fixed_severity_map[category]
    text_lower = text.lower()
    for severity, keywords in severity_keyword_map.items():
        for kw in keywords:
            if kw in text_lower:
                return severity
    return None  # fallback to model
