"""Unified script to build scripts/phase5_specs_data.py with exactly 300 clusters and 30 hard negatives."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the base specs
from scripts.generate_all_specs import sci_data
from scripts.append_all_specs import math_data
from scripts.build_phase5_benchmark import SPEC_DEFINITIONS as hist_definitions

specs = []

# 1. Science (40 clusters: 10 val, 30 test)
for s in sci_data:
    cid, q, lex, syn, conc, gold, aliases, prov, crit, split = s
    specs.append((cid, "science", q, lex, syn, conc, gold, aliases, prov, crit, split))

# 2. Math (35 clusters: 8 val, 27 test)
specs.extend(math_data)

# 3. History (35 clusters: 8 val, 27 test)
hist_data = [item for item in hist_definitions if item[0].startswith("p5_hist_")]
specs.extend(hist_data)

# 4. Geography (35 clusters: 8 val, 27 test)
geo_raw = [
    ("p5_geo_001", "What is the capital city of Australia?", "Canberra", ["Canberra"], "Australian Government", "val"),
    ("p5_geo_002", "What is the longest river in the world by conventional geographical agreement?", "Nile River", ["Nile", "Nile River"], "Encyclopaedia Britannica", "val"),
    ("p5_geo_003", "Which strait connects the Atlantic Ocean to the Mediterranean Sea?", "Strait of Gibraltar", ["Strait of Gibraltar", "Gibraltar"], "Hydrographic Office", "val"),
    ("p5_geo_004", "What is the capital city of Canada?", "Ottawa", ["Ottawa"], "Natural Resources Canada", "val"),
    ("p5_geo_005", "What is the highest mountain peak in North America?", "Denali", ["Denali", "Mount McKinley"], "USGS National Map", "val"),
    ("p5_geo_006", "Which desert is the largest hot desert on Earth?", "Sahara Desert", ["Sahara", "Sahara Desert"], "Physical Geography", "val"),
    ("p5_geo_007", "What is the capital city of Brazil?", "Brasilia", ["Brasilia", "Brasília"], "IBGE Brazil", "val"),
    ("p5_geo_008", "Which country possesses the longest coastline in the world?", "Canada", ["Canada"], "CIA World Factbook", "val"),
    ("p5_geo_009", "What is the capital city of Japan?", "Tokyo", ["Tokyo"], "Geospatial Information Authority of Japan", "test"),
    ("p5_geo_010", "What mountain range separates the European and Asian continents in Russia?", "Ural Mountains", ["Urals", "Ural Mountains"], "Geography", "test"),
    ("p5_geo_011", "What is the capital city of New Zealand?", "Wellington", ["Wellington"], "Land Information New Zealand", "test"),
    ("p5_geo_012", "Which African lake is the largest by surface area?", "Lake Victoria", ["Lake Victoria", "Victoria"], "East African Geography", "test"),
    ("p5_geo_013", "What is the capital city of Turkey?", "Ankara", ["Ankara"], "General Directorate of Mapping", "test"),
    ("p5_geo_014", "What is the deepest lake in the world by maximum depth?", "Lake Baikal", ["Lake Baikal", "Baikal"], "Limnology", "test"),
    ("p5_geo_015", "What is the capital city of South Korea?", "Seoul", ["Seoul"], "National Geographic Information Institute", "test"),
    ("p5_geo_016", "Which European river flows through Budapest, Belgrade, and Vienna?", "Danube River", ["Danube", "Danube River"], "Danube Commission", "test"),
    ("p5_geo_017", "What is the capital city of Egypt?", "Cairo", ["Cairo"], "Survey of Egypt", "test"),
    ("p5_geo_018", "What island nation lies south of the Strait of Messina from peninsular Italy?", "Sicily", ["Sicily"], "Italian Geography", "test"),
    ("p5_geo_019", "What is the capital city of Argentina?", "Buenos Aires", ["Buenos Aires"], "IGN Argentina", "test"),
    ("p5_geo_020", "Which sea is bordered by Kazakhstan, Turkmenistan, Iran, Azerbaijan, and Russia?", "Caspian Sea", ["Caspian Sea", "Caspian"], "World Ocean Atlas", "test"),
    ("p5_geo_021", "What is the capital city of Kenya?", "Nairobi", ["Nairobi"], "Survey of Kenya", "test"),
    ("p5_geo_022", "Which mountain range extends along the western coast of South America?", "Andes Mountains", ["Andes", "Andes Mountains"], "South American Geography", "test"),
    ("p5_geo_023", "What is the capital city of Thailand?", "Bangkok", ["Bangkok", "Krung Thep"], "Royal Thai Survey Department", "test"),
    ("p5_geo_024", "What narrow body of water separates Alaska from Russia?", "Bering Strait", ["Bering Strait"], "NOAA", "test"),
    ("p5_geo_025", "What is the capital city of Poland?", "Warsaw", ["Warsaw", "Warszawa"], "GUGiK Poland", "test"),
    ("p5_geo_026", "What is the largest island in the Mediterranean Sea by area?", "Sicily", ["Sicily"], "Mediterranean Geography", "test"),
    ("p5_geo_027", "What is the capital city of Switzerland?", "Bern", ["Bern", "Berne"], "Swisstopo", "test"),
    ("p5_geo_028", "Which river flows through the Grand Canyon in the United States?", "Colorado River", ["Colorado River", "Colorado"], "USGS National Map", "test"),
    ("p5_geo_029", "What is the capital city of Norway?", "Oslo", ["Oslo"], "Kartverket Norway", "test"),
    ("p5_geo_030", "What country contains the Salar de Uyuni, the world's largest salt flat?", "Bolivia", ["Bolivia"], "IGM Bolivia", "test"),
    ("p5_geo_031", "What is the capital city of Vietnam?", "Hanoi", ["Hanoi"], "Vietnam Mapping Agency", "test"),
    ("p5_geo_032", "Which canal connects the Mediterranean Sea to the Red Sea?", "Suez Canal", ["Suez Canal", "Suez"], "Suez Canal Authority", "test"),
    ("p5_geo_033", "What is the capital city of Ireland?", "Dublin", ["Dublin"], "Ordnance Survey Ireland", "test"),
    ("p5_geo_034", "Which plateau is known as the 'Roof of the World' due to its high altitude?", "Tibetan Plateau", ["Tibetan Plateau", "Tibet Plateau", "Qinghai-Tibet Plateau"], "Physical Geography", "test"),
    ("p5_geo_035", "What is the capital city of Portugal?", "Lisbon", ["Lisbon", "Lisboa"], "DGT Portugal", "test"),
]

for item in geo_raw:
    cid, q, gold, aliases, prov, split = item
    lex = f"In global geography, what city is the official capital of {q.split('of ')[-1]}" if "capital" in q else f"Which geographical feature is known as: {q}?"
    syn = f"The official designated capital of {q.split('of ')[-1]} is which city?" if "capital" in q else f"Identify the following: {q}"
    conc = f"Capital of {q.split('of ')[-1]}?" if "capital" in q else f"{q}"
    specs.append((cid, "geography", q, lex, syn, conc, gold, aliases, prov, "entity_match", split))

# 5. Dates (30 clusters: 8 val, 22 test)
from scripts.generate_full_benchmark_data import dates_raw
for item in dates_raw:
    cid, q, gold, aliases, prov, split = item
    lex = f"What is the exact calendar year in which {q.replace('In what year did ', '').replace('What year saw ', '').replace('What year marked ', '').rstrip('?')}?"
    syn = f"Which year corresponds to the historical event: {q}?"
    conc = f"Year: {q.replace('In what year did ', '').replace('What year marked ', '')}?"
    specs.append((cid, "dates", q, lex, syn, conc, gold, aliases, prov, "numerical_exact", split))

# 6. Units (30 clusters: 8 val, 22 test)
from scripts.generate_full_benchmark_data import units_raw
for item in units_raw:
    cid, q, gold, aliases, prov, split = item
    lex = f"State the measured quantity: {q}"
    syn = f"Which specific value or unit answers the inquiry: {q}?"
    conc = f"{q.rstrip('?')}:"
    specs.append((cid, "units", q, lex, syn, conc, gold, aliases, prov, "numerical_exact" if any(c.isdigit() for c in gold) else "entity_match", split))

# 7. Everyday (30 clusters: 8 val, 22 test)
from scripts.generate_full_benchmark_data import evr_raw
for item in evr_raw:
    cid, q, gold, aliases, prov, split = item
    lex = f"In everyday knowledge: {q}"
    syn = f"Which common fact corresponds to the query: {q}?"
    conc = f"Practical fact: {q.rstrip('?')}:"
    specs.append((cid, "everyday", q, lex, syn, conc, gold, aliases, prov, "entity_match", split))

# 8. Structured (25 clusters: 6 val, 19 test)
from scripts.generate_full_benchmark_data import str_raw
for item in str_raw:
    cid, q, gold, aliases, prov, split = item
    lex = f"Format the requested response: {q}"
    syn = f"Please output the structured answer for: {q}"
    conc = f"Structured format: {q.rstrip('?')}:"
    specs.append((cid, "structured", q, lex, syn, conc, gold, aliases, prov, "slot_match", split))

# 9. Negation (20 clusters: 5 val, 15 test)
from scripts.generate_full_benchmark_data import neg_raw
for item in neg_raw:
    cid, q, gold, aliases, prov, split = item
    lex = f"Identify the negative/contrasting exception: {q}"
    syn = f"Which specific entity represents the exception in: {q}?"
    conc = f"Negative constraint: {q.rstrip('?')}:"
    specs.append((cid, "negation", q, lex, syn, conc, gold, aliases, prov, "entity_match", split))

# 10. Constraints (20 clusters: 5 val, 15 test)
from scripts.generate_full_benchmark_data import cst_raw
for item in cst_raw:
    cid, q, gold, aliases, prov, split = item
    lex = f"Complying with all constraints: {q}"
    syn = f"Please satisfy this constrained query: {q}"
    conc = f"Constraint query: {q.rstrip('?')}:"
    specs.append((cid, "constraints", q, lex, syn, conc, gold, aliases, prov, "slot_match", split))

from scripts.generate_full_benchmark_data import hard_negatives

print(f"Total unified specs: {len(specs)}")
val_count = sum(1 for s in specs if s[10] == "val")
test_count = sum(1 for s in specs if s[10] == "test")
print(f"Validation: {val_count} | Test: {test_count}")

# Write out scripts/phase5_specs_data.py
out_path = Path("scripts/phase5_specs_data.py")
with open(out_path, "w", encoding="utf-8") as f:
    f.write('"""Phase 5 Curated Benchmark Data: 300 Clusters & 30 Hard Negatives."""\\n\\n')
    f.write("from typing import Any, Dict, List, Tuple\\n\\n")
    f.write("ALL_PHASE5_SPECS: List[Tuple[str, str, str, str, str, str, str, List[str], str, str, str]] = [\\n")
    for s in specs:
        f.write(f"    {repr(s)},\\n")
    f.write("]\\n\\n")
    f.write("HARD_NEGATIVE_SPECS: List[Dict[str, Any]] = [\\n")
    for hn in hard_negatives:
        f.write(f"    {repr(hn)},\\n")
    f.write("]\\n")

print(f"Successfully generated {out_path} with {len(specs)} specs and {len(hard_negatives)} hard negatives.")
