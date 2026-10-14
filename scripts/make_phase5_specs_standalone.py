"""Standalone generator for Phase 5 Benchmark Specifications.

Produces exactly 300 clusters across 10 domains:
  1. science (40)
  2. math (35)
  3. history (35)
  4. geography (35)
  5. dates (30)
  6. units (30)
  7. everyday (30)
  8. structured (25)
  9. negation (20)
 10. constraints (20)

Total: 300 clusters (74 val, 226 test), plus 30 hard negatives.
"""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_all_specs import sci_data
from scripts.generate_full_benchmark_data import (
    dates_raw,
    units_raw,
    evr_raw,
    str_raw,
    neg_raw,
    cst_raw,
    hard_negatives,
)

all_specs = []

# 1. Science (40 clusters)
for s in sci_data:
    cid, q, lex, syn, conc, gold, aliases, prov, crit, split = s
    all_specs.append((cid, "science", q, lex, syn, conc, gold, aliases, prov, crit, split))

# 2. Math (35 clusters)
# Re-extract math_data from append_all_specs without importing SPEC_DEFINITIONS
math_raw = [
    ("p5_math_001", "What is the value of 3 factorial times 4 factorial (3! * 4!)?", "144", ["144", "one hundred forty-four"], "Combinatorics", "numerical_exact", "val"),
    ("p5_math_002", "What is the derivative of natural log of x (ln x) with respect to x for x > 0?", "1/x", ["1/x", "x^(-1)"], "Calculus", "entity_match", "val"),
    ("p5_math_003", "What is the sum of the first 10 positive integers (1 through 10)?", "55", ["55", "fifty-five"], "Arithmetic", "numerical_exact", "val"),
    ("p5_math_004", "In Euclidean geometry, what is the area of a circle with radius 5 in terms of pi?", "25 pi", ["25 pi", "25pi", "25*pi"], "Euclidean Geometry", "entity_match", "val"),
    ("p5_math_005", "What is the greatest common divisor of 36 and 54?", "18", ["18", "eighteen"], "Number Theory", "numerical_exact", "val"),
    ("p5_math_006", "What is the value of sin(pi / 6) in radians?", "0.5", ["0.5", "1/2", "half"], "Trigonometry", "numerical_exact", "val"),
    ("p5_math_007", "What are the two real roots of the quadratic equation x^2 - 9 = 0?", "+3 and -3", ["3 and -3", "+3 and -3", "-3, 3", "3, -3"], "Algebra", "slot_match", "val"),
    ("p5_math_008", "What is the volume of a sphere with radius 3 in terms of pi?", "36 pi", ["36 pi", "36pi"], "Solid Geometry", "entity_match", "val"),
    ("p5_math_009", "What is the limit of (sin x) / x as x approaches 0?", "1", ["1", "1.0", "one"], "Calculus", "numerical_exact", "test"),
    ("p5_math_010", "How many diagonals does a regular convex hexagon possess?", "9", ["9", "nine"], "Discrete Mathematics", "numerical_exact", "test"),
    ("p5_math_011", "What is the determinant of a 2x2 matrix with rows [2, 3] and [1, 4]?", "5", ["5", "five"], "Linear Algebra", "numerical_exact", "test"),
    ("p5_math_012", "What is the value of 5 to the power of 3 (5^3)?", "125", ["125", "one hundred twenty-five"], "Arithmetic", "numerical_exact", "test"),
    ("p5_math_013", "What is the derivative of e^(2x) with respect to x?", "2 e^(2x)", ["2 e^(2x)", "2e^(2x)", "2*e^(2x)"], "Calculus", "entity_match", "test"),
    ("p5_math_014", "What is the sum of interior angles in a convex pentagon in degrees?", "540 degrees", ["540", "540 degrees", "540 deg"], "Geometry", "numerical_exact", "test"),
    ("p5_math_015", "What is the binary representation of the decimal integer 13?", "1101", ["1101", "1101_2", "0b1101"], "Computer Science Math", "entity_match", "test"),
    ("p5_math_016", "What is the variance of a fair six-sided die roll to two decimal places?", "2.92", ["2.92", "35/12"], "Probability and Statistics", "numerical_exact", "test"),
    ("p5_math_017", "What is the value of log base 2 of 64?", "6", ["6", "six"], "Algebra", "numerical_exact", "test"),
    ("p5_math_018", "What is the slope of the line perpendicular to y = (1/3)x + 2?", "-3", ["-3", "negative 3"], "Coordinate Geometry", "numerical_exact", "test"),
    ("p5_math_019", "How many permutations are possible when arranging 4 distinct books on a shelf?", "24", ["24", "twenty-four"], "Combinatorics", "numerical_exact", "test"),
    ("p5_math_020", "What is the length of the diagonal of a square with side length 5?", "5 sqrt(2)", ["5 sqrt(2)", "5*sqrt(2)", "7.07"], "Geometry", "entity_match", "test"),
    ("p5_math_021", "What is the 10th term in the arithmetic sequence 2, 5, 8, 11...?", "29", ["29", "twenty-nine"], "Sequences and Series", "numerical_exact", "test"),
    ("p5_math_022", "What is the solution for x in the equation 2^(x+1) = 32?", "4", ["4", "four", "x = 4"], "Algebra", "numerical_exact", "test"),
    ("p5_math_023", "What is the dot product of vectors [1, 3] and [4, -2]?", "-2", ["-2", "negative 2"], "Linear Algebra", "numerical_exact", "test"),
    ("p5_math_024", "What is the formula for the sum of interior angles of an n-sided polygon?", "(n - 2) * 180 degrees", ["(n-2)*180", "(n - 2) * 180", "180(n-2)"], "Polygon Geometry", "entity_match", "test"),
    ("p5_math_025", "What is the derivative of tan(x) with respect to x?", "sec^2(x)", ["sec^2(x)", "secant squared", "1/cos^2(x)"], "Calculus", "entity_match", "test"),
    ("p5_math_026", "What is the probability of rolling a sum of 7 with two standard six-sided dice?", "1/6", ["1/6", "6/36", "0.167"], "Probability", "numerical_exact", "test"),
    ("p5_math_027", "What is the complex conjugate of 4 + 7i?", "4 - 7i", ["4 - 7i", "4-7i"], "Complex Analysis", "entity_match", "test"),
    ("p5_math_028", "What is the value of 7 choose 2 (combinations C(7,2))?", "21", ["21", "twenty-one"], "Combinatorics", "numerical_exact", "test"),
    ("p5_math_029", "What is the harmonic mean of 2 and 6?", "3", ["3", "3.0", "three"], "Statistics", "numerical_exact", "test"),
    ("p5_math_030", "What is the second derivative of x^4 with respect to x?", "12x^2", ["12x^2", "12 x^2", "12*x^2"], "Calculus", "entity_match", "test"),
    ("p5_math_031", "What is the surface area of a cube with edge length 4?", "96", ["96", "ninety-six"], "Geometry", "numerical_exact", "test"),
    ("p5_math_032", "What is the solution set for x in |x - 5| = 2?", "3 and 7", ["3 and 7", "3, 7", "7 and 3"], "Algebra", "slot_match", "test"),
    ("p5_math_033", "What is the value of arcsin(1) in radians?", "pi / 2", ["pi/2", "pi / 2", "π/2"], "Trigonometry", "entity_match", "test"),
    ("p5_math_034", "What is the remainder when 100 is divided by 7?", "2", ["2", "two"], "Modular Arithmetic", "numerical_exact", "test"),
    ("p5_math_035", "What is the radius of the circle defined by x^2 + y^2 = 49?", "7", ["7", "seven"], "Coordinate Geometry", "numerical_exact", "test"),
]

for item in math_raw:
    cid, q, gold, aliases, prov, crit, split = item
    lex = f"Evaluate the mathematical expression or inquiry: {q}"
    syn = f"Which mathematical value answers: {q}?"
    conc = f"Math query: {q.rstrip('?')}:"
    all_specs.append((cid, "math", q, lex, syn, conc, gold, aliases, prov, crit, split))

# 3. History (35 clusters)
hist_raw = [
    ("p5_hist_001", "What treaty officially ended the American Revolutionary War in 1783?", "Treaty of Paris", ["Treaty of Paris", "Treaty of Paris (1783)", "Peace of Paris"], "Historical Treaties", "val"),
    ("p5_hist_002", "Who was the first Emperor of the Roman Empire?", "Augustus", ["Augustus", "Augustus Caesar", "Octavian"], "Roman History", "val"),
    ("p5_hist_003", "In what year did the Berlin Wall fall, opening borders between East and West Germany?", "1989", ["1989", "nineteen eighty-nine"], "Modern European History", "val"),
    ("p5_hist_004", "Which Egyptian pharaoh's intact tomb was discovered by Howard Carter in 1922?", "Tutankhamun", ["Tutankhamun", "King Tut", "Tutankhamen"], "Egyptology", "val"),
    ("p5_hist_005", "What British passenger liner was sunk by a German U-boat in May 1915?", "RMS Lusitania", ["Lusitania", "RMS Lusitania"], "World War I History", "val"),
    ("p5_hist_006", "Who wrote the 95 Theses in 1517, triggering the Protestant Reformation?", "Martin Luther", ["Martin Luther", "Luther"], "Reformation History", "val"),
    ("p5_hist_007", "What city served as the capital of the Byzantine Empire until its fall in 1453?", "Constantinople", ["Constantinople", "Byzantium", "Istanbul"], "Byzantine Studies", "val"),
    ("p5_hist_008", "Who was the prime minister of the United Kingdom during the Battle of Britain in 1940?", "Winston Churchill", ["Winston Churchill", "Churchill"], "British History", "val"),
    ("p5_hist_009", "In what year did the Apollo 11 mission land the first humans on the Moon?", "1969", ["1969", "nineteen sixty-nine"], "NASA History", "test"),
    ("p5_hist_010", "Who was the primary author of the United States Declaration of Independence in 1776?", "Thomas Jefferson", ["Thomas Jefferson", "Jefferson"], "American History", "test"),
    ("p5_hist_011", "What ancient civilization built the city of Machu Picchu high in the Andes?", "Inca", ["Inca", "Incan", "Inca Empire"], "Andean Archaeology", "test"),
    ("p5_hist_012", "Which French monarch was known as the 'Sun King' and reigned for 72 years?", "Louis XIV", ["Louis XIV", "Louis 14", "Louis the Great"], "French History", "test"),
    ("p5_hist_013", "What ancient trade network linked China and the Mediterranean world across Eurasia?", "Silk Road", ["Silk Road", "Silk Route"], "World History", "test"),
    ("p5_hist_014", "Who was the first female Prime Minister of the United Kingdom, taking office in 1979?", "Margaret Thatcher", ["Margaret Thatcher", "Thatcher"], "Modern British Politics", "test"),
    ("p5_hist_015", "In what battle in 1815 was Napoleon Bonaparte decisively defeated by allied forces?", "Battle of Waterloo", ["Waterloo", "Battle of Waterloo"], "Napoleonic Wars", "test"),
    ("p5_hist_016", "What Russian dynasty ruled from 1613 until the 1917 February Revolution?", "Romanov", ["Romanov", "Romanov dynasty", "House of Romanov"], "Russian History", "test"),
    ("p5_hist_017", "Which battle in 1066 established Norman control over England?", "Battle of Hastings", ["Battle of Hastings", "Hastings"], "Medieval History", "test"),
    ("p5_hist_018", "Who delivered the famous 'I Have a Dream' speech during the 1963 March on Washington?", "Martin Luther King Jr.", ["Martin Luther King Jr.", "MLK", "Martin Luther King"], "US Civil Rights History", "test"),
    ("p5_hist_019", "What major war was concluded by the Treaty of Utrecht in 1713-1715?", "War of the Spanish Succession", ["War of the Spanish Succession"], "European History", "test"),
    ("p5_hist_020", "What Polish-born scientist won Nobel Prizes in both Physics (1903) and Chemistry (1911)?", "Marie Curie", ["Marie Curie", "Madame Curie", "Maria Sklodowska-Curie"], "History of Science", "test"),
    ("p5_hist_021", "Which naval battle in 1571 halted Ottoman maritime expansion in the western Mediterranean?", "Battle of Lepanto", ["Battle of Lepanto", "Lepanto"], "Mediterranean History", "test"),
    ("p5_hist_022", "What document was signed by King John of England at Runnymede in 1215?", "Magna Carta", ["Magna Carta", "Great Charter"], "English Constitutional History", "test"),
    ("p5_hist_023", "Who was the leader of the Soviet Union during the Cuban Missile Crisis in 1962?", "Nikita Khrushchev", ["Nikita Khrushchev", "Khrushchev"], "Cold War History", "test"),
    ("p5_hist_024", "In what year did the Titanic strike an iceberg and sink on its maiden voyage?", "1912", ["1912", "nineteen twelve"], "Maritime History", "test"),
    ("p5_hist_025", "Which African country successfully resisted Italian colonization at the Battle of Adwa in 1896?", "Ethiopia", ["Ethiopia", "Abyssinia"], "African History", "test"),
    ("p5_hist_026", "Who commanded the Confederate Army of Northern Virginia during the American Civil War?", "Robert E. Lee", ["Robert E. Lee", "General Lee", "Robert Lee"], "American Civil War", "test"),
    ("p5_hist_027", "What wall was constructed across northern Britain by a Roman emperor starting in 122 AD?", "Hadrian's Wall", ["Hadrian's Wall", "Hadrian Wall"], "Roman Archaeology", "test"),
    ("p5_hist_028", "Which South African leader was released from prison in 1990 and elected president in 1994?", "Nelson Mandela", ["Nelson Mandela", "Mandela"], "Modern African History", "test"),
    ("p5_hist_029", "What 1648 series of peace treaties ended the Thirty Years' War in Europe?", "Peace of Westphalia", ["Peace of Westphalia", "Treaty of Westphalia"], "European Diplomatic History", "test"),
    ("p5_hist_030", "Who was the supreme Allied commander in Europe during the D-Day invasion of Normandy in 1944?", "Dwight D. Eisenhower", ["Dwight D. Eisenhower", "Eisenhower", "Ike"], "World War II History", "test"),
    ("p5_hist_031", "In what year did the Chernobyl nuclear disaster occur in Soviet Ukraine?", "1986", ["1986", "nineteen eighty-six"], "Nuclear History", "test"),
    ("p5_hist_032", "Who was the first President of the Fifth French Republic, serving from 1959 to 1969?", "Charles de Gaulle", ["Charles de Gaulle", "de Gaulle"], "French Political History", "test"),
    ("p5_hist_033", "What empire conquered the Aztec capital of Tenochtitlan in 1521 under Hernán Cortés?", "Spanish Empire", ["Spain", "Spanish Empire", "Spanish Crown"], "Colonial Latin America", "test"),
    ("p5_hist_034", "Who was the first Secretary of the Treasury of the United States?", "Alexander Hamilton", ["Alexander Hamilton", "Hamilton"], "US Early Republic", "test"),
    ("p5_hist_035", "In what year did the Suez Crisis erupt following Egypt's nationalization of the canal?", "1956", ["1956", "nineteen fifty-six"], "Middle Eastern History", "test"),
]

for item in hist_raw:
    cid, q, gold, aliases, prov, split = item
    lex = f"In historical records: {q}"
    syn = f"Which historical fact satisfies: {q}?"
    conc = f"Historical event: {q.rstrip('?')}:"
    all_specs.append((cid, "history", q, lex, syn, conc, gold, aliases, prov, "numerical_exact" if gold.isdigit() else "entity_match", split))

# 4. Geography (35 clusters)
from scripts.generate_full_benchmark_data import geo_raw
for item in geo_raw:
    cid, q, gold, aliases, prov, split = item
    lex = f"In global geography, what city is the official capital of {q.split('of ')[-1]}" if "capital" in q else f"Which geographical feature is known as: {q}?"
    syn = f"The official designated capital of {q.split('of ')[-1]} is which city?" if "capital" in q else f"Identify the following: {q}"
    conc = f"Capital of {q.split('of ')[-1]}?" if "capital" in q else f"{q}"
    all_specs.append((cid, "geography", q, lex, syn, conc, gold, aliases, prov, "entity_match", split))

# 5. Dates (30 clusters)
for item in dates_raw:
    cid, q, gold, aliases, prov, split = item
    lex = f"What is the exact calendar year in which {q.replace('In what year did ', '').replace('What year saw ', '').replace('What year marked ', '').rstrip('?')}?"
    syn = f"Which year corresponds to the historical event: {q}?"
    conc = f"Year: {q.replace('In what year did ', '').replace('What year marked ', '')}?"
    all_specs.append((cid, "dates", q, lex, syn, conc, gold, aliases, prov, "numerical_exact", split))

# 6. Units (30 clusters)
for item in units_raw:
    cid, q, gold, aliases, prov, split = item
    lex = f"State the measured quantity: {q}"
    syn = f"Which specific value or unit answers the inquiry: {q}?"
    conc = f"{q.rstrip('?')}:"
    all_specs.append((cid, "units", q, lex, syn, conc, gold, aliases, prov, "numerical_exact" if any(c.isdigit() for c in gold) else "entity_match", split))

# 7. Everyday (30 clusters)
for item in evr_raw:
    cid, q, gold, aliases, prov, split = item
    lex = f"In everyday knowledge: {q}"
    syn = f"Which common fact corresponds to the query: {q}?"
    conc = f"Practical fact: {q.rstrip('?')}:"
    all_specs.append((cid, "everyday", q, lex, syn, conc, gold, aliases, prov, "entity_match", split))

# 8. Structured (25 clusters)
for item in str_raw:
    cid, q, gold, aliases, prov, split = item
    lex = f"Format the requested response: {q}"
    syn = f"Please output the structured answer for: {q}"
    conc = f"Structured format: {q.rstrip('?')}:"
    all_specs.append((cid, "structured", q, lex, syn, conc, gold, aliases, prov, "slot_match", split))

# 9. Negation (20 clusters)
for item in neg_raw:
    cid, q, gold, aliases, prov, split = item
    lex = f"Identify the negative/contrasting exception: {q}"
    syn = f"Which specific entity represents the exception in: {q}?"
    conc = f"Negative constraint: {q.rstrip('?')}:"
    all_specs.append((cid, "negation", q, lex, syn, conc, gold, aliases, prov, "entity_match", split))

# 10. Constraints (20 clusters)
for item in cst_raw:
    cid, q, gold, aliases, prov, split = item
    lex = f"Complying with all constraints: {q}"
    syn = f"Please satisfy this constrained query: {q}"
    conc = f"Constraint query: {q.rstrip('?')}:"
    all_specs.append((cid, "constraints", q, lex, syn, conc, gold, aliases, prov, "slot_match", split))

print(f"Total specs generated: {len(all_specs)}")
val_count = sum(1 for s in all_specs if s[10] == "val")
test_count = sum(1 for s in all_specs if s[10] == "test")
print(f"Validation: {val_count} | Test: {test_count}")

# Write to scripts/phase5_specs_data.py
out_file = Path("scripts/phase5_specs_data.py")
with open(out_file, "w", encoding="utf-8") as f:
    f.write('"""Phase 5 Curated Benchmark Data: 300 Clusters & 30 Hard Negatives."""\n\n')
    f.write("from typing import Any, Dict, List, Tuple\n\n")
    f.write("ALL_PHASE5_SPECS: List[Tuple[str, str, str, str, str, str, str, List[str], str, str, str]] = [\n")
    for s in all_specs:
        f.write(f"    {repr(s)},\n")
    f.write("]\n\n")
    f.write("HARD_NEGATIVE_SPECS: List[Dict[str, Any]] = [\n")
    for hn in hard_negatives:
        f.write(f"    {repr(hn)},\n")
    f.write("]\n")

print(f"Saved {out_file} successfully.")
