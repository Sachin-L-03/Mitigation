import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

base_110 = []
full_specs = []

# --- 4. GEOGRAPHY (35 clusters: 8 val, 27 test) ---
geo_raw = [
    ("p5_geo_001", "What is the capital city of Australia?", "Canberra", ["Canberra"], "Australian Government", "val"),
    ("p5_geo_002", "What is the longest river in the world by conventional geographical agreement?", "Nile River", ["Nile", "Nile River"], "Encyclopaedia Britannica", "val"),
    ("p5_geo_003", "Which strait connects the Atlantic Ocean to the Mediterranean Sea?", "Strait of Gibraltar", ["Strait of Gibraltar", "Gibraltar"], "Hydrographic Office", "val"),
    ("p5_geo_004", "What is the capital city of Finland?", "Helsinki", ["Helsinki"], "Maanmittauslaitos Finland", "val"),
    ("p5_geo_005", "What is the highest mountain peak in North America?", "Denali", ["Denali", "Mount McKinley"], "USGS National Map", "val"),
    ("p5_geo_006", "Which desert is the largest hot desert on Earth?", "Sahara Desert", ["Sahara", "Sahara Desert"], "Physical Geography", "val"),
    ("p5_geo_007", "What is the capital city of Brazil?", "Brasilia", ["Brasilia", "Brasília"], "IBGE Brazil", "val"),
    ("p5_geo_008", "Which country possesses the longest coastline in the world?", "Canada", ["Canada"], "CIA World Factbook", "val"),
    ("p5_geo_009", "What is the capital city of Denmark?", "Copenhagen", ["Copenhagen", "København"], "SDFI Denmark", "test"),
    ("p5_geo_010", "What mountain range separates the European and Asian continents in Russia?", "Ural Mountains", ["Urals", "Ural Mountains"], "Geography", "test"),
    ("p5_geo_011", "What is the capital city of New Zealand?", "Wellington", ["Wellington"], "Land Information New Zealand", "test"),
    ("p5_geo_012", "Which African lake is the largest by surface area?", "Lake Victoria", ["Lake Victoria", "Victoria"], "East African Geography", "test"),
    ("p5_geo_013", "What is the capital city of Turkey?", "Ankara", ["Ankara"], "General Directorate of Mapping", "test"),
    ("p5_geo_014", "What is the deepest lake in the world by maximum depth?", "Lake Baikal", ["Lake Baikal", "Baikal"], "Limnology", "test"),
    ("p5_geo_015", "What is the capital city of Chile?", "Santiago", ["Santiago"], "IGM Chile", "test"),
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
    full_specs.append((cid, "geography", q, lex, syn, conc, gold, aliases, prov, "entity_match", split))

# --- 5. DATES & CHRONOLOGY (30 clusters: 8 val, 22 test) ---
dates_raw = [
    ("p5_date_001", "In what year was the United Nations established following World War II?", "1945", ["1945"], "UN Charter", "val"),
    ("p5_date_002", "What year marked the signing of the Magna Carta at Runnymede?", "1215", ["1215"], "British Library", "val"),
    ("p5_date_003", "In what year did the Apollo 11 mission land humans on the Moon?", "1969", ["1969"], "NASA Historical Data", "val"),
    ("p5_date_004", "What year did the Western Roman Empire fall according to conventional history?", "476 AD", ["476", "476 AD"], "Cambridge Ancient History", "val"),
    ("p5_date_005", "In what year did Christopher Columbus first reach the Caribbean Americas?", "1492", ["1492"], "Columbian Documents", "val"),
    ("p5_date_006", "What year did the United States officially declare its independence from Britain?", "1776", ["1776"], "National Archives", "val"),
    ("p5_date_007", "In what year did World War I officially begin following the Sarajevo assassination?", "1914", ["1914"], "Imperial War Museum", "val"),
    ("p5_date_008", "What year saw the dissolution of the Soviet Union?", "1991", ["1991"], "Modern History", "val"),
    ("p5_date_009", "In what year did the Spanish Armada launch its failed invasion of England?", "1588", ["1588"], "Tudor History", "test"),
    ("p5_date_010", "What year did the Battle of Waterloo take place, ending the Napoleonic Wars?", "1815", ["1815"], "Waterloo Association", "test"),
    ("p5_date_011", "In what year was the printing press invented in Europe by Johannes Gutenberg?", "1440", ["1440", "c. 1440", "1450"], "History of Printing", "test"),
    ("p5_date_012", "What year did the Constantinople fall to the Ottoman Empire under Mehmed II?", "1453", ["1453"], "Byzantine Studies", "test"),
    ("p5_date_013", "In what year was the Treaty of Versailles signed, concluding World War I?", "1919", ["1919"], "League of Nations Archives", "test"),
    ("p5_date_014", "What year did the English Civil War begin between Parliamentarians and Royalists?", "1642", ["1642"], "English Heritage", "test"),
    ("p5_date_015", "In what year did Yuri Gagarin become the first human to travel into space?", "1961", ["1961"], "Roscosmos History", "test"),
    ("p5_date_016", "What year saw the adoption of the United States Constitution in Philadelphia?", "1787", ["1787"], "Library of Congress", "test"),
    ("p5_date_017", "In what year did the Protestant Reformation begin when Luther circulated the 95 Theses?", "1517", ["1517"], "Reformation Archives", "test"),
    ("p5_date_018", "What year did the Russo-Japanese War break out?", "1904", ["1904"], "Naval War College", "test"),
    ("p5_date_019", "In what year was the first permanent English settlement established at Jamestown?", "1607", ["1607"], "Jamestown-Yorktown Foundation", "test"),
    ("p5_date_020", "What year did the Glorious Revolution occur in England, deposing James II?", "1688", ["1688"], "British Constitutional History", "test"),
    ("p5_date_021", "In what year did the Wright Brothers achieve the first controlled powered airplane flight?", "1903", ["1903"], "Smithsonian Air and Space", "test"),
    ("p5_date_022", "What year did the Great Fire of London devastate the city?", "1666", ["1666"], "Museum of London", "test"),
    ("p5_date_023", "In what year was the Emancipation Proclamation issued by Abraham Lincoln?", "1863", ["1863"], "National Archives", "test"),
    ("p5_date_024", "What year saw the start of the California Gold Rush following Sutter's Mill discovery?", "1848", ["1848"], "California Historical Society", "test"),
    ("p5_date_025", "In what year did the Irish Free State gain establishment following the Anglo-Irish Treaty?", "1922", ["1922"], "National Archives of Ireland", "test"),
    ("p5_date_026", "What year did the Japanese attack on Pearl Harbor occur?", "1941", ["1941"], "US Navy Memorial", "test"),
    ("p5_date_027", "In what year did the Protestant Reformation's Council of Trent conclude its sessions?", "1563", ["1563"], "Catholic Historical Review", "test"),
    ("p5_date_028", "What year did the Apollo-Soyuz Test Project mark the first joint US-Soviet spaceflight?", "1975", ["1975"], "NASA History Office", "test"),
    ("p5_date_029", "In what year was the euro physical currency launched in cash circulation?", "2002", ["2002"], "European Central Bank", "test"),
    ("p5_date_030", "What year did the Panama Canal officially open for international maritime traffic?", "1914", ["1914"], "Panama Canal Authority", "test"),
]

for item in dates_raw:
    cid, q, gold, aliases, prov, split = item
    lex = f"What is the exact calendar year in which {q.replace('In what year did ', '').replace('What year saw ', '').replace('What year marked ', '').rstrip('?')}?"
    syn = f"Which year corresponds to the historical event: {q}?"
    conc = f"Year: {q.replace('In what year did ', '').replace('What year marked ', '')}?"
    full_specs.append((cid, "dates", q, lex, syn, conc, gold, aliases, prov, "numerical_exact", split))

# --- 6. UNITS, MEASUREMENTS & CONSTANTS (30 clusters: 8 val, 22 test) ---
units_raw = [
    ("p5_unit_001", "What is the value of Avogadro's constant to three significant figures?", "6.02 x 10^23 mol^-1", ["6.02 x 10^23", "6.022 x 10^23", "6.022e23"], "CODATA", "val"),
    ("p5_unit_002", "How many centimeters are in one standard international inch?", "2.54 cm", ["2.54", "2.54 cm", "2.54 centimeters"], "NIST Weights and Measures", "val"),
    ("p5_unit_003", "What is the standard acceleration due to Earth's gravity in m/s^2 to two decimal places?", "9.81 m/s^2", ["9.81", "9.81 m/s^2", "9.80 m/s^2"], "BIPM", "val"),
    ("p5_unit_004", "How many feet are in one statute mile?", "5280 feet", ["5280", "5,280", "5280 feet"], "US Survey Standards", "val"),
    ("p5_unit_005", "What is the SI derived unit of electric potential and electromotive force?", "volt", ["V", "volt", "Volt"], "BIPM SI Brochure", "val"),
    ("p5_unit_006", "What temperature in degrees Fahrenheit corresponds to 100 degrees Celsius?", "212 degrees Fahrenheit", ["212", "212 F", "212 degrees Fahrenheit"], "Thermodynamics", "val"),
    ("p5_unit_007", "How many grams are in 2.5 kilograms?", "2500 grams", ["2500", "2,500", "2500 g", "2500 grams"], "Metric System", "val"),
    ("p5_unit_008", "What is the universal gas constant R in J/(mol K) to two decimal places?", "8.31 J/(mol K)", ["8.31", "8.314", "8.31 J/(mol K)"], "CODATA 2018", "val"),
    ("p5_unit_009", "How many seconds are in 2 hours?", "7200 seconds", ["7200", "7,200", "7200 s", "7200 seconds"], "Time Measurement", "test"),
    ("p5_unit_010", "What is the SI unit of luminous intensity?", "candela", ["cd", "candela", "Candela"], "BIPM SI Brochure", "test"),
    ("p5_unit_011", "How many milliliters are in 1.75 liters?", "1750 mL", ["1750", "1,750", "1750 mL", "1750 ml"], "Liquid Measurement", "test"),
    ("p5_unit_012", "What is the elementary charge e in Coulombs to two significant figures?", "1.6 x 10^-19 C", ["1.6 x 10^-19", "1.602 x 10^-19", "1.6e-19"], "CODATA", "test"),
    ("p5_unit_013", "How many pounds are in 1 short US ton?", "2000 pounds", ["2000", "2,000", "2000 lbs", "2000 pounds"], "NIST Handbook 44", "test"),
    ("p5_unit_014", "What is the SI unit of frequency representing cycles per second?", "hertz", ["Hz", "hertz", "Hertz"], "BIPM", "test"),
    ("p5_unit_015", "How many meters are in 5 kilometers?", "5000 meters", ["5000", "5,000", "5000 m"], "Metric Standards", "test"),
    ("p5_unit_016", "What is the speed of light in vacuum in km/s rounded to nearest thousand?", "300,000 km/s", ["300,000", "300000", "300,000 km/s"], "NIST Physical Data", "test"),
    ("p5_unit_017", "How many fluid ounces are in one standard US pint?", "16 fluid ounces", ["16", "16 fl oz", "16 fluid ounces"], "Culinary Measures", "test"),
    ("p5_unit_018", "What is the SI unit of luminous intensity?", "candela", ["cd", "candela", "Candela"], "BIPM SI Brochure", "test"),
    ("p5_unit_019", "What is absolute zero temperature in degrees Celsius?", "-273.15 degrees Celsius", ["-273.15", "-273.15 C", "-273.15 degrees Celsius"], "NIST", "test"),
    ("p5_unit_020", "How many square feet are in 1 acre?", "43,560 square feet", ["43560", "43,560", "43560 sq ft"], "Land Measurement Standards", "test"),
    ("p5_unit_021", "What is the SI unit of solid angle?", "steradian", ["sr", "steradian", "Steradian"], "BIPM SI Brochure", "test"),
    ("p5_unit_022", "How many millimeters are in 45 centimeters?", "450 millimeters", ["450", "450 mm", "450 millimeters"], "Metric Standards", "test"),
    ("p5_unit_023", "What is the pressure of 1 standard atmosphere in kilopascals to one decimal place?", "101.3 kPa", ["101.3", "101.325", "101.3 kPa"], "Atmospheric Standards", "test"),
    ("p5_unit_024", "How many yards are in 30 feet?", "10 yards", ["10", "10 yd", "10 yards"], "Imperial Standards", "test"),
    ("p5_unit_025", "What is the SI unit of energy and work?", "joule", ["J", "joule", "Joule"], "BIPM", "test"),
    ("p5_unit_026", "How many hours are in one standard non-leap calendar year?", "8760 hours", ["8760", "8,760", "8760 hours"], "Astronomical Calendar", "test"),
    ("p5_unit_027", "What is the SI derived unit of radioactivity measuring disintegrations per second?", "becquerel", ["Bq", "becquerel", "Becquerel"], "BIPM", "test"),
    ("p5_unit_028", "How many cups are in one standard US gallon?", "16 cups", ["16", "16 cups"], "US Customary Measures", "test"),
    ("p5_unit_029", "What is standard atmospheric temperature in Kelvin at 0 degrees Celsius?", "273.15 K", ["273.15", "273.15 K", "273 K"], "NIST", "test"),
    ("p5_unit_030", "What is the SI unit of force equivalent to kg m/s^2?", "newton", ["N", "newton", "Newton"], "BIPM SI Brochure", "test"),
]

for item in units_raw:
    cid, q, gold, aliases, prov, split = item
    lex = f"State the measured quantity: {q}"
    syn = f"Which specific value or unit answers the inquiry: {q}?"
    conc = f"{q.rstrip('?')}:"
    full_specs.append((cid, "units", q, lex, syn, conc, gold, aliases, prov, "numerical_exact" if any(c.isdigit() for c in gold) else "entity_match", split))

# --- 7. EVERYDAY PRACTICAL QA (30 clusters: 8 val, 22 test) ---
evr_raw = [
    ("p5_evr_001", "What common household mineral substance is chemically identified as sodium bicarbonate?", "baking soda", ["baking soda", "bicarbonate of soda"], "Household Chemistry", "val"),
    ("p5_evr_002", "What is the standard internal cooking temperature for safely cooked chicken in degrees Fahrenheit?", "165 degrees Fahrenheit", ["165", "165 F", "165 degrees Fahrenheit"], "USDA Food Safety", "val"),
    ("p5_evr_003", "What primary gas do green plants absorb from the air during daylight photosynthesis?", "carbon dioxide", ["carbon dioxide", "CO2"], "Botany", "val"),
    ("p5_evr_004", "How many teeth does a typical adult human have, including wisdom teeth?", "32 teeth", ["32", "32 teeth"], "Dental Anatomy", "val"),
    ("p5_evr_005", "What is the primary ingredient in traditional bread that causes dough to rise?", "yeast", ["yeast", "baker's yeast", "Saccharomyces cerevisiae"], "Food Science", "val"),
    ("p5_evr_006", "What color does blue litmus paper turn when exposed to an acidic solution?", "red", ["red", "pink"], "General Chemistry", "val"),
    ("p5_evr_007", "What emergency phone number is used in the United States and Canada for police and fire services?", "911", ["911", "9-1-1"], "Telecommunications", "val"),
    ("p5_evr_008", "What common kitchen liquid is an aqueous solution containing approximately 5% acetic acid?", "vinegar", ["vinegar", "white vinegar"], "Food Chemistry", "val"),
    ("p5_evr_009", "How many sides does a standard honeycomb cell built by honeybees have?", "6 sides", ["6", "six", "6 sides"], "Entomology", "test"),
    ("p5_evr_010", "What organ in the human body produces insulin to regulate blood sugar?", "pancreas", ["pancreas"], "Endocrinology", "test"),
    ("p5_evr_011", "What is the main acid found in lemons, limes, and oranges?", "citric acid", ["citric acid"], "Food Science", "test"),
    ("p5_evr_012", "What is the boiling point of pure water at sea level in degrees Fahrenheit?", "212 degrees Fahrenheit", ["212", "212 F", "212 degrees Fahrenheit"], "Physics Handbook", "test"),
    ("p5_evr_013", "Which insect is known for producing commercial natural silk fiber?", "silkworm", ["silkworm", "Bombyx mori"], "Textile Science", "test"),
    ("p5_evr_014", "What is the largest internal organ in the human body by weight?", "liver", ["liver"], "Human Anatomy", "test"),
    ("p5_evr_015", "What substance makes up the majority of a bird's eggshell by mass?", "calcium carbonate", ["calcium carbonate", "CaCO3"], "Poultry Science", "test"),
    ("p5_evr_016", "What is the primary solvent in commercial glass cleaning sprays?", "water", ["water", "H2O"], "Consumer Chemistry", "test"),
    ("p5_evr_017", "What common grain is used as the base for brewing traditional Japanese sake?", "rice", ["rice"], "Culinary Science", "test"),
    ("p5_evr_018", "How many chambers does a normal human heart contain?", "4 chambers", ["4", "four", "4 chambers"], "Cardiology", "test"),
    ("p5_evr_019", "What naturally occurring sweet substance is produced by honeybees from floral nectar?", "honey", ["honey"], "Apiculture", "test"),
    ("p5_evr_020", "What is the normal human body core temperature in Celsius?", "37 degrees Celsius", ["37", "37 C", "36.5-37.5", "37 degrees Celsius"], "Physiology", "test"),
    ("p5_evr_021", "What type of simple machine is a standard crowbar classified as?", "lever", ["lever", "first class lever"], "Classical Mechanics", "test"),
    ("p5_evr_022", "What chemical element is added to municipal tap water in many countries to prevent dental cavities?", "fluoride", ["fluorine", "fluoride"], "Public Health", "test"),
    ("p5_evr_023", "What vegetable is the primary base ingredient in traditional sauerkraut?", "cabbage", ["cabbage"], "Food Preservation", "test"),
    ("p5_evr_024", "What is the hardest substance in the human body covering the outer tooth crown?", "tooth enamel", ["enamel", "tooth enamel"], "Dentistry", "test"),
    ("p5_evr_025", "What common fruit is known for containing significant amounts of potassium and turning brown due to polyphenol oxidase?", "banana", ["banana"], "Plant Biochemistry", "test"),
    ("p5_evr_026", "What type of lens is used in magnifying glasses to enlarge images?", "convex lens", ["convex", "convex lens", "converging lens"], "Optics", "test"),
    ("p5_evr_027", "What domestic animal's milk is used to produce traditional genuine feta cheese in Greece?", "sheep milk", ["sheep", "sheep milk", "sheep and goat milk"], "Dairy Standards", "test"),
    ("p5_evr_028", "What transparent outer layer of the eye covers the iris and pupil, refracting incoming light?", "cornea", ["cornea"], "Ophthalmology", "test"),
    ("p5_evr_029", "What gas is used to inflate party balloons so they float in air?", "helium", ["helium", "He"], "Chemistry", "test"),
    ("p5_evr_030", "What is the primary protein found in human hair and fingernails?", "keratin", ["keratin"], "Dermatology", "test"),
]

for item in evr_raw:
    cid, q, gold, aliases, prov, split = item
    lex = f"In everyday knowledge: {q}"
    syn = f"Which common fact corresponds to the query: {q}?"
    conc = f"Practical fact: {q.rstrip('?')}:"
    full_specs.append((cid, "everyday", q, lex, syn, conc, gold, aliases, prov, "entity_match", split))

# --- 8. STRUCTURED OUTPUTS (25 clusters: 6 val, 19 test) ---
str_raw = [
    ("p5_str_001", "List the 3 primary additive colors of light in alphabetical order.", "blue, green, red", ["blue, green, red"], "Color Science", "val"),
    ("p5_str_002", "Provide the chemical symbol and atomic number of iron as: Symbol: X | Number: Y.", "Symbol: Fe | Number: 26", ["Symbol: Fe | Number: 26", "Fe, 26"], "Periodic Table", "val"),
    ("p5_str_003", "List the 3 domains of biological cellular life separated by commas.", "Archaea, Bacteria, Eukarya", ["Archaea, Bacteria, Eukarya"], "Taxonomy", "val"),
    ("p5_str_004", "State the author and release year of the novel 1984 as: Author (Year).", "George Orwell (1949)", ["George Orwell (1949)", "Orwell (1949)"], "Literature", "val"),
    ("p5_str_005", "List the 3 bones of the human middle ear in order from eardrum to inner ear.", "malleus, incus, stapes", ["malleus, incus, stapes", "hammer, anvil, stirrup"], "Anatomy", "val"),
    ("p5_str_006", "State the chemical formula of methane and ammonia separated by a slash.", "CH4 / NH3", ["CH4 / NH3", "CH4/NH3"], "IUPAC / Chemistry", "val"),
    ("p5_str_007", "List the two houses of the United States Congress separated by a semicolon.", "Senate; House of Representatives", ["Senate; House of Representatives"], "Civics", "test"),
    ("p5_str_008", "Provide the planet with prominent rings and its largest moon as: Planet - Moon.", "Saturn - Titan", ["Saturn - Titan", "Saturn: Titan"], "Astronomy", "test"),
    ("p5_str_009", "List the two subatomic particles composing atomic nuclei separated by a comma.", "protons, neutrons", ["protons, neutrons"], "Physics", "test"),
    ("p5_str_010", "State the capital of France and its river in format: Capital: X | River: Y.", "Capital: Paris | River: Seine", ["Capital: Paris | River: Seine"], "Geography", "test"),
    ("p5_str_011", "List the two official languages of Canada in alphabetical order.", "English, French", ["English, French"], "Canadian Government", "test"),
    ("p5_str_012", "Provide the discoverer and discovery year of penicillin as: Discoverer (Year).", "Alexander Fleming (1928)", ["Alexander Fleming (1928)", "Fleming (1928)"], "Medical History", "test"),
    ("p5_str_013", "List the 3 states of water from coldest to hottest separated by commas.", "ice, liquid water, steam", ["ice, liquid water, steam", "solid, liquid, gas"], "Physics", "test"),
    ("p5_str_014", "Provide the creator and launch year of Python as: Creator (Year).", "Guido van Rossum (1991)", ["Guido van Rossum (1991)", "van Rossum (1991)"], "Software History", "test"),
    ("p5_str_015", "List the two official countries sharing the Iberian Peninsula.", "Spain, Portugal", ["Spain, Portugal"], "Geography", "test"),
    ("p5_str_016", "State the SI unit of force and energy separated by a slash.", "newton / joule", ["newton / joule", "N / J"], "Physics", "test"),
    ("p5_str_017", "List the two chambers of the heart that receive venous blood.", "right atrium, left atrium", ["right atrium, left atrium", "atria"], "Cardiology", "test"),
    ("p5_str_018", "Provide the common name and chemical formula for table salt as: Name (Formula).", "Table salt (NaCl)", ["Table salt (NaCl)", "Salt (NaCl)"], "Chemistry", "test"),
    ("p5_str_019", "List the two islands that constitute mainland New Zealand.", "North Island, South Island", ["North Island, South Island"], "Geography", "test"),
    ("p5_str_020", "State the creator of Git and its initial release year as: Creator, Year.", "Linus Torvalds, 2005", ["Linus Torvalds, 2005", "Torvalds, 2005"], "Software History", "test"),
    ("p5_str_021", "List the two main gases composing the sun's mass in descending order.", "hydrogen, helium", ["hydrogen, helium"], "Astrophysics", "test"),
    ("p5_str_022", "Provide the chemical name and IUPAC formula for laughing gas as: Name: X | Formula: Y.", "Name: Nitrous oxide | Formula: N2O", ["Nitrous oxide | N2O", "Nitrous oxide, N2O"], "Chemistry", "test"),
    ("p5_str_023", "List the 3 branches of the US government in alphabetical order.", "Executive, Judicial, Legislative", ["Executive, Judicial, Legislative"], "Civics", "test"),
    ("p5_str_024", "State the two official currencies of the US and UK separated by a semicolon.", "US Dollar; British Pound", ["US Dollar; British Pound", "USD; GBP"], "Finance", "test"),
    ("p5_str_025", "List the 3 macronutrients in human diet separated by commas.", "carbohydrates, proteins, fats", ["carbohydrates, proteins, fats", "carbs, protein, fat"], "Nutrition", "test"),
]

for item in str_raw:
    cid, q, gold, aliases, prov, split = item
    lex = f"Format the requested response: {q}"
    syn = f"Please output the structured answer for: {q}"
    conc = f"Structured format: {q.rstrip('?')}:"
    full_specs.append((cid, "structured", q, lex, syn, conc, gold, aliases, prov, "slot_match", split))

# --- 9. NEGATION & POLARITY (20 clusters: 5 val, 15 test) ---
neg_raw = [
    ("p5_neg_001", "Which planet in our solar system does NOT have any moons?", "Mercury / Venus", ["Mercury", "Venus", "Mercury and Venus"], "NASA Planetary Fact Sheet", "val"),
    ("p5_neg_002", "Which blood type lacks both A and B erythrocyte surface antigens?", "type O", ["type O", "O", "blood type O"], "Hematology", "val"),
    ("p5_neg_003", "Which state of matter has a definite volume but does NOT have a definite shape?", "liquid", ["liquid", "Liquid"], "Physics", "val"),
    ("p5_neg_004", "What type of chemical bond does NOT involve the sharing or transfer of electrons but rather weak dipole interactions?", "van der Waals force / hydrogen bond", ["van der Waals", "hydrogen bond", "dipole-dipole"], "Chemical Bonding", "val"),
    ("p5_neg_005", "Which of the following does NOT propagate through a vacuum: light, sound, or radio waves?", "sound", ["sound", "sound waves"], "Acoustics", "val"),
    ("p5_neg_006", "Which subatomic nucleon carries NO electric charge?", "neutron", ["neutron", "neutrons"], "Particle Physics", "test"),
    ("p5_neg_007", "Which major biological macromolecule does NOT form repeating polymeric chains: proteins, carbohydrates, or lipids?", "lipids", ["lipids", "fats"], "Biochemistry", "test"),
    ("p5_neg_008", "Which continent does NOT contain any sovereign United Nations member states?", "Antarctica", ["Antarctica"], "Political Geography", "test"),
    ("p5_neg_009", "Which noble gas does NOT exhibit any stable radioactive isotopes in nature: radon or neon?", "neon", ["neon", "Ne"], "Nuclear Chemistry", "test"),
    ("p5_neg_010", "Which blood vessel does NOT carry deoxygenated blood to the heart: vena cava or pulmonary vein?", "pulmonary vein", ["pulmonary vein"], "Cardiovascular Anatomy", "test"),
    ("p5_neg_011", "Which of the four fundamental forces does NOT have an infinite theoretical range: gravity or strong nuclear force?", "strong nuclear force", ["strong force", "strong nuclear force"], "Particle Physics", "test"),
    ("p5_neg_012", "Which category of animals does NOT produce amniotic eggs: reptiles, birds, or amphibians?", "amphibians", ["amphibians"], "Zoology", "test"),
    ("p5_neg_013", "Which temperature scale does NOT use negative numbers for real physical temperatures?", "Kelvin", ["Kelvin", "kelvin", "Rankine"], "Thermodynamics", "test"),
    ("p5_neg_014", "Which human cell type does NOT possess a nucleus at mature functional stage: neuron or erythrocyte?", "erythrocyte", ["erythrocyte", "red blood cell", "RBC"], "Hematology", "test"),
    ("p5_neg_015", "Which organelle does NOT have a lipid membrane envelope: ribosome or lysosome?", "ribosome", ["ribosome", "ribosomes"], "Cell Biology", "test"),
    ("p5_neg_016", "Which celestial body does NOT shine by its own thermonuclear fusion light: Sun or Moon?", "Moon", ["Moon", "the Moon"], "Astronomy", "test"),
    ("p5_neg_017", "Which kingdom of life does NOT contain multicellular organisms: Plantae or Archaea?", "Archaea", ["Archaea"], "Taxonomy", "test"),
    ("p5_neg_018", "Which of the following numbers is NOT prime: 2, 7, 9, or 13?", "9", ["9", "nine"], "Number Theory", "test"),
    ("p5_neg_019", "Which European country does NOT use the euro as its national currency: France or Norway?", "Norway", ["Norway"], "European Economics", "test"),
    ("p5_neg_020", "Which cranial nerve does NOT control ocular eye movements: optic nerve or oculomotor nerve?", "optic nerve", ["optic nerve", "cranial nerve II"], "Neuroanatomy", "test"),
]

for item in neg_raw:
    cid, q, gold, aliases, prov, split = item
    lex = f"Identify the negative/contrasting exception: {q}"
    syn = f"Which specific entity represents the exception in: {q}?"
    conc = f"Negative constraint: {q.rstrip('?')}:"
    full_specs.append((cid, "negation", q, lex, syn, conc, gold, aliases, prov, "entity_match", split))

# --- 10. CONSTRAINTS & HARD NEGATIVES (20 clusters: 5 val, 15 test) ---
cst_raw = [
    ("p5_cst_001", "Name exactly 3 noble gases from the periodic table in any order.", "helium, neon, argon", ["helium, neon, argon", "neon, argon, krypton"], "Periodic Table", "val"),
    ("p5_cst_002", "State the boiling point of pure ethanol at 1 atmosphere in Celsius to one decimal place.", "78.4 degrees Celsius", ["78.4", "78.4 C", "78.37 C", "78.4 degrees Celsius"], "Chemical CRC", "val"),
    ("p5_cst_003", "Answer in exactly one word: What is the capital of Italy?", "Rome", ["Rome", "Roma"], "Geography", "val"),
    ("p5_cst_004", "List exactly 2 countries that share a land border with the United States.", "Canada, Mexico", ["Canada, Mexico", "Mexico, Canada"], "Geography", "val"),
    ("p5_cst_005", "Convert 100 meters into centimeters, answering with the number only.", "10000", ["10000", "10,000"], "Math", "val"),
    ("p5_cst_006", "State the charge of an electron in Coulombs to exactly two decimal places in scientific notation (e.g. -1.60 x 10^-19 C).", "-1.60 x 10^-19 C", ["-1.60 x 10^-19 C", "-1.60e-19 C", "-1.60 x 10^-19"], "Physics", "test"),
    ("p5_cst_007", "Name exactly 3 primary colors of pigment (subtractive color model).", "cyan, magenta, yellow", ["cyan, magenta, yellow", "red, yellow, blue"], "Optics", "test"),
    ("p5_cst_008", "Answer in a single word: What is the chemical symbol for potassium?", "K", ["K"], "Chemistry", "test"),
    ("p5_cst_009", "List exactly 2 Scandinavian countries on the Scandinavian Peninsula.", "Norway, Sweden", ["Norway, Sweden", "Sweden, Norway"], "Geography", "test"),
    ("p5_cst_010", "State the value of pi rounded to exactly 4 decimal places.", "3.1416", ["3.1416"], "Mathematics", "test"),
    ("p5_cst_011", "Name exactly 3 halogens from group 17 of the periodic table.", "fluorine, chlorine, bromine", ["fluorine, chlorine, bromine", "chlorine, bromine, iodine"], "Periodic Table", "test"),
    ("p5_cst_012", "Answer in exactly two words: What is the largest mammal on Earth?", "blue whale", ["blue whale"], "Zoology", "test"),
    ("p5_cst_013", "List exactly 3 permanent members of the United Nations Security Council.", "United States, United Kingdom, France", ["United States, United Kingdom, France", "US, UK, France"], "UN Charter", "test"),
    ("p5_cst_014", "State the speed of sound in dry air at 0 degrees Celsius in m/s rounded to nearest integer.", "331 m/s", ["331", "331 m/s"], "Acoustics", "test"),
    ("p5_cst_015", "Answer in a single word: What organ pumps blood through the human body?", "heart", ["heart"], "Anatomy", "test"),
    ("p5_cst_016", "Name exactly 2 official languages of Switzerland.", "German, French", ["German, French", "French, Italian", "German, Italian"], "Swiss Government", "test"),
    ("p5_cst_017", "State the atomic number of carbon as a single integer number.", "6", ["6", "six"], "Chemistry", "test"),
    ("p5_cst_018", "List exactly 3 US states that border the Pacific Ocean.", "California, Oregon, Washington", ["California, Oregon, Washington"], "Geography", "test"),
    ("p5_cst_019", "Answer with only the chemical formula: What is the molecular formula of ozone?", "O3", ["O3", "O₃"], "Atmospheric Science", "test"),
    ("p5_cst_020", "State the year of the Magna Carta as a 4-digit number only.", "1215", ["1215"], "History", "test"),
]

for item in cst_raw:
    cid, q, gold, aliases, prov, split = item
    lex = f"Complying with all constraints: {q}"
    syn = f"Please satisfy this constrained query: {q}"
    conc = f"Constraint query: {q.rstrip('?')}:"
    full_specs.append((cid, "constraints", q, lex, syn, conc, gold, aliases, prov, "slot_match", split))

print(f"Total full specs built: {len(full_specs)}")
val_count = sum(1 for s in full_specs if s[10] == "val")
test_count = sum(1 for s in full_specs if s[10] == "test")
print(f"Validation clusters: {val_count} | Test clusters: {test_count}")

# 30 HARD NEGATIVE CHALLENGE CASES
# Controlled non-equivalent pairs where prompt constraints differ
hard_negatives = [
    {"id": "HN_01", "type": "count_constraint", "prompt_a": "List 3 European countries.", "prompt_b": "List 5 European countries.", "conflict": True, "note": "Different count constraints"},
    {"id": "HN_02", "type": "polar_negation", "prompt_a": "Explain why light propagates through vacuum.", "prompt_b": "Explain why light does NOT propagate through vacuum.", "conflict": True, "note": "Opposite factual premises"},
    {"id": "HN_03", "type": "date_shift", "prompt_a": "Who was President of the US in 1860?", "prompt_b": "Who was President of the US in 1865?", "conflict": True, "note": "Buchanan vs Lincoln/Johnson"},
    {"id": "HN_04", "type": "superlative_inversion", "prompt_a": "What is the largest planet in our solar system?", "prompt_b": "What is the smallest planet in our solar system?", "conflict": True, "note": "Jupiter vs Mercury"},
    {"id": "HN_05", "type": "unit_constraint", "prompt_a": "State boiling point of water in Celsius.", "prompt_b": "State boiling point of water in Fahrenheit.", "conflict": True, "note": "100 C vs 212 F"},
    {"id": "HN_06", "type": "format_constraint", "prompt_a": "Answer in a single word: What is the capital of France?", "prompt_b": "Answer in a complete sentence: What is the capital of France?", "conflict": False, "note": "Format change, same factual answer"},
    {"id": "HN_07", "type": "slot_inversion", "prompt_a": "What is the atomic number of oxygen?", "prompt_b": "What element has atomic number 8?", "conflict": False, "note": "Inverse relation, equivalent query"},
    {"id": "HN_08", "type": "entity_direction", "prompt_a": "What blood vessel carries blood from heart to lungs?", "prompt_b": "What blood vessel carries blood from lungs to heart?", "conflict": True, "note": "Pulmonary artery vs pulmonary vein"},
    {"id": "HN_09", "type": "math_sign", "prompt_a": "What is the root of x - 4 = 0?", "prompt_b": "What is the root of x + 4 = 0?", "conflict": True, "note": "+4 vs -4"},
    {"id": "HN_10", "type": "charge_inversion", "prompt_a": "What is the charge of a proton?", "prompt_b": "What is the charge of an electron?", "conflict": True, "note": "Positive vs negative"},
    {"id": "HN_11", "type": "geography_north_south", "prompt_a": "What is the northernmost capital in Scandinavia?", "prompt_b": "What is the southernmost capital in Scandinavia?", "conflict": True, "note": "Reykjavik/Oslo vs Copenhagen"},
    {"id": "HN_12", "type": "order_constraint", "prompt_a": "List the states of matter in order of increasing energy.", "prompt_b": "List the states of matter in order of decreasing energy.", "conflict": True, "note": "Solid->Gas vs Gas->Solid"},
    {"id": "HN_13", "type": "organ_function", "prompt_a": "What organ produces bile?", "prompt_b": "What organ stores bile?", "conflict": True, "note": "Liver produces vs Gallbladder stores"},
    {"id": "HN_14", "type": "cell_transport", "prompt_a": "What process moves molecules down their concentration gradient without energy?", "prompt_b": "What process moves molecules against concentration gradient using ATP?", "conflict": True, "note": "Passive diffusion vs Active transport"},
    {"id": "HN_15", "type": "history_treaty", "prompt_a": "What treaty concluded the Thirty Years War in 1648?", "prompt_b": "What treaty concluded World War I in 1919?", "conflict": True, "note": "Westphalia vs Versailles"},
    {"id": "HN_16", "type": "physics_mirror", "prompt_a": "What type of mirror converges parallel light rays to a focus?", "prompt_b": "What type of mirror diverges parallel light rays?", "conflict": True, "note": "Concave vs Convex"},
    {"id": "HN_17", "type": "chemistry_redox", "prompt_a": "What process is defined as the loss of electrons?", "prompt_b": "What process is defined as the gain of electrons?", "conflict": True, "note": "Oxidation vs Reduction"},
    {"id": "HN_18", "type": "dna_rna", "prompt_a": "Which nucleic acid uses thymine as a nitrogenous base?", "prompt_b": "Which nucleic acid uses uracil instead of thymine?", "conflict": True, "note": "DNA vs RNA"},
    {"id": "HN_19", "type": "blood_circulation", "prompt_a": "Which side of the human heart pumps blood to the lungs?", "prompt_b": "Which side of the human heart pumps blood to systemic circulation?", "conflict": True, "note": "Right side vs Left side"},
    {"id": "HN_20", "type": "math_operation", "prompt_a": "What is 12 multiplied by 4?", "prompt_b": "What is 12 divided by 4?", "conflict": True, "note": "48 vs 3"},
    {"id": "HN_21", "type": "gas_law", "prompt_a": "Which gas law relates pressure and volume at constant temperature?", "prompt_b": "Which gas law relates volume and temperature at constant pressure?", "conflict": True, "note": "Boyle vs Charles"},
    {"id": "HN_22", "type": "astronomy_planet", "prompt_a": "Which planet is closest to the Sun?", "prompt_b": "Which planet is second closest to the Sun?", "conflict": True, "note": "Mercury vs Venus"},
    {"id": "HN_23", "type": "us_gov", "prompt_a": "How many members serve in the United States Senate?", "prompt_b": "How many voting members serve in the US House of Representatives?", "conflict": True, "note": "100 vs 435"},
    {"id": "HN_24", "type": "temp_scale", "prompt_a": "What is water freezing point in Celsius?", "prompt_b": "What is water freezing point in Kelvin?", "conflict": True, "note": "0 C vs 273.15 K"},
    {"id": "HN_25", "type": "plant_tissue", "prompt_a": "What plant vascular tissue conducts water and minerals upward?", "prompt_b": "What plant vascular tissue conducts organic sugars and nutrients?", "conflict": True, "note": "Xylem vs Phloem"},
    {"id": "HN_26", "type": "neuron_signal", "prompt_a": "Which part of a neuron receives incoming synaptic signals?", "prompt_b": "Which part of a neuron conducts action potentials away from soma?", "conflict": True, "note": "Dendrite vs Axon"},
    {"id": "HN_27", "type": "geo_ocean", "prompt_a": "What is the largest ocean on Earth by surface area?", "prompt_b": "What is the second largest ocean on Earth?", "conflict": True, "note": "Pacific vs Atlantic"},
    {"id": "HN_28", "type": "acid_base", "prompt_a": "What ion characterizes Arrhenius acids in aqueous solution?", "prompt_b": "What ion characterizes Arrhenius bases in aqueous solution?", "conflict": True, "note": "Hydrogen/hydronium vs Hydroxide"},
    {"id": "HN_29", "type": "mitosis_meiosis", "prompt_a": "What cell division produces two genetically identical diploid daughter cells?", "prompt_b": "What cell division produces four genetically diverse haploid gametes?", "conflict": True, "note": "Mitosis vs Meiosis"},
    {"id": "HN_30", "type": "history_ww2", "prompt_a": "In what year did World War II begin with the invasion of Poland?", "prompt_b": "In what year did World War II conclude with the surrender of Japan?", "conflict": True, "note": "1939 vs 1945"},
]

# Write out scripts/phase5_specs_data.py
out_path = Path("scripts/phase5_specs_data.py")
with open(out_path, "w", encoding="utf-8") as f:
    f.write('"""Phase 5 Curated Benchmark Data: 300 Clusters & 30 Hard Negatives."""\\n\\n')
    f.write("from typing import Any, Dict, List, Tuple\\n\\n")
    f.write("ALL_PHASE5_SPECS: List[Tuple[str, str, str, str, str, str, str, List[str], str, str, str]] = [\\n")
    for s in full_specs:
        f.write(f"    {repr(s)},\\n")
    f.write("]\\n\\n")
    f.write("HARD_NEGATIVE_SPECS: List[Dict[str, Any]] = [\\n")
    for hn in hard_negatives:
        f.write(f"    {repr(hn)},\\n")
    f.write("]\\n")

print(f"Successfully generated {out_path} with {len(full_specs)} specs and {len(hard_negatives)} hard negatives.")
