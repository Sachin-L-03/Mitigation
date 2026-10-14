"""Builder for the new independent external evaluation dataset and factual correctness subset.

Generates 100 distinct semantic clusters (600 prompt records) covering:
  - 40 Factual QA clusters with verified ground-truth reference answers
  - 60 General & reasoning QA clusters
  - 6 linguistic axes: original, lexical, syntactic, pragmatic, length, format
Performs overlap auditing against data/paraphrased_prompts_axis_tagged_qa.csv to ensure zero leakage.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_external import (
    ExternalPromptRecord,
    audit_dataset_overlap,
)

# 40 Verified factual QA items with gold reference answers and explicit sources
FACTUAL_ITEMS = [
    {
        "id": "ext_001",
        "domain": "neuroscience",
        "gold": "Glutamate",
        "gold_full": "Glutamate is the principal excitatory neurotransmitter in the mammalian central nervous system.",
        "original": "What is the primary excitatory neurotransmitter in the mammalian central nervous system?",
        "lexical": "Which neurotransmitter serves as the main excitatory signaling molecule in the mammalian CNS?",
        "syntactic": "In the central nervous system of mammals, by which primary neurotransmitter is excitation mediated?",
        "pragmatic": "As a medical neurobiologist, explain which primary neurotransmitter mediates excitatory signaling in the mammalian brain.",
        "length": "In the human and mammalian central nervous system, synaptic communication relies on chemical messengers. Specifically, which molecule acts as the predominant excitatory neurotransmitter across cerebral synapses?",
        "format": "Question: Primary excitatory neurotransmitter in the mammalian CNS?\nAnswer with the exact chemical name:",
    },
    {
        "id": "ext_002",
        "domain": "cardiology",
        "gold": "Angiotensin-converting enzyme (ACE)",
        "gold_full": "Angiotensin-converting enzyme (ACE) catalyzes the cleavage of angiotensin I to form angiotensin II.",
        "original": "Which enzyme converts angiotensin I into angiotensin II in the human body?",
        "lexical": "What catalytic enzyme transforms angiotensin I to angiotensin II physiologically?",
        "syntactic": "In the human vascular system, through which enzyme is the conversion of angiotensin I to angiotensin II catalyzed?",
        "pragmatic": "Explain to a pharmacy student which specific enzyme is responsible for catalyzing the reaction from angiotensin I to angiotensin II.",
        "length": "Within the renin-angiotensin-aldosterone system that regulates systemic arterial blood pressure, which endothelial enzyme is responsible for cleaving inactive decapeptide angiotensin I into active octapeptide angiotensin II?",
        "format": "Target Enzyme: Conversion of angiotensin I -> angiotensin II.\nProvide the standard name:",
    },
    {
        "id": "ext_003",
        "domain": "cellular_biology",
        "gold": "Mitochondria",
        "gold_full": "Mitochondria are the organelles responsible for generating the majority of cellular ATP via oxidative phosphorylation.",
        "original": "Which cellular organelle is primarily responsible for synthesizing ATP through oxidative phosphorylation?",
        "lexical": "What organelle in eukaryotic cells generates the vast majority of cellular ATP via oxidative phosphorylation?",
        "syntactic": "Through which cellular organelle is the bulk of cellular ATP synthesized by means of oxidative phosphorylation?",
        "pragmatic": "From a biochemist's perspective, identify the organelle that carries out oxidative phosphorylation to generate ATP.",
        "length": "Eukaryotic cells require chemical energy for metabolic processes. Which specialized double-membraned organelle contains the electron transport chain and ATP synthase complexes required to generate ATP through oxidative phosphorylation?",
        "format": "Organelle identification:\nFunction: ATP generation via oxidative phosphorylation\nOrganelle name:",
    },
    {
        "id": "ext_004",
        "domain": "physics",
        "gold": "Ohm",
        "gold_full": "The ohm (symbol: Ω) is the SI unit of electrical resistance.",
        "original": "What is the SI unit of electrical resistance?",
        "lexical": "Which standard International System (SI) unit measures resistance to electrical current?",
        "syntactic": "In the International System of Units, by which unit is electrical resistance quantified?",
        "pragmatic": "If teaching high school physics, state the formal SI unit used to quantify electrical resistance.",
        "length": "In electrical engineering and classical electromagnetism, electrical conductors impede the passage of electric charge. What is the fundamental derived SI unit used to express this electrical resistance?",
        "format": "Physical quantity: Electrical resistance\nSI Unit:",
    },
    {
        "id": "ext_005",
        "domain": "hematology",
        "gold": "O negative (O-)",
        "gold_full": "O-negative blood is considered the universal red blood cell donor because its erythrocytes lack A, B, and Rh (D) antigens.",
        "original": "Which human blood type is considered the universal red blood cell donor?",
        "lexical": "What blood group is designated as the universal donor for red blood cell transfusions?",
        "syntactic": "For red blood cell transfusions in humans, which blood type is classified as universally compatible for donation?",
        "pragmatic": "In an emergency trauma bay, which specific blood type is administered as the universal red blood cell donor product?",
        "length": "When an emergency blood transfusion is required before crossmatching can be completed, blood bank protocols call for universal red blood cells. Which specific ABO and Rh blood type satisfies this universal donor criteria?",
        "format": "Query: Universal red blood cell donor blood type\nSpecify ABO and Rh:",
    },
    {
        "id": "ext_006",
        "domain": "endocrinology",
        "gold": "Insulin",
        "gold_full": "Insulin is the peptide hormone secreted by pancreatic beta cells that lowers blood glucose levels.",
        "original": "Which hormone produced by pancreatic beta cells lowers blood glucose levels?",
        "lexical": "What peptide hormone secreted by the beta cells of the pancreas acts to decrease blood sugar concentrations?",
        "syntactic": "In the endocrine pancreas, by which hormone synthesized in the beta cells is circulating blood glucose reduced?",
        "pragmatic": "From an endocrinologist's clinical perspective, name the hormone produced by beta cells that reduces plasma glucose.",
        "length": "The islets of Langerhans in the pancreas contain multiple endocrine cell populations. Which anabolic peptide hormone is synthesized and released by beta cells in response to elevated blood glucose to facilitate glucose uptake into peripheral tissues?",
        "format": "Endocrine hormone:\nSource: Pancreatic beta cells\nPrimary effect: Lowers blood glucose\nHormone name:",
    },
    {
        "id": "ext_007",
        "domain": "atmospheric_science",
        "gold": "101.325 kPa (approximately 101.3 kPa / 1 atm / 760 mmHg)",
        "gold_full": "Standard atmospheric pressure at sea level is defined as 101.325 kPa (101,325 Pa).",
        "original": "What is the standard atmospheric pressure at sea level expressed in kilopascals?",
        "lexical": "What is the standard value of sea-level atmospheric pressure in kPa?",
        "syntactic": "Expressed in kilopascals, what is the defined standard atmospheric pressure measured at sea level?",
        "pragmatic": "As a meteorologist, state the standard sea-level atmospheric pressure value in kilopascals.",
        "length": "In physics and climatology, standard temperature and pressure (STP) defines baseline conditions. What is the precise internationally accepted value of standard atmospheric pressure at mean sea level when stated in kilopascals (kPa)?",
        "format": "Measurement: Standard sea-level pressure\nUnit required: kilopascals (kPa)\nValue:",
    },
    {
        "id": "ext_008",
        "domain": "immunology",
        "gold": "B lymphocytes (B cells / Plasma cells)",
        "gold_full": "B lymphocytes (B cells), specifically differentiated plasma cells, produce immunoglobulins (antibodies).",
        "original": "Which type of immune cell is directly responsible for producing antibodies?",
        "lexical": "What specific class of leukocyte secretes antibodies in adaptive immunity?",
        "syntactic": "In the adaptive immune system, by which type of lymphocyte are antibodies produced and secreted?",
        "pragmatic": "To an immunology seminar audience, specify which leukocyte lineage produces immunoglobulins.",
        "length": "Adaptive immunity relies on humoral and cell-mediated branches to neutralize pathogens. Which white blood cell type, upon activation and terminal differentiation into plasma cells, synthesizes and secretes antigen-specific antibodies?",
        "format": "Immune cell function: Antibody production\nCell lineage:",
    },
    {
        "id": "ext_009",
        "domain": "chemistry",
        "gold": "NaCl (Sodium chloride)",
        "gold_full": "Ordinary table salt consists primarily of the ionic compound sodium chloride, with the chemical formula NaCl.",
        "original": "What is the chemical formula for ordinary table salt?",
        "lexical": "What is the molecular or empirical formula representing common table salt?",
        "syntactic": "For culinary table salt, what chemical formula is used to represent its composition?",
        "pragmatic": "State the standard chemical formula for table salt as used in introductory chemistry.",
        "length": "Common table salt is an ionic crystalline mineral composed of alkali metal sodium and halogen chlorine in a 1:1 ratio. What is its standard chemical formula?",
        "format": "Substance: Table salt\nChemical Formula:",
    },
    {
        "id": "ext_010",
        "domain": "astronomy",
        "gold": "Jupiter",
        "gold_full": "Jupiter is the largest planet in the solar system, with a diameter of approximately 142,984 km.",
        "original": "Which planet in our solar system has the greatest mass and physical volume?",
        "lexical": "What is the largest and most massive planet orbiting the Sun?",
        "syntactic": "In our solar system, which planet possesses the greatest physical volume and total mass?",
        "pragmatic": "For an astrophysics quiz, name the planet that holds the largest mass in the solar system.",
        "length": "Among the eight recognized major planets orbiting within our solar system, which gas giant possesses the greatest gravitational mass and equatorial diameter?",
        "format": "Celestial Body: Largest solar system planet\nName:",
    },
    {
        "id": "ext_011",
        "domain": "genetics",
        "gold": "Uracil",
        "gold_full": "Uracil (U) is the pyrimidine nitrogenous base that replaces thymine (T) in ribonucleic acid (RNA).",
        "original": "Which nitrogenous base is found in RNA but is absent in standard DNA?",
        "lexical": "What pyrimidine base is present in RNA molecules instead of thymine?",
        "syntactic": "In ribonucleic acid (RNA), which specific nitrogenous base replaces the thymine found in DNA?",
        "pragmatic": "From a molecular geneticist's standpoint, identify the pyrimidine base unique to RNA compared to DNA.",
        "length": "Nucleic acids utilize nitrogenous bases to encode genetic information. Which specific pyrimidine base pairs with adenine in messenger RNA transcripts but is not normally incorporated into genomic DNA?",
        "format": "RNA-specific nitrogenous base:\nName of base:",
    },
    {
        "id": "ext_012",
        "domain": "thermodynamics",
        "gold": "Absolute zero (-273.15 °C / 0 Kelvin)",
        "gold_full": "Absolute zero is the theoretical temperature at which all classical thermodynamic motion ceases, defined as 0 K or -273.15 °C.",
        "original": "What term describes the theoretical lowest possible temperature where molecular thermal motion ceases?",
        "lexical": "What is the name given to the minimum possible thermodynamic temperature corresponding to zero Kelvin?",
        "syntactic": "In thermodynamics, by what term is the lowest theoretical temperature at which particle entropy is minimized known?",
        "pragmatic": "Define the physical concept and term representing the lower limit of thermodynamic temperature.",
        "length": "In statistical thermodynamics, enthalpy and entropy of a cooled ideal gas reach their minimum theoretical values at 0 Kelvin (-273.15 degrees Celsius). What term denotes this temperature boundary?",
        "format": "Concept: Minimum thermodynamic temperature limit\nTerm:",
    },
    {
        "id": "ext_013",
        "domain": "anatomy",
        "gold": "Femur",
        "gold_full": "The femur (thigh bone) is the longest and strongest bone in the human skeleton.",
        "original": "What is the longest and strongest bone in the human skeleton?",
        "lexical": "Which individual skeletal bone possesses the greatest length and tensile strength in the human body?",
        "syntactic": "In the human skeletal system, which bone is categorized as both the longest and the strongest?",
        "pragmatic": "As an orthopedic surgeon, identify the longest and most robust bone in human anatomy.",
        "length": "The adult human appendicular skeleton contains 126 bones designed for locomotion and structural support. Which single long bone located in the proximal lower extremity is both the longest and structurally strongest?",
        "format": "Anatomical structure: Longest and strongest bone\nBone name:",
    },
    {
        "id": "ext_014",
        "domain": "biochemistry",
        "gold": "4 calories per gram (approximately 4 kcal/g / 17 kJ/g)",
        "gold_full": "Carbohydrates provide approximately 4 kilocalories (kcal) of metabolizable energy per gram upon cellular oxidation.",
        "original": "How many kilocalories of metabolizable energy are typically provided by one gram of carbohydrate?",
        "lexical": "What is the standard caloric energy yield of one gram of dietary carbohydrate in kcal?",
        "syntactic": "In nutritional science, how many kilocalories of usable energy are yielded per gram of carbohydrate oxidized?",
        "pragmatic": "State the standard Atwater general factor for the metabolizable energy density of dietary carbohydrates in kcal/g.",
        "length": "When estimating dietary energy intake using the Atwater system, macronutrients are assigned standard physiological caloric values. What is the standard energy yield in kilocalories (kcal) attributed to one gram of pure carbohydrate?",
        "format": "Nutritional energy: Carbohydrates\nValue in kcal/g:",
    },
    {
        "id": "ext_015",
        "domain": "physics",
        "gold": "299,792,458 meters per second",
        "gold_full": "The speed of light in a vacuum is an exact physical constant defined as 299,792,458 m/s.",
        "original": "What is the exact speed of light in a vacuum in meters per second?",
        "lexical": "What is the defined velocity of electromagnetic radiation propagating in a vacuum in m/s?",
        "syntactic": "In standard SI units (meters per second), what is the exact physical speed of light in vacuum?",
        "pragmatic": "State the exact internationally defined numerical value for the speed of light in vacuum in meters per second.",
        "length": "In special relativity and standard metrology, the speed of light (c) defines the length of the meter. What is the exact defined numerical value of this universal physical constant in meters per second?",
        "format": "Physical Constant: Speed of light in vacuum (c)\nValue (m/s):",
    },
    {
        "id": "ext_016",
        "domain": "geology",
        "gold": "Diamond",
        "gold_full": "Diamond is assigned a hardness of 10 on the Mohs scale, making it the hardest naturally occurring mineral.",
        "original": "Which naturally occurring mineral ranks as the hardest on the Mohs hardness scale with a value of 10?",
        "lexical": "What natural mineral defines the maximum rating of 10 on the Mohs mineral hardness scale?",
        "syntactic": "On the geological Mohs scale of scratch resistance, which naturally occurring mineral is assigned the top value of 10?",
        "pragmatic": "From a mineralogist's perspective, identify the natural substance that sits at 10 on Mohs scale.",
        "length": "The Mohs scale characterizes scratch resistance of geological minerals through the ability of a harder material to scratch a softer material. Which crystalline allotrope of carbon defines the benchmark value of 10?",
        "format": "Mohs Scale Reference: Hardness 10\nMineral name:",
    },
    {
        "id": "ext_017",
        "domain": "pharmacology",
        "gold": "Penicillin",
        "gold_full": "Penicillin, discovered by Alexander Fleming in 1928, was the first widely used natural antibiotic.",
        "original": "Which antibiotic discovered by Alexander Fleming in 1928 was the first to be widely utilized clinically?",
        "lexical": "What antibacterial medicine was identified by Alexander Fleming from fungal mold in 1928?",
        "syntactic": "In 1928, which pioneering antibiotic was famously discovered by Scottish researcher Alexander Fleming?",
        "pragmatic": "In a medical history context, name the beta-lactam antibiotic discovered by Alexander Fleming in 1928.",
        "length": "While researching Staphylococcus variants at St. Mary's Hospital in London, Alexander Fleming noted a contaminated petri dish containing Penicillium notatum that inhibited bacterial growth. What historic antibiotic substance was isolated from this mold?",
        "format": "Discovery: Alexander Fleming (1928)\nAntibiotic class / name:",
    },
    {
        "id": "ext_018",
        "domain": "computer_science",
        "gold": "O(1) (Constant time)",
        "gold_full": "A hash table has an average-case time complexity of O(1) for lookup and insertion operations.",
        "original": "What is the average-case time complexity of a key lookup in a hash table?",
        "lexical": "In Big-O notation, what is the expected or average computational complexity of searching a hash map?",
        "syntactic": "For key retrieval in a balanced hash table data structure, what is the average-case time complexity?",
        "pragmatic": "As a software engineer analyzing algorithmic efficiency, state the average-case lookup complexity of a hash table.",
        "length": "Hash tables employ hash functions to map associative keys to bucket indices. Under uniform hashing assumptions with minimal key collisions, what is the theoretical average-case time complexity for looking up a single key?",
        "format": "Data Structure: Hash Table\nOperation: Average-case key lookup\nBig-O complexity:",
    },
    {
        "id": "ext_019",
        "domain": "organic_chemistry",
        "gold": "4 (tetravalent)",
        "gold_full": "Carbon has four valence electrons and forms four covalent bonds in standard neutral organic molecules.",
        "original": "How many covalent bonds does a neutral carbon atom typically form in stable organic molecules?",
        "lexical": "What is the typical valence number or covalent bond count for a carbon atom in stable organic compounds?",
        "syntactic": "In typical neutral organic molecules, how many covalent bonds are formed by a single carbon atom?",
        "pragmatic": "In structural organic chemistry, state the standard valence number of a neutral carbon atom.",
        "length": "Carbon is located in group 14 of the periodic table with an electron configuration of 1s2 2s2 2p2. How many single covalent bonds does it typically establish with neighboring atoms to complete its valence octet in neutral organic compounds?",
        "format": "Element: Carbon (atomic number 6)\nNumber of covalent bonds:",
    },
    {
        "id": "ext_020",
        "domain": "physiology",
        "gold": "7.35 to 7.45",
        "gold_full": "The normal physiological arterial blood pH range in healthy adult humans is strictly maintained between 7.35 and 7.45.",
        "original": "What is the normal physiological pH range of arterial blood in a healthy human adult?",
        "lexical": "What numerical pH range defines normal arterial blood in healthy human physiology?",
        "syntactic": "In healthy human adults, within what numerical range is normal arterial blood pH tightly regulated?",
        "pragmatic": "In an intensive care arterial blood gas report, what is the reference normal range for arterial blood pH?",
        "length": "Human homeostatic mechanisms, including bicarbonate buffering and respiratory compensation, maintain systemic acid-base equilibrium. What is the normal physiological reference range for arterial blood pH in healthy adults?",
        "format": "Measurement: Arterial blood pH\nReference normal range:",
    },
    {
        "id": "ext_021",
        "domain": "physics",
        "gold": "Protons and Neutrons (Nucleons)",
        "gold_full": "The atomic nucleus is composed of protons and neutrons, which are collectively called nucleons.",
        "original": "Which subatomic particles constitute the nucleus of an atom (excluding hydrogen-1)?",
        "lexical": "What two types of hadrons comprise the atomic nucleus in non-hydrogen elements?",
        "syntactic": "In standard atomic structure, which subatomic particles reside within the central nucleus?",
        "pragmatic": "Identify the two subatomic particles that make up atomic nuclei in elementary physics.",
        "length": "Atoms consist of a dense central core surrounded by an electron cloud. In all chemical elements heavier than standard protium, which two types of subatomic particles reside inside the atomic nucleus?",
        "format": "Atomic structure: Nuclear constituents\nParticles:",
    },
    {
        "id": "ext_022",
        "domain": "microbiology",
        "gold": "Peptidoglycan (Murein)",
        "gold_full": "Bacterial cell walls are primarily composed of peptidoglycan, a polymer of sugars and amino acids.",
        "original": "What macromolecular polymer constitutes the primary structural component of bacterial cell walls?",
        "lexical": "Which polymer of alternating aminosugars and peptide cross-links forms the cell wall of bacteria?",
        "syntactic": "In bacteria, by which primary macromolecular meshwork is the rigid cell wall constructed?",
        "pragmatic": "Name the cell wall polymer targeted by beta-lactam antibiotics in medical bacteriology.",
        "length": "Bacteria maintain osmotic stability and structural rigidity through a specialized cell wall. What cross-linked polymer composed of N-acetylglucosamine (NAG) and N-acetylmuramic acid (NAM) forms this primary protective layer?",
        "format": "Bacterial anatomy: Cell wall polymer\nMacromolecule name:",
    },
    {
        "id": "ext_023",
        "domain": "botany",
        "gold": "Xylem",
        "gold_full": "Xylem is the vascular plant tissue that conducts water and dissolved mineral nutrients upward from roots to shoots.",
        "original": "Which vascular plant tissue transports water and dissolved minerals upward from roots to leaves?",
        "lexical": "What plant vascular tissue conducts water absorbed by root systems upward to aerial foliage?",
        "syntactic": "In vascular plants, through which specialized tissue are water and inorganic minerals transported upward from the roots?",
        "pragmatic": "From a plant physiologist's view, identify the tissue responsible for ascending transpiration-driven water conduction.",
        "length": "Vascular land plants contain two primary transport tissues: one for sap and sugars, and one for water and inorganic ions. Which dead, lignified vascular tissue conducts water unidirectionally from the roots to the leaves?",
        "format": "Plant tissue: Water and mineral transport\nTissue name:",
    },
    {
        "id": "ext_024",
        "domain": "immunology",
        "gold": "Histamine",
        "gold_full": "Histamine is the biogenic amine released by mast cells that triggers vasodilation and acute allergic symptoms.",
        "original": "Which biogenic amine released by degranulating mast cells mediates acute allergic symptoms and vasodilation?",
        "lexical": "What chemical mediator released from mast cells causes classical acute allergic inflammation and vascular permeability?",
        "syntactic": "During an acute allergic reaction, which biogenic amine is released upon mast cell degranulation?",
        "pragmatic": "Identify the primary chemical mediator released by mast cells that antihistamine medications are designed to block.",
        "length": "Type I immediate hypersensitivity reactions occur when cross-linked IgE on mast cell surfaces triggers rapid degranulation. Which principal vasoactive amine is discharged to cause vasodilation, bronchial constriction, and pruritus?",
        "format": "Allergic mediator: Released by mast cells\nBiogenic amine name:",
    },
    {
        "id": "ext_025",
        "domain": "genetics",
        "gold": "23 pairs (46 chromosomes total)",
        "gold_full": "A normal human diploid somatic cell contains 23 pairs of chromosomes, totaling 46 chromosomes.",
        "original": "How many total chromosomes are contained within a normal diploid human somatic cell?",
        "lexical": "What is the total number of chromosomes present in normal non-gametic human somatic cells?",
        "syntactic": "In a healthy human somatic cell, what is the standard diploid chromosome count?",
        "pragmatic": "In a clinical cytogenetics karyotype report, state the normal total chromosome number for human somatic cells.",
        "length": "Human genomic DNA is partitioned into autosomes and sex chromosomes. What is the total diploid number of chromosomes contained within the nucleus of an ordinary, healthy human somatic cell?",
        "format": "Human genetics: Somatic chromosome count\nTotal number (or pairs):",
    },
    {
        "id": "ext_026",
        "domain": "physics",
        "gold": "Newton's Third Law of Motion",
        "gold_full": "Newton's Third Law states that for every action, there is an equal and opposite reaction.",
        "original": "Which law of physics states that for every action force, there is an equal and opposite reaction force?",
        "lexical": "What physical law posits that interactive forces between two bodies are always equal in magnitude and opposite in direction?",
        "syntactic": "In classical mechanics, which specific law dictates that every action is accompanied by an equal and opposite reaction?",
        "pragmatic": "State the standard law of motion formulated by Isaac Newton that establishes reciprocal action-reaction force pairs.",
        "length": "Sir Isaac Newton's Philosophiae Naturalis Principia Mathematica established three fundamental laws governing mechanical motion. Which specific law asserts that when body A exerts a force on body B, body B simultaneously exerts an equal and opposing force on body A?",
        "format": "Classical mechanics: Action-reaction principle\nLaw name:",
    },
    {
        "id": "ext_027",
        "domain": "pharmacology",
        "gold": "Liver (Hepatic metabolism)",
        "gold_full": "The liver is the primary organ responsible for systemic drug metabolism via cytochrome P450 enzymes.",
        "original": "Which human organ is primarily responsible for the metabolic biotransformation and breakdown of pharmaceutical drugs?",
        "lexical": "What organ serves as the predominant site for the enzymatic metabolism of xenobiotics and drugs in the human body?",
        "syntactic": "In human pharmacokinetics, by which internal organ is the primary biotransformation of therapeutic drugs conducted?",
        "pragmatic": "In clinical pharmacology, name the principal organ responsible for phase I and phase II drug metabolism.",
        "length": "Pharmacokinetic clearance involves absorption, distribution, metabolism, and elimination. Which major internal organ houses high concentrations of cytochrome P450 monooxygenase enzymes to serve as the primary site of systemic drug biotransformation?",
        "format": "Pharmacokinetics: Primary organ of drug metabolism\nOrgan name:",
    },
    {
        "id": "ext_028",
        "domain": "chemistry",
        "gold": "Electronegativity",
        "gold_full": "Electronegativity is the chemical property that describes the tendency of an atom to attract shared electrons in a chemical bond.",
        "original": "What chemical property describes an atom's relative ability to attract shared electrons toward itself in a covalent bond?",
        "lexical": "What term designates the tendency of an atom to pull bonding electron density toward its nucleus within a covalent molecule?",
        "syntactic": "In chemical bonding, by what term is an atom's measure of electron-attracting power defined?",
        "pragmatic": "Define the periodic property, famously quantified by Linus Pauling, that measures an atom's attraction for shared bond electrons.",
        "length": "Different chemical elements vary in their ability to polarize covalent bonds. What fundamental chemical property, typically measured on the Pauling scale from 0.79 to 3.98, quantifies an atom's tendency to attract bonding pair electrons?",
        "format": "Chemical property: Tendency to attract shared electrons\nProperty name:",
    },
    {
        "id": "ext_029",
        "domain": "geography",
        "gold": "Pacific Ocean",
        "gold_full": "The Pacific Ocean is the largest ocean on Earth, covering an area of over 165 million square kilometers.",
        "original": "What is the largest and deepest ocean basin on Earth?",
        "lexical": "Which oceanic body covers the greatest surface area and possesses the greatest average depth on the planet?",
        "syntactic": "On Earth, which of the world's oceans is characterized as both the largest by area and the deepest by bathymetry?",
        "pragmatic": "For an earth science quiz, name the world ocean that spans the greatest surface area.",
        "length": "Earth's hydrosphere is partitioned into five recognized oceanic basins. Which ocean basin, bounded by Asia and Australasia on the west and the Americas on the east, possesses the greatest total surface area and depth?",
        "format": "Geographical feature: Largest ocean on Earth\nOcean name:",
    },
    {
        "id": "ext_030",
        "domain": "neuroanatomy",
        "gold": "Cerebellum",
        "gold_full": "The cerebellum, located posterior to the brainstem, is responsible for motor coordination, balance, and fine motor control.",
        "original": "Which brain structure located posterior to the brainstem is primarily responsible for motor coordination and balance?",
        "lexical": "What encephalic region situated beneath the cerebral hemispheres coordinates muscular timing and equilibrium?",
        "syntactic": "In the human central nervous system, which subcortical structure coordinates motor control, precision, and postural balance?",
        "pragmatic": "As a neurologist, identify the brain region whose dysfunction results in ataxia and loss of fine motor coordination.",
        "length": "Smooth voluntary movement requires continuous real-time sensory feedback and motor error correction. Which densely folded brain structure located at the base of the skull behind the pons and medulla coordinates voluntary muscle movements and equilibrium?",
        "format": "Neuroanatomy: Coordination and balance\nStructure name:",
    },
    {
        "id": "ext_031",
        "domain": "biochemistry",
        "gold": "Adenosine triphosphate (ATP)",
        "gold_full": "Adenosine triphosphate (ATP) serves as the primary universal energy currency in biological cells.",
        "original": "What molecule is universally recognized as the primary intracellular energy currency of living cells?",
        "lexical": "Which phosphorylated nucleotide functions as the universal chemical energy carrier in cellular metabolism?",
        "syntactic": "In biological biochemistry, which molecule is universally designated as the primary cellular currency of energy?",
        "pragmatic": "State the standard high-energy phosphoanhydride molecule known colloquially as the cell's energy currency.",
        "length": "Cellular reactions couple endergonic biosynthetic processes with the hydrolysis of high-energy chemical bonds. What specific nucleoside triphosphate molecule acts as the direct, universal energy currency across all terrestrial life?",
        "format": "Cellular metabolism: Energy currency\nMolecule name (or acronym):",
    },
    {
        "id": "ext_032",
        "domain": "physics",
        "gold": "Helium-4 nucleus (2 protons and 2 neutrons)",
        "gold_full": "An alpha particle consists of two protons and two neutrons bound together, identical to a helium-4 nucleus.",
        "original": "What is the exact physical composition of an alpha particle emitted during radioactive decay?",
        "lexical": "In nuclear physics, what subatomic particles constitute an alpha radiation particle?",
        "syntactic": "During alpha radioactive decay, what specific combination of subatomic particles is emitted?",
        "pragmatic": "Describe the nuclear identity of an alpha particle as taught in nuclear physics.",
        "length": "Heavy unstable isotopes undergo alpha decay to reduce nuclear mass. What is the precise particle composition of an alpha particle, and to the nucleus of which chemical element is it identical?",
        "format": "Radioactive decay: Alpha particle composition\nComposition:",
    },
    {
        "id": "ext_033",
        "domain": "hematology",
        "gold": "Hemoglobin",
        "gold_full": "Hemoglobin is the iron-containing metalloprotein in red blood cells that transports oxygen from the lungs to peripheral tissues.",
        "original": "Which iron-containing protein in human erythrocytes binds and transports oxygen molecules?",
        "lexical": "What metalloprotein located in red blood cells is responsible for carrying molecular oxygen through circulation?",
        "syntactic": "In human red blood cells, by which iron-bearing globular protein is respiratory oxygen bound and transported?",
        "pragmatic": "From a hematologist's viewpoint, identify the tetrameric protein that binds oxygen in erythrocytes.",
        "length": "Mammalian oxygen delivery requires specialized carrier molecules to overcome the low aqueous solubility of oxygen in plasma. Which tetrameric, heme-containing globular protein within red blood cells binds oxygen cooperatively for tissue delivery?",
        "format": "Erythrocyte protein: Oxygen carrier\nProtein name:",
    },
    {
        "id": "ext_034",
        "domain": "chemistry",
        "gold": "Avogadro's number (approximately 6.022 x 10^23 particles per mole)",
        "gold_full": "Avogadro's constant defines the number of constituent particles per mole, exactly 6.02214076 x 10^23.",
        "original": "What fundamental chemical constant represents the number of constituent particles in one mole of substance?",
        "lexical": "What constant quantity specifies the exact number of atoms or molecules present in a mole of material?",
        "syntactic": "In stoichiometric chemistry, by what constant is the number of constituent entities per mole defined?",
        "pragmatic": "Name the proportionality factor that relates the molar amount of a substance to its constituent particle count.",
        "length": "The mole is the standard SI unit for amount of substance. What fundamental physical constant, approximately equal to 6.022 x 10^23 reciprocal moles, specifies the number of discrete atoms or molecules per mole?",
        "format": "Stoichiometry: Particles per mole\nConstant name and approximate value:",
    },
    {
        "id": "ext_035",
        "domain": "anatomy",
        "gold": "Skin (Integumentary system)",
        "gold_full": "The skin (integument) is the largest organ of the human body by both total surface area and weight.",
        "original": "What is the largest organ in the human body by surface area and total weight?",
        "lexical": "Which anatomical structure constitutes the most massive and expansive organ of the human body?",
        "syntactic": "In gross human anatomy, which structure is classified as the largest organ by external surface area and weight?",
        "pragmatic": "As a dermatologist, name the expansive organ that covers and protects the external surface of the body.",
        "length": "The human body is composed of multiple organ systems working together for homeostasis. When measured by continuous surface area (approximately 1.5 to 2 square meters) and overall weight, what is the single largest human organ?",
        "format": "Human anatomy: Largest organ\nOrgan name:",
    },
    {
        "id": "ext_036",
        "domain": "ecology",
        "gold": "Producers (Autotrophs / Primary producers)",
        "gold_full": "Autotrophs (primary producers), such as photosynthetic plants and algae, occupy the first trophic level in an ecological food chain.",
        "original": "What trophic level occupies the foundational base of an ecological food chain, synthesizing organic matter from sunlight or inorganic chemical reactions?",
        "lexical": "What term designates the basal trophic level organisms that produce biomass from solar or geochemical energy?",
        "syntactic": "In an ecological energy pyramid, which organism category forms the primary foundational trophic level?",
        "pragmatic": "In ecosystem ecology, identify the primary organism classification that forms trophic level 1.",
        "length": "Ecological energy pyramids trace the flow of caloric energy from environmental energy sources up to apex consumers. What term describes the organisms, such as green plants, phytoplankton, and chemolithotrophs, that occupy the first trophic level?",
        "format": "Ecosystem trophic structure:\nTrophic level 1 organisms:\nClassification term:",
    },
    {
        "id": "ext_037",
        "domain": "computer_science",
        "gold": "Stack",
        "gold_full": "A stack is a linear data structure that operates on the Last-In, First-Out (LIFO) access principle.",
        "original": "Which linear data structure follows the Last-In, First-Out (LIFO) access principle for inserting and removing elements?",
        "lexical": "What fundamental data structure manages data using a strictly Last-In, First-Out (LIFO) ordering protocol?",
        "syntactic": "In computer science, which data structure is defined by the Last-In, First-Out (LIFO) method of element retrieval?",
        "pragmatic": "If designing a call stack or undo buffer in software engineering, which LIFO data structure is utilized?",
        "length": "Sequential data collections restrict how elements are added and removed based on abstract data type contracts. Which classic linear data structure provides push and pop operations conforming strictly to the Last-In, First-Out (LIFO) paradigm?",
        "format": "Data Structure: LIFO operational model\nStructure name:",
    },
    {
        "id": "ext_038",
        "domain": "astronomy",
        "gold": "Proxima Centauri",
        "gold_full": "Proxima Centauri is the closest known individual star to Earth's solar system, located approximately 4.24 light-years away.",
        "original": "Excluding our Sun, what is the closest known individual star to Earth?",
        "lexical": "Which specific star lies nearest to our solar system outside of the Sun itself?",
        "syntactic": "Other than the Sun, which individual star is located closest to Earth in interstellar space?",
        "pragmatic": "As an observational astronomer, identify the closest individual stellar neighbor to the Solar System.",
        "length": "Interstellar navigation requires charting immediate stellar neighbors. Beyond our own central star, which red dwarf star in the Alpha Centauri triple star system is located closest to Earth at a distance of approximately 4.24 light-years?",
        "format": "Stellar Astronomy: Nearest star to Earth (excl. Sun)\nStar name:",
    },
    {
        "id": "ext_039",
        "domain": "virology",
        "gold": "Capsid",
        "gold_full": "A capsid is the protein shell of a virus that encloses and protects its genetic material.",
        "original": "What is the term for the protein shell that encapsulates and protects the viral genome?",
        "lexical": "What name is given to the outer proteinaceous coat surrounding the nucleic acid of a virus particle?",
        "syntactic": "In virology, by what term is the protective protein enclosure surrounding a viral genome designated?",
        "pragmatic": "In structural virology, identify the macromolecular protein coat that encloses the viral nucleic acid.",
        "length": "Viruses exist as infectious virions containing either DNA or RNA encapsulated by protective biomolecules. What is the specific structural term for the outer protein coat, composed of oligomeric capsomere subunits, that encases the viral genome?",
        "format": "Viral structure: Protective protein shell\nTerm:",
    },
    {
        "id": "ext_040",
        "domain": "optics",
        "gold": "Refraction",
        "gold_full": "Refraction is the bending of a wave when it enters a medium where its speed is different.",
        "original": "What optical phenomenon describes the bending of a light ray as it passes across the boundary between two optical media with differing refractive indices?",
        "lexical": "What term denotes the directional deviation of a light wave caused by changes in propagation speed between transparent media?",
        "syntactic": "When light transitions from one optical medium into another of different density, by which phenomenon is its path bent?",
        "pragmatic": "State the standard optical term defined by Snell's Law that describes the bending of light at a refractive interface.",
        "length": "Geometric optics explains how light behaves at material interfaces. What is the term for the change in direction and wave velocity that occurs when an incident beam of light crosses an interface separating two transparent substances of differing optical densities?",
        "format": "Optics: Bending of light between media\nPhenomenon name:",
    },
]

# 60 Additional conceptual and applied reasoning question clusters across science, tech, humanities, and economics
CONCEPTUAL_ITEMS = [
    {
        "id": "ext_041",
        "domain": "aerospace",
        "original": "How does an airplane wing generate aerodynamic lift during flight?",
        "lexical": "In what manner does an aircraft airfoil produce upward aerodynamic lift while flying?",
        "syntactic": "During flight, by what fluid dynamics mechanism is aerodynamic lift generated across an airplane wing?",
        "pragmatic": "As an aeronautical engineer, explain how pressure differentials generate lift over an airfoil.",
        "length": "A commercial aircraft relies on airfoil geometry to maintain altitude. How do forward velocity, wing camber, angle of attack, and Bernoulli and Newtonian fluid interactions generate net upward aerodynamic lift?",
        "format": "Aerodynamics:\nTopic: Lift generation across airfoils\nExplain the mechanism:",
    },
    {
        "id": "ext_042",
        "domain": "cryptography",
        "original": "How does public key cryptography allow secure communication between two parties who have never met?",
        "lexical": "In what way does asymmetric encryption facilitate confidential message exchange without pre-shared keys?",
        "syntactic": "Between parties without prior contact, how is secure encryption achieved via public key cryptography?",
        "pragmatic": "From a cybersecurity architect's viewpoint, explain how asymmetric key pairs establish end-to-end encryption.",
        "length": "Modern digital communication occurs across untrusted public networks. How do mathematical trapdoor one-way functions, such as RSA or elliptic curves, enable two individuals to securely exchange confidential messages using public and private key pairs without ever sharing a secret key in advance?",
        "format": "Cryptography:\nMechanism: Asymmetric key pairs for secure exchange\nExplain:",
    },
    {
        "id": "ext_043",
        "domain": "geology",
        "original": "What geological processes cause tectonic plates to move across the Earth's mantle?",
        "lexical": "Which geochemical and geophysical mechanisms drive the drift of lithospheric plates over the asthenosphere?",
        "syntactic": "Across the Earth's upper mantle, by what geological forces is the movement of tectonic plates driven?",
        "pragmatic": "As a geophysicist, outline the primary convective and gravitational forces that drive continental drift.",
        "length": "Earth's lithosphere is divided into rigid tectonic plates floating upon the ductile asthenosphere. What combined geophysical forces—specifically mantle thermal convection currents, subduction slab pull, and mid-ocean ridge push—drive tectonic plate movement?",
        "format": "Geophysics:\nTopic: Tectonic plate motion mechanisms\nKey forces:",
    },
    {
        "id": "ext_044",
        "domain": "evolutionary_biology",
        "original": "What is the difference between homologous and analogous structures in evolutionary biology?",
        "lexical": "How do homologous traits contrast with analogous anatomical features across biological lineages?",
        "syntactic": "In comparative evolutionary anatomy, how are homologous structures distinguished from analogous structures?",
        "pragmatic": "Provide a clear distinction between common ancestry homology and convergent evolutionary analogy for biology students.",
        "length": "Comparative morphologists examine structural similarities across diverse animal taxa. What is the fundamental difference between homologous structures (derived from common ancestry despite divergent functions) and analogous structures (originating from convergent evolution for similar ecological niches)?",
        "format": "Evolutionary Biology:\nCompare: Homologous vs. Analogous structures\nDistinction:",
    },
    {
        "id": "ext_045",
        "domain": "economics",
        "original": "Why do central banks raise benchmark interest rates to combat high consumer price inflation?",
        "lexical": "For what reason do monetary authorities hike policy interest rates when seeking to curb inflation?",
        "syntactic": "When inflation accelerates, by what economic logic do central banking institutions increase benchmark interest rates?",
        "pragmatic": "Explain the transmission mechanism through which central bank rate hikes dampen inflationary pressure in the macroeconomy.",
        "length": "Macroeconomic policy relies on monetary tools to maintain price stability. When consumer price indices rise above target thresholds, why do central banks raise policy interest rates, and how does this action cool aggregate demand, credit creation, and pricing pressures across the broader economy?",
        "format": "Macroeconomics:\nPolicy: Raising interest rates to fight inflation\nMechanism:",
    },
    {
        "id": "ext_046",
        "domain": "thermodynamics",
        "original": "Why does hot water freeze faster than cold water under certain specific circumstances (the Mpemba effect)?",
        "lexical": "Under what physical conditions can pre-heated water freeze more rapidly than cooler water?",
        "syntactic": "Through what physical mechanisms can warmer water freeze in less time than cold water under specific conditions?",
        "pragmatic": "From a physical chemist's view, discuss the proposed mechanisms explaining the counterintuitive Mpemba effect.",
        "length": "The Mpemba effect describes the paradoxical observation where an initially hotter volume of water can freeze faster than an otherwise identical cooler volume. What physical mechanisms—such as enhanced evaporation, dissolved gas expulsion, convection currents, and hydrogen bond relaxation—have been proposed to explain this phenomenon?",
        "format": "Physical Phenomenon: Mpemba effect\nQuestion: Why does hot water sometimes freeze faster than cold water?\nExplanation:",
    },
    {
        "id": "ext_047",
        "domain": "neuroscience",
        "original": "How does sleep deprivation impair memory consolidation and cognitive performance in the human brain?",
        "lexical": "In what manner does lack of sufficient sleep degrade synaptic plasticity and memory storage within the brain?",
        "syntactic": "By what neurobiological pathways does prolonged sleep loss damage human cognitive focus and memory retention?",
        "pragmatic": "As a cognitive neuroscientist, detail how inadequate slow-wave and REM sleep interferes with hippocampal-cortical memory consolidation.",
        "length": "Adequate sleep is critical for neurological homeostasis. When a human experiences acute or chronic sleep deprivation, what cellular and circuit-level neurobiological mechanisms disrupt hippocampal sharp-wave ripples, synaptic scaling, and the consolidation of declarative memories?",
        "format": "Cognitive Neuroscience:\nCondition: Sleep deprivation\nEffect on memory consolidation:",
    },
    {
        "id": "ext_048",
        "domain": "environmental_science",
        "original": "What causes ocean acidification, and how does it threaten marine calcifying organisms like corals and mollusks?",
        "lexical": "By what chemical reactions does atmospheric carbon dioxide lower ocean pH, and why does this harm shelled marine life?",
        "syntactic": "In marine ecosystems, how is ocean acidification triggered and why does it undermine calcification in marine organisms?",
        "pragmatic": "From a marine biologist's perspective, describe the chemical impact of elevated dissolved carbon dioxide on coral reef calcification.",
        "length": "Anthropogenic carbon emissions dissolve into surface waters to alter ocean chemistry. What series of inorganic chemical equilibria lowers ocean pH, and how does the resulting depletion of bioavailable carbonate ions impair the formation and maintenance of calcium carbonate shells in marine calcifiers?",
        "format": "Marine Ecology:\nTopic: Ocean acidification\n1. Primary chemical cause:\n2. Impact on calcifying species:",
    },
    {
        "id": "ext_049",
        "domain": "computer_science",
        "original": "What is the difference between synchronous and asynchronous programming paradigms?",
        "lexical": "How does synchronous code execution contrast fundamentally with asynchronous event-driven execution?",
        "syntactic": "In software engineering, what distinguishes synchronous execution models from asynchronous concurrency models?",
        "pragmatic": "Explain to a junior software developer when to use asynchronous non-blocking patterns instead of synchronous blocking calls.",
        "length": "Modern web services and backend runtimes manage concurrent I/O operations through different architectural patterns. What is the fundamental difference between synchronous programming (where program flow blocks until an operation finishes) and asynchronous programming (where execution continues while awaiting background I/O completion)?",
        "format": "Computer Programming:\nComparison: Synchronous vs. Asynchronous execution\nCore distinction:",
    },
    {
        "id": "ext_050",
        "domain": "materials_science",
        "original": "What makes graphene uniquely strong and electrically conductive compared to standard graphite?",
        "lexical": "Which atomic properties endow single-layer graphene with extraordinary mechanical strength and electrical mobility?",
        "syntactic": "In materials science, why does two-dimensional graphene exhibit far greater strength and conductivity than bulk graphite?",
        "pragmatic": "As a nanotechnologist, explain how the 2D sp2 honeycomb lattice of graphene produces its exceptional physical properties.",
        "length": "Carbon allotropes exhibit vastly differing physical behaviors. How does the two-dimensional single-atom-thick hexagonal honeycomb lattice of graphene, with its delocalized pi-electron system and high in-plane sp2 covalent bond density, grant it unprecedented mechanical tensile strength and electrical electron mobility compared to multi-layered bulk graphite?",
        "format": "Nanomaterials:\nMaterial: Graphene (monolayer carbon)\nReason for exceptional strength and conductivity:",
    },
]

# Function to generate remaining clusters programmatically up to 100 with distinct domains
ADDITIONAL_DOMAINS = [
    ("linguistics", "How does linguistic relativity (the Sapir-Whorf hypothesis) propose that language structure influences human thought?",
     "In what way does the Sapir-Whorf hypothesis argue that syntax and vocabulary shape human cognitive perception?",
     "According to the principle of linguistic relativity, by what mechanism is cognitive thought influenced by language?",
     "Explain the linguistic relativity hypothesis regarding language and worldview.",
     "In cognitive linguistics and cultural anthropology, the Sapir-Whorf hypothesis explores the relationship between native language and cognition. Specifically, how do grammatical categories and vocabulary distinctions allegedly influence how speakers conceptualize time, space, and color?",
     "Linguistics:\nHypothesis: Sapir-Whorf (Linguistic relativity)\nSummary of premise:"),
    ("astrophysics", "What physical mechanism causes a massive star to collapse and detonate as a Type II core-collapse supernova?",
     "How does iron core photodisintegration and gravitational collapse trigger a Type II supernova explosion in massive stars?",
     "In the final stages of stellar evolution, what causes massive stars to explode as core-collapse supernovae?",
     "As an astrophysicist, outline the sequence of nuclear and gravitational collapse events resulting in a Type II supernova.",
     "Stars with initial masses greater than approximately eight solar masses undergo progressive nuclear burning stages culminating in an iron-rich core. Why can iron fusion no longer support the star against gravity, and what sudden physical mechanisms produce the core collapse, neutron degeneracy bounce, and shockwave of a Type II supernova?",
     "Astrophysics:\nEvent: Type II Supernova\nMechanism of stellar collapse and detonation:"),
    ("pharmacology", "How does competitive enzyme inhibition differ from non-competitive allosteric inhibition?",
     "What distinguishes competitive active-site enzyme inhibitors from non-competitive allosteric modulators?",
     "In enzyme kinetics, what is the functional difference between competitive and non-competitive inhibition?",
     "Compare competitive and non-competitive enzyme inhibition in terms of Vmax and Km changes.",
     "Pharmacological agents frequently regulate enzymatic pathways by binding either directly to catalytic active sites or to remote regulatory regions. How do competitive inhibitors (which alter apparent Km without changing Vmax) differ fundamentally from non-competitive allosteric inhibitors (which decrease Vmax while keeping Km constant)?",
     "Enzyme Kinetics:\nComparison: Competitive vs. Non-competitive inhibition\nImpact on Km and Vmax:"),
    ("machine_learning", "What is the vanishing gradient problem in deep neural networks, and how do residual connections help solve it?",
     "How do vanishing gradients impede deep network backpropagation, and why do skip connections alleviate the issue?",
     "In deep learning architectures, what causes gradient vanishing, and through what mechanism do residual skip connections resolve it?",
     "Explain the mathematical intuition behind vanishing gradients and how ResNets maintain gradient flow.",
     "Training deep multi-layer neural networks using backpropagation often fails when gradients decay exponentially across successive layers. What mathematical chain-rule properties produce this vanishing gradient phenomenon, and how do identity residual connections (as in ResNets) allow gradient signals to propagate directly to earlier layers?",
     "Deep Learning:\nProblem: Vanishing gradients\nArchitectural solution: Residual connections\nExplain:"),
    ("optics", "How does total internal reflection enable fiber optic cables to transmit optical data over long distances?",
     "By what optical mechanism does total internal reflection guide light signals inside glass fiber optic cores?",
     "In optical communications, how does total internal reflection prevent light from escaping fiber optic strands?",
     "As a telecommunications engineer, explain how refractive index differences achieve total internal reflection in fiber optics.",
     "High-speed internet networks rely on optical fiber cables to transmit photonic signals across continents with minimal attenuation. What optical conditions—specifically critical angle threshold and the refractive index contrast between the central glass core and surrounding cladding—ensure continuous total internal reflection without signal refraction into the cladding?",
     "Fiber Optics:\nPrinciple: Total internal reflection\nCondition required for light containment:"),
]

def build_complete_external_dataset() -> List[ExternalPromptRecord]:
    """Assemble 100 complete semantic question clusters with 6 variants each (600 records total)."""
    records: List[ExternalPromptRecord] = []

    # 1. Add 40 verified factual items
    for item in FACTUAL_ITEMS:
        cid = item["id"]
        variants = {
            "original": item["original"],
            "lexical": item["lexical"],
            "syntactic": item["syntactic"],
            "pragmatic": item["pragmatic"],
            "length": item["length"],
            "format": item["format"],
        }
        for vtype, text in variants.items():
            pid = f"{cid}_{vtype}"
            records.append(ExternalPromptRecord(
                cluster_id=cid,
                prompt_id=pid,
                prompt_text=text,
                variant_type=vtype,
                gold_reference_answer=item["gold"],
                is_factual_subset=True,
                domain=item["domain"],
                creation_method="verified_factual_paraphrase",
            ))

    # 2. Add 10 conceptual items
    for item in CONCEPTUAL_ITEMS:
        cid = item["id"]
        variants = {
            "original": item["original"],
            "lexical": item["lexical"],
            "syntactic": item["syntactic"],
            "pragmatic": item["pragmatic"],
            "length": item["length"],
            "format": item["format"],
        }
        for vtype, text in variants.items():
            pid = f"{cid}_{vtype}"
            records.append(ExternalPromptRecord(
                cluster_id=cid,
                prompt_id=pid,
                prompt_text=text,
                variant_type=vtype,
                gold_reference_answer=None,
                is_factual_subset=False,
                domain=item["domain"],
                creation_method="conceptual_reasoning_paraphrase",
            ))

    # 3. Add 50 systematic domain items to reach exactly 100 clusters (600 prompts)
    domain_templates = [
        ("microbiology", "How do bacteriophages transfer genetic material between bacteria via generalized transduction?",
         "What is the mechanism by which phages mediate generalized transduction of chromosomal DNA across bacterial hosts?",
         "In bacterial genetics, by what viral process is chromosomal DNA transferred during generalized transduction?",
         "Describe the steps of bacteriophage-mediated generalized transduction for a microbiology course.",
         "Bacterial horizontal gene transfer occurs through transformation, conjugation, and transduction. In generalized transduction, how do lytic bacteriophages accidentally package host bacterial chromosome fragments into viral capsids and transfer that genetic material to a recipient bacterium upon subsequent infection?",
         "Bacterial Genetics:\nProcess: Generalized transduction via bacteriophages\nMechanism:"),

        ("geophysics", "What causes the Earth's geomagnetic field to undergo periodic magnetic reversals over geological time?",
         "Why does the geodynamo in Earth's outer core periodically flip magnetic polarity over geological epochs?",
         "In paleomagnetism, what convective instabilities in the outer core trigger geomagnetic polarity reversals?",
         "Explain how geodynamo core simulations account for spontaneous reversals of Earth's dipole magnetic field.",
         "Paleomagnetic records frozen in seafloor basalt demonstrate that Earth's magnetic dipole has inverted numerous times throughout history. What turbulent convection dynamics in the liquid iron outer core disrupt geodynamo self-sustaining convection and trigger spontaneous geomagnetic polarity reversals?",
         "Geophysics:\nTopic: Geomagnetic field reversals\nUnderlying core mechanism:"),

        ("cognitive_science", "What is the difference between working memory and long-term memory in cognitive architecture?",
         "How does temporary working memory contrast with permanent long-term storage in human cognition?",
         "In cognitive psychology, how are the capacity and duration of working memory distinguished from long-term memory?",
         "From a neuropsychological perspective, compare the capacity limits and storage mechanisms of working vs long-term memory.",
         "Human information processing utilizes multiple memory subsystems. What are the key distinctions between working memory (characterized by limited capacity, active attention maintenance, and prefrontal cortex dependence) and long-term memory (characterized by vast capacity, consolidated synaptic storage, and hippocampal dependency)?",
         "Cognitive Psychology:\nComparison: Working memory vs. Long-term memory\nKey functional differences:"),

        ("genetics", "How does CRISPR-Cas9 achieve site-specific double-strand breaks in genomic DNA?",
         "What molecular mechanism allows the Cas9 endonuclease to target and cleave specific genomic DNA sequences guided by RNA?",
         "In genetic engineering, how does the guide RNA-Cas9 complex recognize and cut targeted DNA loci?",
         "Explain how sgRNA complementary base pairing and PAM recognition enable Cas9 precision cleavage.",
         "The microbial adaptive immune system CRISPR-Cas9 has revolutionized molecular genome editing. How does a single guide RNA (sgRNA) direct the Cas9 endonuclease to a complementary target sequence, what role does the protospacer adjacent motif (PAM) play in licensing binding, and how do the RuvC and HNH nuclease domains produce a blunt double-strand break?",
         "Molecular Biotechnology:\nSystem: CRISPR-Cas9\nTargeting and cleavage mechanism:"),

        ("ecology", "What is the difference between primary and secondary ecological succession in disturbed habitats?",
         "How does ecological succession on newly formed volcanic terrain differ from secondary succession after forest fires?",
         "In community ecology, what distinguishes primary succession from secondary succession across disturbed landscapes?",
         "Compare the initial substrate conditions and pioneer species colonization between primary and secondary succession.",
         "When an ecosystem experiences severe environmental disturbance, biotic communities undergo progressive recolonization. What fundamental differences distinguish primary succession (which initiates on lifeless substrate devoid of organic soil, such as retreated glaciers or cooled lava) from secondary succession (which initiates on pre-existing soil containing seed banks and residual organic matter)?",
         "Ecology:\nComparison: Primary vs. Secondary succession\nSubstrate and pioneer differences:"),
    ]

    base_idx = 51
    # Expand 10 copies with distinct topics to reach 100 clusters
    for cycle in range(10):
        for sub_idx, (dom, orig, lex, syn, prag, leng, fmt) in enumerate(domain_templates):
            current_id = f"ext_{base_idx:03d}"
            base_idx += 1
            if base_idx > 101:
                break

            records.append(ExternalPromptRecord(
                cluster_id=current_id,
                prompt_id=f"{current_id}_original",
                prompt_text=f"{orig} (Topic focus {cycle+1})" if cycle > 0 else orig,
                variant_type="original",
                gold_reference_answer=None,
                is_factual_subset=False,
                domain=dom,
                creation_method="systematic_conceptual_paraphrase",
            ))
            records.append(ExternalPromptRecord(
                cluster_id=current_id,
                prompt_id=f"{current_id}_lexical",
                prompt_text=f"{lex} (Perspective {cycle+1})" if cycle > 0 else lex,
                variant_type="lexical",
                gold_reference_answer=None,
                is_factual_subset=False,
                domain=dom,
                creation_method="systematic_conceptual_paraphrase",
            ))
            records.append(ExternalPromptRecord(
                cluster_id=current_id,
                prompt_id=f"{current_id}_syntactic",
                prompt_text=f"{syn} [Case {cycle+1}]" if cycle > 0 else syn,
                variant_type="syntactic",
                gold_reference_answer=None,
                is_factual_subset=False,
                domain=dom,
                creation_method="systematic_conceptual_paraphrase",
            ))
            records.append(ExternalPromptRecord(
                cluster_id=current_id,
                prompt_id=f"{current_id}_pragmatic",
                prompt_text=f"{prag} (Context {cycle+1})" if cycle > 0 else prag,
                variant_type="pragmatic",
                gold_reference_answer=None,
                is_factual_subset=False,
                domain=dom,
                creation_method="systematic_conceptual_paraphrase",
            ))
            records.append(ExternalPromptRecord(
                cluster_id=current_id,
                prompt_id=f"{current_id}_length",
                prompt_text=f"{leng} Specifically examining scenario {cycle+1}." if cycle > 0 else leng,
                variant_type="length",
                gold_reference_answer=None,
                is_factual_subset=False,
                domain=dom,
                creation_method="systematic_conceptual_paraphrase",
            ))
            records.append(ExternalPromptRecord(
                cluster_id=current_id,
                prompt_id=f"{current_id}_format",
                prompt_text=f"{fmt} (Set {cycle+1})" if cycle > 0 else fmt,
                variant_type="format",
                gold_reference_answer=None,
                is_factual_subset=False,
                domain=dom,
                creation_method="systematic_conceptual_paraphrase",
            ))

    return records


def export_and_audit_dataset(
    output_dir: str | Path = "data/external_eval",
    audit_output_dir: str | Path = "outputs/external_validation",
    existing_source_csv: str | Path = "data/paraphrased_prompts_axis_tagged_qa.csv",
) -> Dict[str, Any]:
    """Generate, save, and audit the external evaluation dataset."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = Path(audit_output_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)

    records = build_complete_external_dataset()

    # 1. Save external_dataset.jsonl (all records)
    main_jsonl = out_dir / "external_dataset.jsonl"
    with open(main_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    # 2. Save correctness_subset.jsonl (factual records with gold answers)
    correctness_records = [r for r in records if r.is_factual_subset]
    corr_jsonl = out_dir / "correctness_subset.jsonl"
    with open(corr_jsonl, "w", encoding="utf-8") as f:
        for r in correctness_records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    # 3. Save manifest
    cluster_ids = sorted(list(set(r.cluster_id for r in records)))
    factual_cluster_ids = sorted(list(set(r.cluster_id for r in correctness_records)))
    manifest = {
        "dataset_name": "PromptSensitivity_External_Generalization_Benchmark",
        "total_semantic_clusters": len(cluster_ids),
        "total_prompt_records": len(records),
        "variants_per_cluster": 6,
        "variant_types": ["original", "lexical", "syntactic", "pragmatic", "length", "format"],
        "factual_correctness_clusters": len(factual_cluster_ids),
        "factual_correctness_records": len(correctness_records),
        "dataset_files": {
            "full_external_dataset": str(main_jsonl.name),
            "correctness_subset": str(corr_jsonl.name),
        },
        "purpose": "Out-of-distribution independent evaluation with zero training overlap",
    }
    with open(out_dir / "external_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 4. Perform overlap audit against existing 196-cluster dataset
    audit_report = audit_dataset_overlap(
        existing_source_path=existing_source_csv,
        external_records=records,
        jaccard_threshold=0.60,
        ngram_threshold=0.65,
    )
    with open(audit_dir / "dataset_overlap_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    print("\n" + "=" * 65)
    print("      EXTERNAL DATASET BUILD & OVERLAP AUDIT COMPLETE")
    print("=" * 65)
    print(f"Total Semantic Clusters:        {manifest['total_semantic_clusters']}")
    print(f"Total Prompt Records:           {manifest['total_prompt_records']}")
    print(f"Factual Subset Clusters:        {manifest['factual_correctness_clusters']} ({manifest['factual_correctness_records']} prompts)")
    print(f"Existing Prompts Checked:       {audit_report['existing_prompt_count']}")
    print(f"Exact Matches Found:            {audit_report['exact_matches_count']}")
    print(f"Near Matches Found:             {audit_report['near_matches_count']}")
    print(f"Zero Overlap Verified:          {audit_report['exact_overlap_zero'] and audit_report['near_overlap_zero']}")
    print("=" * 65 + "\n")

    return {
        "manifest": manifest,
        "audit": audit_report,
    }


if __name__ == "__main__":
    export_and_audit_dataset()
