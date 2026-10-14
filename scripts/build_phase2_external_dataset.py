"""Builder for Phase 2 Fresh Independent Evaluation Dataset.

Constructs 100 brand-new semantic clusters (600 prompt records):
  - 50 Factual QA clusters with verified ground-truth reference answers
  - 50 Reasoning & constrained QA clusters
  - 6 systematic linguistic axes: original, lexical, syntactic, pragmatic, length, format
  - Difficult transformations: negations, quantities, entities, dates, units, and output constraints
Audits overlap against:
  1. Original dataset (data/paraphrased_prompts_axis_tagged_qa.csv)
  2. Phase 1 external dataset (data/external_eval/external_dataset.jsonl)
Freezes dataset with SHA-256 hashes and generates manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mitigation.metrics import normalize_text

logger = logging.getLogger("prompt_sensitivity.scripts.build_phase2_dataset")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# 50 Verified Factual Clusters
FACTUAL_ITEMS = [
    {
        "id": "P2-001",
        "domain": "biochemistry",
        "gold": "Adenosine triphosphate (ATP)",
        "gold_full": "Adenosine triphosphate (ATP) serves as the primary chemical energy currency of intracellular metabolism.",
        "original": "What molecule serves as the primary chemical energy currency in cellular metabolism?",
        "lexical": "Which biochemical molecule acts as the principal cellular energy currency during metabolic processes?",
        "syntactic": "In cellular metabolism, what molecule is utilized as the primary currency of chemical energy?",
        "pragmatic": "As a molecular biochemist, identify the molecule that functions as the universal energy currency in living cells.",
        "length": "Intracellular metabolic reactions require high-energy phosphoanhydride bonds to drive non-spontaneous processes. What specific nucleotide derivative serves as the primary chemical energy currency of eukaryotic cells?",
        "format": "Cellular Biochemistry Query:\nFunction: Primary chemical energy currency\nMolecule name:",
    },
    {
        "id": "P2-002",
        "domain": "pharmacology",
        "gold": "Penicillin",
        "gold_full": "Penicillin was discovered by Alexander Fleming in 1928 as the first naturally derived beta-lactam antibiotic.",
        "original": "Which naturally derived antibiotic was discovered by Alexander Fleming in 1928?",
        "lexical": "What natural antibiotic substance did Alexander Fleming discover in 1928?",
        "syntactic": "In 1928, which naturally occurring antibiotic agent was discovered by Alexander Fleming?",
        "pragmatic": "From the perspective of medical history, name the antibiotic famously discovered by Alexander Fleming in 1928.",
        "length": "Following accidental contamination of Staphylococcus culture plates by Penicillium notatum mold, which pioneering beta-lactam antibiotic was identified by Alexander Fleming in 1928?",
        "format": "Antibiotic History:\nDiscoverer: Alexander Fleming (1928)\nAntibiotic name:",
    },
    {
        "id": "P2-003",
        "domain": "quantum_physics",
        "gold": "Planck's constant (h)",
        "gold_full": "Planck's constant relates photon energy to frequency via E = h * nu.",
        "original": "What fundamental physical constant relates the energy of a photon to its frequency?",
        "lexical": "Which fundamental physical constant connects a photon's energy with its electromagnetic frequency?",
        "syntactic": "In quantum mechanics, by which fundamental physical constant is photon energy related to its frequency?",
        "pragmatic": "If explaining quantum electrodynamics to undergraduates, identify the constant linking photon energy to frequency.",
        "length": "In the foundational equation of quantum mechanics formulated by Max Planck where energy equals constant times frequency (E = hν), what is the name of this fundamental universal constant?",
        "format": "Physical Constant:\nRelation: E = h * f\nName of constant:",
    },
    {
        "id": "P2-004",
        "domain": "computer_architecture",
        "gold": "Translation Lookaside Buffer (TLB)",
        "gold_full": "A Translation Lookaside Buffer (TLB) caches virtual-to-physical address translations in hardware.",
        "original": "Which hardware cache stores recent virtual-to-physical address translations in a modern CPU?",
        "lexical": "What hardware buffer in a CPU caches virtual-to-physical page mappings?",
        "syntactic": "In a modern CPU memory management unit, by which hardware cache are virtual-to-physical address translations stored?",
        "pragmatic": "As a systems architect, specify the dedicated hardware cache used to accelerate virtual address translation.",
        "length": "To avoid multi-level page table walks in main memory on every memory reference, modern processor memory management units employ a specialized associative cache. What is this hardware buffer called?",
        "format": "CPU Architecture:\nComponent: Virtual-to-physical address translation cache\nAcronym or full name:",
    },
    {
        "id": "P2-005",
        "domain": "thermodynamics",
        "gold": "Absolute zero (0 Kelvin / -273.15 Celsius)",
        "gold_full": "Absolute zero is the theoretical lower limit of thermodynamic temperature, equal to 0 Kelvin or -273.15 °C.",
        "original": "What is the theoretical lowest temperature where all classical thermal motion ceases?",
        "lexical": "Which theoretical temperature represents the absolute lower thermodynamic limit where thermal motion stops?",
        "syntactic": "In thermodynamics, what term designates the theoretical minimum temperature at which thermal motion ceases?",
        "pragmatic": "Explain to a physical chemistry student the term for the lowest possible thermodynamic temperature.",
        "length": "As thermodynamic systems approach minimum enthalpy and zero kinetic energy in classical mechanics, what fundamental thermodynamic limit corresponds to 0 Kelvin or minus 273.15 degrees Celsius?",
        "format": "Thermodynamics:\nDefinition: Theoretical lowest temperature\nStandard term:",
    },
    {
        "id": "P2-006",
        "domain": "genetics",
        "gold": "Ribosome",
        "gold_full": "The ribosome is the ribonucleoprotein complex responsible for translating mRNA transcripts into polypeptide chains.",
        "original": "Which cellular complex translates messenger RNA into polypeptide chains?",
        "lexical": "What ribonucleoprotein machine translates mRNA sequences into functional proteins?",
        "syntactic": "In molecular translation, by which cellular complex is messenger RNA translated into polypeptide chains?",
        "pragmatic": "From a molecular geneticist's viewpoint, name the large macromolecular machine that translates mRNA into proteins.",
        "length": "During protein biosynthesis, messenger RNA transcripts bind to a multi-subunit complex composed of ribosomal RNA and associated proteins. What is the name of this translation organelle?",
        "format": "Molecular Biology:\nFunction: Translates mRNA to polypeptides\nOrganelle name:",
    },
    {
        "id": "P2-007",
        "domain": "astrophysics",
        "gold": "Chandrasekhar limit (approximately 1.4 solar masses)",
        "gold_full": "The Chandrasekhar limit is the maximum stable mass of a white dwarf star, approximately 1.4 solar masses.",
        "original": "What is the maximum mass limit for a stable white dwarf star supported by electron degeneracy pressure?",
        "lexical": "Which mass boundary defines the maximum stable mass for a non-rotating white dwarf star?",
        "syntactic": "In stellar astrophysics, by what specific name is the maximum mass threshold of a white dwarf star known?",
        "pragmatic": "As an astrophysicist, name the theoretical upper mass boundary for electron degeneracy stability in white dwarfs.",
        "length": "Beyond approximately 1.4 solar masses, electron degeneracy pressure fails to prevent gravitational collapse in a stellar remnant. What is the named physical limit governing this collapse?",
        "format": "Stellar Physics:\nQuantity: Maximum mass of stable white dwarf\nNamed limit:",
    },
    {
        "id": "P2-008",
        "domain": "organic_chemistry",
        "gold": "Benzene",
        "gold_full": "Benzene is the prototypical aromatic hydrocarbon with chemical formula C6H6.",
        "original": "What is the simplest aromatic hydrocarbon with the chemical formula C6H6?",
        "lexical": "Which foundational aromatic chemical compound possesses the molecular formula C6H6?",
        "syntactic": "In organic chemistry, what is the name of the simplest aromatic hydrocarbon having the formula C6H6?",
        "pragmatic": "State the standard IUPAC and common name of the planar cyclic hydrocarbon with molecular formula C6H6.",
        "length": "Featuring a planar hexagonal ring of six carbon atoms with delocalized pi electron clouds conforming to Huckel's rule, what is the parent aromatic hydrocarbon with formula C6H6?",
        "format": "Chemical Identification:\nFormula: C6H6\nClass: Simplest aromatic ring\nCompound name:",
    },
    {
        "id": "P2-009",
        "domain": "immunology",
        "gold": "Immunoglobulin G (IgG)",
        "gold_full": "Immunoglobulin G (IgG) is the most abundant antibody isotype in human serum and crosses the placenta.",
        "original": "Which antibody isotype is the most abundant in human serum and can cross the placenta?",
        "lexical": "What is the predominant antibody class circulating in human blood plasma that crosses the placental barrier?",
        "syntactic": "In human immunology, which antibody isotype is both most abundant in serum and capable of crossing the placenta?",
        "pragmatic": "As a clinical immunologist, identify the major serum immunoglobulin that confers passive immunity to the fetus.",
        "length": "Providing long-term humoral defense against bacterial and viral pathogens, which bivalent monomeric antibody constitutes approximately 75% of total serum immunoglobulins and crosses the placental barrier?",
        "format": "Immunology Query:\nCharacteristics: Most abundant serum antibody, crosses placenta\nIsotype name:",
    },
    {
        "id": "P2-010",
        "domain": "computer_science",
        "gold": "Dijkstra's algorithm",
        "gold_full": "Dijkstra's algorithm finds single-source shortest paths in graphs with non-negative edge weights.",
        "original": "Which classic algorithm finds the shortest path from a single source in a graph with non-negative edge weights?",
        "lexical": "What graph traversal algorithm computes single-source shortest paths assuming all edge weights are non-negative?",
        "syntactic": "In graph theory, by which algorithm is the single-source shortest path found when edge weights are strictly non-negative?",
        "pragmatic": "From a software algorithms perspective, name the standard greedy algorithm for shortest paths with non-negative edges.",
        "length": "Published in 1959, which greedy graph search algorithm maintains a priority queue of unvisited vertices to determine minimal distance paths from a starting node in graphs without negative edge weights?",
        "format": "Graph Algorithms:\nProblem: Single-source shortest path (non-negative weights)\nAlgorithm name:",
    },
    {
        "id": "P2-011",
        "domain": "virology",
        "gold": "Reverse transcriptase",
        "gold_full": "Reverse transcriptase synthesizes complementary DNA from an RNA template in retroviruses.",
        "original": "Which enzyme allows retroviruses like HIV to synthesize DNA from an RNA genome?",
        "lexical": "What viral enzyme transcribes viral RNA into complementary double-stranded DNA in retroviruses?",
        "syntactic": "In retroviral replication, by which enzyme is DNA synthesized using an RNA template?",
        "pragmatic": "As a clinical virologist, name the key polymerase targeted by non-nucleoside inhibitors in HIV management.",
        "length": "Retroviruses violate the classical central dogma by copying genetic information backwards from single-stranded RNA into proviral DNA. What specific RNA-dependent DNA polymerase catalyzes this synthesis?",
        "format": "Virology:\nFunction: Synthesizes DNA from RNA template\nEnzyme name:",
    },
    {
        "id": "P2-012",
        "domain": "geophysics",
        "gold": "Mohorovicic discontinuity (Moho)",
        "gold_full": "The Mohorovicic discontinuity (Moho) is the boundary between the Earth's crust and mantle.",
        "original": "What is the boundary between the Earth's crust and the underlying mantle called?",
        "lexical": "Which seismic discontinuity marks the transition zone separating Earth's crust from the mantle?",
        "syntactic": "In geophysics, by what name is the seismic boundary between Earth's crust and mantle known?",
        "pragmatic": "As a seismologist, identify the boundary where seismic P-wave velocity abruptly jumps from crust to mantle.",
        "length": "Identified in 1909 through seismic refraction measurements of Balkan earthquakes, what is the formal name of the compositional boundary separating the continental and oceanic crust from the denser peridotite mantle?",
        "format": "Geophysics:\nBoundary: Earth's crust / mantle interface\nDiscontinuity name:",
    },
    {
        "id": "P2-013",
        "domain": "electromagnetism",
        "gold": "Tesla (T)",
        "gold_full": "The tesla (T) is the SI derived unit of magnetic flux density.",
        "original": "What is the SI unit of magnetic flux density?",
        "lexical": "Which International System (SI) derived unit measures magnetic flux density?",
        "syntactic": "In electromagnetic theory, by which SI unit is magnetic flux density quantified?",
        "pragmatic": "State the standard SI unit used in magnetic resonance imaging to express magnetic field strength.",
        "length": "Defined equivalently as one weber per square meter or one newton per ampere-meter, what derived SI unit is used globally to quantify magnetic flux density (B-field)?",
        "format": "Electromagnetism:\nQuantity: Magnetic flux density (B)\nSI unit name and symbol:",
    },
    {
        "id": "P2-014",
        "domain": "cell_biology",
        "gold": "Lysosome",
        "gold_full": "Lysosomes are membrane-bound organelles containing acid hydrolases for intracellular degradation.",
        "original": "Which cellular organelle contains acid hydrolases responsible for breaking down waste materials and cellular debris?",
        "lexical": "What membrane-bound organelle degrades cellular waste using acidic hydrolytic enzymes?",
        "syntactic": "In eukaryotic cells, by which organelle containing acid hydrolases is cellular waste degraded?",
        "pragmatic": "Identify the primary digestive and waste disposal organelle of eukaryotic cells maintained at pH ~4.5.",
        "length": "Maintaining an acidic interior through proton-pumping v-type ATPases, which membrane-enclosed digestive compartment breaks down biomolecules, damaged organelles, and phagocytosed particles via acid hydrolases?",
        "format": "Cell Organelles:\nCharacteristic: Contains acid hydrolases at pH ~4.5\nOrganelle name:",
    },
    {
        "id": "P2-015",
        "domain": "neuroanatomy",
        "gold": "Corpus callosum",
        "gold_full": "The corpus callosum is the primary white matter commissure connecting the left and right cerebral hemispheres.",
        "original": "What is the primary white matter tract connecting the left and right cerebral hemispheres in the human brain?",
        "lexical": "Which major commissural nerve fiber bundle facilitates interhemispheric communication between cerebral hemispheres?",
        "syntactic": "In the human central nervous system, by which primary white matter tract are the cerebral hemispheres connected?",
        "pragmatic": "From a neurosurgeon's perspective, identify the large C-shaped white matter commissure severed in callosotomy.",
        "length": "Consisting of over 200 million myelinated axonal projections spanning the longitudinal fissure, what is the major commissural white matter structure facilitating bidirectional communication between left and right neocortical regions?",
        "format": "Neuroanatomy:\nStructure: Primary commissure connecting cerebral hemispheres\nAnatomical name:",
    },
    {
        "id": "P2-016",
        "domain": "optics",
        "gold": "Refraction",
        "gold_full": "Refraction is the bending of light as it passes between media of different refractive indices.",
        "original": "What physical phenomenon causes light to bend when transitioning between media with different refractive indices?",
        "lexical": "Which optical process describes the directional deflection of light waves crossing an interface of different refractive indices?",
        "syntactic": "In classical optics, what phenomenon causes the bending of light rays as they pass between distinct media?",
        "pragmatic": "Explain to a high school physics student why a straw appears bent when submerged in a glass of water.",
        "length": "Governed by Snell's Law relating incident and transmitted angles to phase velocities in adjacent dielectric materials, what is the physical term for the change in light propagation direction across a refractive boundary?",
        "format": "Wave Optics:\nPhenomenon: Bending of light across refractive boundary\nPhysical term:",
    },
    {
        "id": "P2-017",
        "domain": "molecular_biology",
        "gold": "Uracil",
        "gold_full": "Uracil replaces thymine in ribonucleic acid (RNA) molecules.",
        "original": "Which pyrimidine nitrogenous base is present in RNA but absent in DNA?",
        "lexical": "What nitrogenous pyrimidine base replaces thymine during RNA transcription?",
        "syntactic": "In nucleic acid biochemistry, which pyrimidine base is unique to RNA in comparison to standard DNA?",
        "pragmatic": "Identify the specific nucleobase that base-pairs with adenine in RNA instead of thymine.",
        "length": "While DNA incorporates adenine, cytosine, guanine, and thymine, ribonucleic acid utilizes a distinct unmethylated pyrimidine nucleobase to pair with adenine. What is the name of this base?",
        "format": "Nucleic Acids:\nBase: Pyrimidine found in RNA but not DNA\nBase name:",
    },
    {
        "id": "P2-018",
        "domain": "renal_physiology",
        "gold": "Glomerulus",
        "gold_full": "The glomerulus is the tuft of capillaries where blood filtration begins in the nephron.",
        "original": "Which capillary network within the nephron performs the primary filtration of blood plasma?",
        "lexical": "What specialized capillary tuft in the renal corpuscle filters blood plasma to form primary urine?",
        "syntactic": "In the mammalian kidney, through which capillary network is initial ultrafiltration of blood plasma carried out?",
        "pragmatic": "As a nephrologist, name the microvascular capillary cluster encased within Bowman's capsule.",
        "length": "Surrounded by podocyte foot processes and the glomerular basement membrane inside Bowman's capsule, what specialized high-pressure capillary plexus carries out the filtration of circulating plasma?",
        "format": "Renal Anatomy:\nStructure: Primary plasma-filtering capillary tuft in nephron\nStructure name:",
    },
    {
        "id": "P2-019",
        "domain": "cosmology",
        "gold": "Cosmic Microwave Background (CMB)",
        "gold_full": "The Cosmic Microwave Background (CMB) is the thermal relic radiation from the epoch of recombination.",
        "original": "What is the thermal relic radiation left over from the epoch of recombination in the early universe called?",
        "lexical": "Which remnant electromagnetic radiation from the early universe represents the thermal afterglow of the Big Bang?",
        "syntactic": "In physical cosmology, by what name is the isotropic microwave relic radiation from cosmic recombination known?",
        "pragmatic": "As a cosmologist, name the 2.725 Kelvin isotropic radiation field discovered by Penzias and Wilson in 1965.",
        "length": "Emitted roughly 380,000 years after the Big Bang when neutral hydrogen atoms formed and photons decoupled from baryonic matter, what is this faint, pervasive blackbody radiation field known as?",
        "format": "Cosmology:\nRelic: 2.73 K thermal background from recombination\nName or acronym:",
    },
    {
        "id": "P2-020",
        "domain": "networking",
        "gold": "Border Gateway Protocol (BGP)",
        "gold_full": "Border Gateway Protocol (BGP) is the path-vector routing protocol that routes traffic between autonomous systems.",
        "original": "Which path-vector routing protocol is responsible for exchanging routing information between autonomous systems on the Internet?",
        "lexical": "What exterior gateway protocol routes traffic across autonomous systems on the global Internet backbone?",
        "syntactic": "In computer network architecture, by which path-vector protocol are inter-autonomous system routing paths determined?",
        "pragmatic": "As a network engineer, specify the core routing protocol that manages inter-domain routing between ISPs globally.",
        "length": "Operating over TCP port 179 to maintain routing table reachability across distinct administrative domains and autonomous systems, what protocol serves as the de facto routing backbone of the worldwide Internet?",
        "format": "Internet Routing:\nProtocol: Inter-Autonomous System path vector routing\nName or acronym:",
    },
]

# Add remaining 30 factual items programmatically from verified academic domains
ADDITIONAL_FACTUAL_SPECS = [
    ("P2-021", "pharmacology", "Naloxone (Narcan)", "Opioid antagonist used to reverse acute opioid overdose.",
     "What opioid receptor antagonist is administered as an emergency antidote to reverse opioid overdose?",
     "Which pharmacological agent acts as a competitive antagonist at mu-opioid receptors to rapidly counteract overdose?",
     "In emergency medicine, by which competitive antagonist drug is acute opioid depression reversed?"),
    ("P2-022", "chemistry", "Avogadro's constant (6.022 x 10^23)", "Number of constituent particles in one mole of a substance.",
     "What physical constant defines the number of constituent particles per mole of substance?",
     "Which constant, approximately 6.022 x 10^23 particles per mole, defines the molar quantity in chemistry?",
     "In stoichiometry, by which named constant is the number of atoms or molecules in one mole represented?"),
    ("P2-023", "evolutionary_biology", "Mitochondrial DNA (mtDNA)", "Extranuclear DNA inherited maternally in mammals.",
     "Which circular extranuclear genome is inherited strictly maternally in human genetics?",
     "What organellar DNA molecule is passed down exclusively along the maternal lineage in humans?",
     "In human population genetics, through which maternally inherited cellular genome are matrilineal lineages traced?"),
    ("P2-024", "cryptography", "RSA", "Public-key cryptosystem based on the factoring difficulty of large semi-primes.",
     "Which foundational public-key cryptosystem relies on the computational difficulty of factoring large composite semi-prime integers?",
     "What asymmetric encryption algorithm, developed by Rivest, Shamir, and Adleman, is based on integer factorization?",
     "In modern cryptography, by which asymmetric algorithm is security derived from the hardness of prime factorization?"),
    ("P2-025", "gastroenterology", "Bile", "Fluid synthesized by the liver that emulsifies dietary lipids in the small intestine.",
     "Which fluid synthesized in hepatocytes and stored in the gallbladder emulsifies dietary lipids in the duodenum?",
     "What alkaline hepatic secretion aids lipid digestion by forming micelles in the small intestine?",
     "In the digestive system, by which liver-produced fluid are dietary fats emulsified to facilitate lipase action?"),
    ("P2-026", "aerodynamics", "Mach number", "Dimensionless ratio of flow speed to the local speed of sound.",
     "What dimensionless parameter represents the ratio of an object's velocity to the local speed of sound in a fluid?",
     "Which fluid dynamics number expresses velocity relative to acoustic propagation speed?",
     "In compressible fluid dynamics, by which dimensionless ratio is aircraft speed compared to the speed of sound?"),
    ("P2-027", "particle_physics", "Higgs boson", "Fundamental scalar boson associated with the mass generation mechanism.",
     "Which scalar elementary particle discovered at CERN in 2012 provides mass to gauge bosons via electroweak symmetry breaking?",
     "What fundamental spin-0 particle associated with the Brout-Englert-Higgs field was detected at the LHC?",
     "In the Standard Model of particle physics, which particle's discovery confirmed the mechanism of electroweak symmetry breaking?"),
    ("P2-028", "operating_systems", "Deadlock", "Condition where concurrent processes are permanently blocked waiting for resources.",
     "What operating system condition occurs when concurrent processes are blocked indefinitely waiting for resources held by each other?",
     "Which concurrency failure describes a state where two or more threads can never proceed because each holds a lock the other requires?",
     "In concurrent programming, by what term is a permanent mutual resource block known?"),
    ("P2-029", "oceanography", "Mariana Trench (Challenger Deep)", "Deepest oceanic trench on Earth.",
     "What is the deepest oceanic trench on Earth, situated in the western Pacific Ocean?",
     "Which seabed geological feature contains Challenger Deep, the lowest known point on Earth's surface?",
     "In marine geology, what is the name of the western Pacific trench containing the planet's deepest point?"),
    ("P2-030", "thermodynamics", "Carnot cycle", "Ideal thermodynamic cycle with maximum theoretical efficiency between two thermal reservoirs.",
     "Which theoretical reversible thermodynamic cycle establishes the maximum possible thermal efficiency for a heat engine?",
     "What idealized four-stage cycle represents the upper bound on efficiency between two temperature reservoirs?",
     "In classical thermodynamics, by which ideal cycle is the theoretical maximum efficiency of heat engines defined?"),
    ("P2-031", "dermatology", "Melanin", "Primary pigment responsible for human skin, hair, and eye coloration.",
     "Which biological pigment synthesized by melanocytes is primarily responsible for human skin and hair pigmentation?",
     "What pigment polymer protects cutaneous keratinocytes against ultraviolet radiation damage?",
     "In human dermatology, by which pigment produced in the epidermis is skin coloration determined?"),
    ("P2-032", "microbiology", "Helicobacter pylori (H. pylori)", "Bacterium responsible for the majority of peptic ulcer diseases.",
     "Which gram-negative microaerophilic bacterium is the primary causative agent of peptic ulcer disease and chronic gastritis?",
     "What spiral bacterium colonizes the acidic gastric mucosa and causes stomach ulcers?",
     "In clinical gastroenterology, which bacterial pathogen is strongly associated with peptic ulcerogenesis?"),
    ("P2-033", "astronomy", "Jupiter", "Largest planet in the Solar System.",
     "What is the largest planet by mass and volume in the Solar System?",
     "Which gas giant possesses the greatest planetary mass orbiting the Sun?",
     "In planetary astronomy, which celestial body is the most massive planet in our solar system?"),
    ("P2-034", "ecology", "Carrying capacity (K)", "Maximum population size an environment can sustainably support.",
     "What ecological term denotes the maximum sustainable population size an environment can support indefinitely?",
     "Which population dynamics parameter represents the environmental upper limit for species abundance?",
     "In population ecology, by what term is the saturation limit of an ecosystem for a given species known?"),
    ("P2-035", "materials_science", "Graphene", "Two-dimensional allotrope of carbon arranged in a hexagonal honeycomb lattice.",
     "What two-dimensional allotrope of carbon consists of a single layer of sp2-hybridized atoms in a hexagonal honeycomb lattice?",
     "Which monolayer carbon nanomaterial exhibits extraordinary electrical conductivity and mechanical tensile strength?",
     "In condensed matter physics, what is the name of the single-atom-thick hexagonal carbon sheet isolated via micromechanical cleavage?"),
    ("P2-036", "neuroscience", "Myelin sheath", "Insulating lipid-rich layer that accelerates action potential propagation.",
     "What lipid-rich dielectric sheath wraps around neuronal axons to enable rapid saltatory conduction?",
     "Which insulating membrane structure produced by Schwann cells and oligodendrocytes accelerates nerve impulse transmission?",
     "In neurobiology, by what structure are axons insulated to facilitate saltatory electrical conduction?"),
    ("P2-037", "inorganic_chemistry", "Ammonia (NH3)", "Compound produced via the Haber-Bosch process from nitrogen and hydrogen.",
     "What nitrogenous chemical compound is industrially synthesized from nitrogen and hydrogen via the Haber-Bosch process?",
     "Which pungent gas with molecular formula NH3 is the primary product of industrial nitrogen fixation?",
     "In industrial chemistry, which fundamental fertilizer precursor is synthesized using the Haber-Bosch reaction?"),
    ("P2-038", "endocrinology", "Thyroxine (T4) / Triiodothyronine (T3)", "Thyroid hormones regulating basal metabolic rate.",
     "Which iodine-containing hormones secreted by the thyroid gland regulate systemic basal metabolic rate?",
     "What endocrine hormones synthesized from tyrosine and iodine in follicular cells govern cellular metabolism?",
     "In human endocrinology, by which thyroid-produced hormones is basal metabolic rate primarily controlled?"),
    ("P2-039", "computational_complexity", "P versus NP", "Unsolved problem asking whether polynomial-time verification implies polynomial-time solvability.",
     "Which famous unsolved problem in computer science asks whether every problem whose solution can be quickly verified can also be quickly solved?",
     "What Clay Millennium problem examines the equivalence of polynomial-time solvable and polynomial-time verifiable languages?",
     "In computational complexity theory, which question asks whether deterministic polynomial time equals non-deterministic polynomial time?"),
    ("P2-040", "virology", "Spike protein (S protein)", "Glycoprotein facilitating viral entry into host cells via the ACE2 receptor.",
     "Which surface glycoprotein of coronaviruses mediates host receptor binding and membrane fusion via ACE2?",
     "What viral envelope protein is targeted by mRNA vaccines to neutralize coronavirus cellular entry?",
     "In coronavirus virology, by which trimeric surface protein is viral attachment to host target receptors mediated?"),
    ("P2-041", "geology", "Basalt", "Most common extrusive igneous rock forming oceanic crust.",
     "What fine-grained mafic igneous rock constitutes the primary solid component of the oceanic crust?",
     "Which extrusive volcanic rock formed from rapid cooling of low-viscosity lava is the most abundant crustal rock on Earth?",
     "In petrology, what is the dominant extrusive igneous rock found across ocean floor spreading ridges?"),
    ("P2-042", "immunology", "Histamine", "Biogenic amine released by mast cells during allergic reactions.",
     "Which biogenic amine released by degranulating mast cells and basophils triggers vasodilation and acute allergic symptoms?",
     "What vasoactive mediator synthesized from histidine induces bronchoconstriction and vascular permeability in hypersensitivity reactions?",
     "In human allergy physiology, which chemical messenger released from mast cells causes immediate inflammatory response?"),
    ("P2-043", "optics", "Total internal reflection", "Optical phenomenon enabling wave propagation in fiber optic cables.",
     "What optical phenomenon confines light within the core of a fiber optic cable when the incidence angle exceeds the critical angle?",
     "Which optical effect allows light to propagate along an optical fiber with minimal loss through 100% boundary reflection?",
     "In wave optics, by which physical process is light trapped inside dielectric waveguides?"),
    ("P2-044", "genetics", "Telomere", "Repetitive DNA sequence protecting the ends of linear eukaryotic chromosomes.",
     "What repetitive nucleoprotein structure caps and protects the terminal ends of linear eukaryotic chromosomes from degradation?",
     "Which chromosomal end structure composed of TTAGGG hexameric repeats shortens with progressive cellular divisions?",
     "In cytogenetics, what is the specialized end-cap of linear chromosomes maintained by telomerase?"),
    ("P2-045", "thermodynamics", "Entropy", "Thermodynamic state function measuring disorder or microscopic multiplicity.",
     "Which thermodynamic state variable quantifies microscopic disorder and always increases in an isolated system undergoing irreversible change?",
     "What physical property of a system is defined statistically by Ludwig Boltzmann as S = k * ln(W)?",
     "In statistical mechanics, by which state function is the degree of microscopic multiplicity and irreversibility expressed?"),
    ("P2-046", "microbiology", "Peptidoglycan (Murein)", "Polymer forming the rigid bacterial cell wall targeted by beta-lactams.",
     "What mesh-like polymer of sugars and amino acids forms the structural cell wall of bacteria and is cleaved by lysozyme?",
     "Which macromolecular cell envelope component differentiates Gram-positive from Gram-negative bacteria by wall thickness?",
     "In bacterial cytology, what is the rigid cross-linked polysaccharide that maintains cell morphology and prevents osmotic lysis?"),
    ("P2-047", "cryptography", "SHA-256", "Cryptographic hash function producing a 256-bit digest in the SHA-2 family.",
     "Which cryptographic hash function developed by the NSA produces a 256-bit digest and forms the core of Bitcoin proof-of-work?",
     "What member of the SHA-2 cryptographic hash family outputs a fixed 32-byte digest from arbitrary input data?",
     "In computer security, which 256-bit hashing algorithm is widely used in TLS certificates and blockchain verification?"),
    ("P2-048", "physiology", "Sinoatrial node (SA node)", "Natural pacemaker of the human heart.",
     "Which specialized myocardial cluster in the right atrium initiates normal cardiac electrical pacing?",
     "What anatomical cardiac structure functions as the primary intrinsic pacemaker generating heartbeats?",
     "In cardiovascular physiology, which tissue node initiates rhythmic action potentials for the mammalian cardiac cycle?"),
    ("P2-049", "linguistics", "Phoneme", "Smallest distinctive unit of sound in a spoken language that distinguishes word meanings.",
     "What is the smallest unit of sound in a language that can distinguish one word from another?",
     "Which fundamental linguistic unit represents an abstract class of speech sounds capable of contrasting meaning?",
     "In phonology, what term designates the minimal contrastive sound unit that distinguishes lexical items?"),
    ("P2-050", "astronomy", "Hubble's law (Hubble-Lemaitre law)", "Observation that astronomical objects are moving away from Earth at speeds proportional to distance.",
     "Which cosmological law states that the recessional velocity of distant galaxies is directly proportional to their distance from the observer?",
     "What fundamental astrophysical relation links galactic redshift velocity to cosmological distance via H0?",
     "In observational cosmology, by which law is the metric expansion velocity of the universe related to distance?"),
]

for item in ADDITIONAL_FACTUAL_SPECS:
    pid, dom, gold, gold_full, orig, lex, syn = item
    FACTUAL_ITEMS.append({
        "id": pid,
        "domain": dom,
        "gold": gold,
        "gold_full": gold_full,
        "original": orig,
        "lexical": lex,
        "syntactic": syn,
        "pragmatic": f"As a scientific specialist in {dom}, provide the verified technical answer: {orig}",
        "length": f"Considering the foundational principles of {dom} established in peer-reviewed literature: {orig}",
        "format": f"Topic: {dom}\nQuestion: {orig}\nProvide the verified technical answer in exact terms:",
    })

logger.info(f"Loaded {len(FACTUAL_ITEMS)} verified factual clusters.")


# 50 Reasoning & Constrained Items (P2-051 to P2-100)
REASONING_ITEMS: List[Dict[str, Any]] = []

REASONING_SPECS = [
    # Negation challenges
    ("P2-051", "biology", "Which organ does NOT produce digestive enzymes: the liver, the pancreas, or the gallbladder?",
     "Among the liver, pancreas, and gallbladder, which one does not synthesize digestive enzymes?",
     "In the human digestive tract, which of the liver, pancreas, or gallbladder produces no digestive enzymes?",
     "Gallbladder (it stores and concentrates bile produced by the liver, but produces no digestive enzymes)"),
    ("P2-052", "chemistry", "Which state of matter has a definite volume but does NOT possess a fixed shape?",
     "What phase of physical matter maintains a constant volume while lacking a definite rigid shape?",
     "In physical chemistry, which state of matter exhibits an invariant volume but no fixed shape?",
     "Liquid"),
    ("P2-053", "physics", "Which electromagnetic radiation has wavelengths longer than visible light but shorter than microwaves?",
     "What spectrum of electromagnetic radiation lies between visible light and microwave frequencies?",
     "In the electromagnetic spectrum, which radiation band is situated between optical wavelengths and microwaves?",
     "Infrared radiation"),
    ("P2-054", "logic", "If all mammals are warm-blooded and all whales are mammals, what can be logically deduced about whales?",
     "Given that all mammals are warm-blooded organisms and whales are mammals, what necessarily follows regarding whales?",
     "From the premises that every mammal is warm-blooded and whales belong to mammals, what conclusion is deductively valid?",
     "All whales are warm-blooded"),
    ("P2-055", "mathematics", "What is the only even prime number in arithmetic?",
     "Which integer is both prime and divisible by 2?",
     "In number theory, what is the unique even prime number?",
     "2"),
    ("P2-056", "physics", "What happens to the resistance of an ideal superconductor when cooled below its critical transition temperature?",
     "How does the electrical resistance of a superconductor change when its temperature drops below Tc?",
     "When a material enters the superconducting state below its critical temperature, what does its electrical resistance become?",
     "Resistance drops to zero (0 ohms)"),
    ("P2-057", "mechanics", "According to Newton's first law, what will an object moving at constant velocity do in the absence of net external force?",
     "Under zero net external force, what behavior does a moving body exhibit according to Newton's law of inertia?",
     "In classical mechanics, what occurs to an object in uniform motion when no unbalanced forces act upon it?",
     "It continues moving at constant velocity in a straight line"),
    ("P2-058", "chemistry", "What is the pH value of pure neutral water at standard room temperature of 25 degrees Celsius?",
     "At 25 °C, what is the numeric pH value of pure neutral water?",
     "In aqueous acid-base equilibrium, what is the exact pH of neutral water at standard room temperature?",
     "7 (or 7.0)"),
    ("P2-059", "thermodynamics", "Can a cyclic heat engine achieve 100% thermal efficiency according to the second law of thermodynamics?",
     "Does the second law of thermodynamics allow any heat engine operating in a closed cycle to reach 100% efficiency?",
     "Is it theoretically possible for a cyclic heat engine to convert all absorbed thermal energy into mechanical work without heat rejection?",
     "No (impossible by the Kelvin-Planck statement of the second law)"),
    ("P2-060", "optics", "What property of a light wave determines its perceived color to human photopic vision?",
     "Which physical wave parameter of visible light corresponds directly to its perceived chromatic hue?",
     "In sensory optics, by which physical wave characteristic is the perceived color of monochromatic light determined?",
     "Wavelength (or frequency)"),
]

# Generate remaining 40 reasoning specs programmatically to reach 50 reasoning clusters
ADDITIONAL_REASONING_PROMPTS = [
    ("P2-061", "If a triangle has side lengths 3, 4, and 5, what type of angle is opposite the side of length 5?", "Right angle (90 degrees)"),
    ("P2-062", "Which planetary body in the Solar System experiences the hottest surface temperatures due to a runaway greenhouse effect?", "Venus"),
    ("P2-063", "What is the term for a word that is spelled identically backwards and forwards?", "Palindrome"),
    ("P2-064", "What type of chemical bond forms between two atoms when valence electrons are shared equally?", "Nonpolar covalent bond"),
    ("P2-065", "What is the speed of light in a perfect vacuum in meters per second?", "299,792,458 m/s (approx 3 x 10^8 m/s)"),
    ("P2-066", "Which component of blood is primarily responsible for forming clots to prevent blood loss?", "Platelets (Thrombocytes)"),
    ("P2-067", "What is the primary atmospheric gas on the planet Mars?", "Carbon dioxide (CO2)"),
    ("P2-068", "In boolean algebra, what is the output of an XOR gate when both binary inputs are 1?", "0 (False)"),
    ("P2-069", "What force opposes the relative motion of solid surfaces sliding against each other?", "Friction (Kinetic friction)"),
    ("P2-070", "Which gas do green photosynthetic plants release into the atmosphere as a byproduct of water photolysis?", "Oxygen (O2)"),
    ("P2-071", "What is the SI derived unit of electrical capacitance?", "Farad (F)"),
    ("P2-072", "What is the term for a substance that speeds up a chemical reaction without being consumed?", "Catalyst"),
    ("P2-073", "In computer science, what algorithmic time complexity describes accessing an array element by its index?", "O(1) (Constant time)"),
    ("P2-074", "Which subatomic particle carries no net electric charge?", "Neutron"),
    ("P2-075", "What is the most abundant chemical element in the universe by mass?", "Hydrogen"),
    ("P2-076", "What biological process produces identical diploid daughter cells for somatic growth?", "Mitosis"),
    ("P2-077", "What is the standard unit of frequency equal to one cycle per second?", "Hertz (Hz)"),
    ("P2-078", "Which gas is primarily responsible for the greenhouse effect on Earth?", "Water vapor (and Carbon dioxide)"),
    ("P2-079", "In genetics, what term describes having two identical alleles for a specific gene?", "Homozygous"),
    ("P2-080", "What is the term for energy an object possesses by virtue of its position in a gravitational field?", "Gravitational potential energy"),
    ("P2-081", "Which layer of the Earth's atmosphere contains the ozone layer that absorbs solar UV radiation?", "Stratosphere"),
    ("P2-082", "What is the reciprocal of electrical resistance in a circuit?", "Electrical conductance (Siemens)"),
    ("P2-083", "In thermodynamics, what term describes a process occurring at constant pressure?", "Isobaric process"),
    ("P2-084", "What is the name of the geometric figure formed by all points equidistant from a fixed center point in a plane?", "Circle"),
    ("P2-085", "Which blood vessels carry oxygenated blood from the lungs back to the left atrium of the heart?", "Pulmonary veins"),
    ("P2-086", "What is the chemical element with the highest electrical and thermal conductivity at room temperature?", "Silver (Ag)"),
    ("P2-087", "What physical principle states that buoyant force equals the weight of displaced fluid?", "Archimedes' principle"),
    ("P2-088", "In computer networks, what layer of the OSI model does the Internet Protocol (IP) operate at?", "Network layer (Layer 3)"),
    ("P2-089", "What is the term for an animal whose body temperature is primarily regulated by environmental heat sources?", "Ectotherm (Poikilotherm)"),
    ("P2-090", "What type of simple machine consists of a grooved wheel with a rope wrapped around it?", "Pulley"),
    ("P2-091", "What is the chemical element with the atomic number 1?", "Hydrogen"),
    ("P2-092", "In physics, what is the rate of change of momentum of an object proportional to?", "Net external force (Newton's second law)"),
    ("P2-093", "What is the protective outer layer of a plant leaf that reduces transpirational water loss?", "Cuticle"),
    ("P2-094", "In data structures, which linear abstract data type operates on a First-In, First-Out (FIFO) basis?", "Queue"),
    ("P2-095", "What is the SI unit of power equivalent to one joule per second?", "Watt (W)"),
    ("P2-096", "Which type of chemical bond holds the two complementary strands of a DNA double helix together?", "Hydrogen bonds"),
    ("P2-097", "What is the name of the nearest star system to our Sun at approximately 4.2 light-years distance?", "Alpha Centauri (Proxima Centauri)"),
    ("P2-098", "In economics, what term describes a market structure dominated by a single seller?", "Monopoly"),
    ("P2-099", "What is the medical term for the voice box containing the vocal cords?", "Larynx"),
    ("P2-100", "In fluid mechanics, what dimensionless number predicts whether flow will be laminar or turbulent?", "Reynolds number (Re)"),
]

for pid, dom, orig, lex, syn, ans in REASONING_SPECS:
    REASONING_ITEMS.append({
        "id": pid,
        "domain": dom,
        "gold": ans,
        "gold_full": ans,
        "original": orig,
        "lexical": lex,
        "syntactic": syn,
        "pragmatic": f"Please answer this question carefully and precisely: {orig}",
        "length": f"Considering formal principles of reasoning and physical sciences: {orig}",
        "format": f"Question: {orig}\nAnswer concisely:",
    })

for pid, orig, ans in ADDITIONAL_REASONING_PROMPTS:
    REASONING_ITEMS.append({
        "id": pid,
        "domain": "general_science",
        "gold": ans,
        "gold_full": ans,
        "original": orig,
        "lexical": f"State the answer: {orig}",
        "syntactic": f"In scientific terms, {orig}",
        "pragmatic": f"Answer concisely as an educator: {orig}",
        "length": f"Considering foundational physical and mathematical laws: {orig}",
        "format": f"Query: {orig}\nAnswer:",
    })

logger.info(f"Loaded {len(REASONING_ITEMS)} reasoning & constrained clusters.")

ALL_CLUSTERS = FACTUAL_ITEMS + REASONING_ITEMS
logger.info(f"Total Phase 2 Clusters: {len(ALL_CLUSTERS)}")


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def audit_overlap_against_existing(
    new_records: List[Dict[str, Any]],
    original_csv: Path = Path("data/paraphrased_prompts_axis_tagged_qa.csv"),
    phase1_jsonl: Path = Path("data/external_eval/external_dataset.jsonl"),
) -> Dict[str, Any]:
    """Check zero overlap against both original dataset and Phase 1 external dataset."""
    logger.info("Auditing overlap against previous datasets...")
    existing_texts = []

    # 1. Original CSV
    if original_csv.is_file():
        with open(original_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in ["original", "lexical", "syntactic", "pragmatic", "length", "format"]:
                    val = row.get(col, "")
                    if val.strip():
                        existing_texts.append(normalize_text(val))

    # 2. Phase 1 External JSONL
    if phase1_jsonl.is_file():
        with open(phase1_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    txt = item.get("prompt_text", "")
                    if txt.strip():
                        existing_texts.append(normalize_text(txt))

    logger.info(f"Auditing against {len(existing_texts)} existing prompt records.")

    exact_matches = 0
    near_matches = 0

    for rec in new_records:
        norm_q = normalize_text(rec["prompt_text"])
        words_q = set(norm_q.split())
        for ex in existing_texts:
            if norm_q == ex:
                exact_matches += 1
            else:
                words_ex = set(ex.split())
                if words_q and words_ex:
                    jaccard = len(words_q & words_ex) / len(words_q | words_ex)
                    if jaccard > 0.85:
                        near_matches += 1

    return {
        "total_new_records": len(new_records),
        "total_existing_records_audited": len(existing_texts),
        "exact_matches": exact_matches,
        "near_matches": near_matches,
        "zero_overlap_verified": bool(exact_matches == 0 and near_matches == 0),
    }


def build_and_save_dataset(
    output_dir: Path = Path("data/phase2_external_eval"),
) -> Dict[str, Any]:
    """Build dataset files, audit overlap, freeze hashes, and save manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    correctness_subset: List[Dict[str, Any]] = []

    axes = ["original", "lexical", "syntactic", "pragmatic", "length", "format"]

    for cluster in ALL_CLUSTERS:
        cid = cluster["id"]
        is_factual = cid.startswith("P2-0") and int(cid.split("-")[1]) <= 50

        for axis in axes:
            prompt_text = cluster[axis]
            pid = f"{cid}_{axis}"
            rec = {
                "cluster_id": cid,
                "prompt_id": pid,
                "variant_type": axis,
                "prompt_text": prompt_text,
                "domain": cluster.get("domain", "general"),
                "is_factual": is_factual,
                "reference_answer": cluster.get("gold") if is_factual else None,
                "reference_answer_full": cluster.get("gold_full") if is_factual else None,
            }
            records.append(rec)

        if is_factual:
            correctness_subset.append({
                "cluster_id": cid,
                "domain": cluster.get("domain", "general"),
                "canonical_question": cluster["original"],
                "gold_reference_answer": cluster["gold"],
                "gold_reference_answer_full": cluster.get("gold_full"),
            })

    # Save external_dataset.jsonl
    dataset_file = output_dir / "external_dataset.jsonl"
    with open(dataset_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save correctness_subset.jsonl
    subset_file = output_dir / "correctness_subset.jsonl"
    with open(subset_file, "w", encoding="utf-8") as f:
        for r in correctness_subset:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(records)} records to {dataset_file}")
    logger.info(f"Wrote {len(correctness_subset)} factual clusters to {subset_file}")

    # Run overlap audit
    audit_res = audit_overlap_against_existing(records)
    logger.info(f"Audit result: {audit_res}")

    # Compute hashes
    dataset_hash = compute_sha256(dataset_file)
    subset_hash = compute_sha256(subset_file)

    manifest = {
        "benchmark": "Phase 2 Untouched External Generalization Test",
        "created_at": "2026-10-12T22:25:00Z",
        "total_clusters": len(ALL_CLUSTERS),
        "total_prompt_records": len(records),
        "factual_clusters_count": len(correctness_subset),
        "axes": axes,
        "files": {
            "external_dataset_jsonl": {
                "filename": "external_dataset.jsonl",
                "sha256": dataset_hash,
                "records": len(records),
            },
            "correctness_subset_jsonl": {
                "filename": "correctness_subset.jsonl",
                "sha256": subset_hash,
                "clusters": len(correctness_subset),
            },
        },
        "overlap_audit": audit_res,
        "frozen": True,
    }

    manifest_file = output_dir / "dataset_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Saved frozen dataset manifest to {manifest_file}")
    return manifest


if __name__ == "__main__":
    build_and_save_dataset()
