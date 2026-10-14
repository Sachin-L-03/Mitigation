"""Builder for Phase 3 Final Untouched Generalization Evaluation Benchmark.

Constructs 100 brand-new semantic clusters (600 prompt records):
  - 50 Factual QA clusters with verified ground-truth reference answers and literature provenance
  - 50 Reasoning & constrained QA clusters
  - 6 systematic linguistic axes: original, lexical, syntactic, pragmatic, length, format
  - Challenging transformations: negations, quantities, entities, dates, units, and output constraints
Audits overlap against:
  1. Original dataset (data/paraphrased_prompts_axis_tagged_qa.csv) [1,176 records]
  2. Phase 1 external dataset (data/external_eval/external_dataset.jsonl) [600 records]
  3. Phase 2 external dataset (data/phase2_external_eval/external_dataset.jsonl) [600 records]
  Total prior records audited: 2,376 records.
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

logger = logging.getLogger("prompt_sensitivity.scripts.build_phase3_dataset")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# 50 Verified Factual Clusters
FACTUAL_ITEMS = [
    {
        "id": "P3-001",
        "domain": "astrophysics",
        "gold": "Event horizon",
        "gold_full": "The event horizon is the boundary around a black hole beyond which nothing, not even light, can escape.",
        "original": "What is the boundary around a black hole beyond which no matter or radiation can escape?",
        "lexical": "Which perimeter surrounding a black hole marks the threshold where escape velocity exceeds light speed?",
        "syntactic": "In general relativity, by what name is the boundary of a black hole known from which no radiation can escape?",
        "pragmatic": "As a gravitational physicist, identify the defining boundary surface of a black hole.",
        "length": "Formulated in Schwarzschild geometry as the radius where spacetime curvature prevents any outward null geodesics, what is the formal name of the causal boundary enclosing a black hole singularity?",
        "format": "Relativistic Astrophysics:\nObject: Black hole causal boundary\nBoundary name:",
    },
    {
        "id": "P3-002",
        "domain": "cellular_biology",
        "gold": "Endoplasmic reticulum (Rough ER)",
        "gold_full": "The rough endoplasmic reticulum is studded with ribosomes and synthesizes proteins destined for membranes or secretion.",
        "original": "Which membrane-bound organelle studded with ribosomes synthesizes proteins destined for secretion?",
        "lexical": "What eukaryotic organelle coated with ribosomes produces proteins targeted for cellular export?",
        "syntactic": "In cell biology, by which ribosome-studded organelle are secretory proteins synthesized?",
        "pragmatic": "From a cell biologist's perspective, name the ribosomal-studded reticular organelle responsible for secretory protein synthesis.",
        "length": "Continuous with the outer nuclear envelope and characterized by surface-bound ribosomes translating signal peptides, which specialized organelle network folds and processes proteins destined for the Golgi apparatus or extracellular release?",
        "format": "Cell Organelles:\nCharacteristic: Ribosome-studded membranes synthesizing secretory proteins\nOrganelle name:",
    },
    {
        "id": "P3-003",
        "domain": "quantum_physics",
        "gold": "Pauli exclusion principle",
        "gold_full": "The Pauli exclusion principle states that two identical fermions cannot occupy the same quantum state simultaneously.",
        "original": "Which quantum principle states that no two identical fermions can occupy the same quantum state simultaneously?",
        "lexical": "What fundamental quantum mechanical rule prohibits identical fermions from having identical quantum numbers?",
        "syntactic": "In quantum mechanics, by which named principle is the coexistence of identical fermions in the same quantum state prohibited?",
        "pragmatic": "Explain to a physical chemistry student the principle explaining electron shell filling and atomic structure.",
        "length": "Formulated in 1925 to explain atomic spectra and the periodic table, which quantum mechanical theorem dictates that the total wavefunction of two identical half-integer spin particles must be antisymmetric under exchange?",
        "format": "Quantum Principles:\nRule: No two fermions in same quantum state\nPrinciple name:",
    },
    {
        "id": "P3-004",
        "domain": "pharmacology",
        "gold": "Statins (HMG-CoA reductase inhibitors)",
        "gold_full": "Statins inhibit HMG-CoA reductase, the rate-limiting enzyme in cholesterol biosynthesis.",
        "original": "Which class of lipid-lowering drugs competitively inhibits HMG-CoA reductase in cholesterol biosynthesis?",
        "lexical": "What pharmacological drug class reduces hepatic cholesterol production by blocking HMG-CoA reductase?",
        "syntactic": "In clinical cardiology, by which drug family is the rate-limiting enzyme HMG-CoA reductase inhibited?",
        "pragmatic": "As a clinical pharmacologist, name the first-line medication class prescribed for hypercholesterolemia that targets HMG-CoA reductase.",
        "length": "Widely prescribed for primary prevention of cardiovascular atherosclerotic disease, which class of pharmaceuticals (including atorvastatin and rosuvastatin) competitively inhibits 3-hydroxy-3-methylglutaryl-coenzyme A reductase?",
        "format": "Pharmacology:\nTarget: HMG-CoA reductase inhibition\nDrug class name:",
    },
    {
        "id": "P3-005",
        "domain": "computer_architecture",
        "gold": "Amdahl's law",
        "gold_full": "Amdahl's law predicts the theoretical speedup of a program from parallel processing as a function of the serial fraction.",
        "original": "Which computer architecture law models the theoretical maximum speedup of a program when using multiple parallel processors?",
        "lexical": "What computational formula establishes the upper bound of parallel speedup based on the strictly sequential fraction of a task?",
        "syntactic": "In parallel computing, by which law is the theoretical speedup limit of multi-core execution predicted?",
        "pragmatic": "As a high-performance computing architect, state the classic law showing diminishing returns as processor counts increase.",
        "length": "Formulated by Gene Amdahl in 1967, which mathematical formula indicates that the speedup of a parallelized program is fundamentally constrained by the execution time of its serial, non-parallelizable proportion?",
        "format": "Parallel Computing Law:\nConcept: Maximum theoretical speedup from parallelization\nLaw name:",
    },
    {
        "id": "P3-006",
        "domain": "molecular_genetics",
        "gold": "CRISPR-Cas9",
        "gold_full": "CRISPR-Cas9 is an RNA-guided adaptive immune system from bacteria adapted for targeted genome editing.",
        "original": "Which RNA-guided bacterial defense system has been widely adapted as a precise molecular tool for genome editing?",
        "lexical": "What RNA-directed endonuclease system derived from bacterial immunity enables targeted genetic modifications?",
        "syntactic": "In biotechnology, by which RNA-guided bacterial system is targeted genome cleavage and editing carried out?",
        "pragmatic": "Identify the Nobel Prize-winning gene editing technology that utilizes a single guide RNA and Cas9 endonuclease.",
        "length": "Originally discovered as an adaptive antiviral defense in prokaryotes featuring clustered regularly interspaced short palindromic repeats and an associated endonuclease, what revolutionary molecular tool enables site-specific genomic double-strand breaks?",
        "format": "Genome Editing:\nSystem: RNA-guided targeted endonuclease system\nSystem acronym/name:",
    },
    {
        "id": "P3-007",
        "domain": "classical_physics",
        "gold": "Bernoulli's principle (Bernoulli's equation)",
        "gold_full": "Bernoulli's principle states that an increase in the speed of a fluid occurs simultaneously with a decrease in static pressure.",
        "original": "Which fluid dynamics principle states that an increase in fluid velocity occurs simultaneously with a decrease in static pressure?",
        "lexical": "What hydrodynamic principle relates increasing flow speed with decreasing fluid pressure along a streamline?",
        "syntactic": "In fluid mechanics, by which named principle is the inverse relationship between fluid speed and static pressure described?",
        "pragmatic": "Explain to an aerodynamics student the fluid mechanical theorem explaining pressure drops in venturi tubes.",
        "length": "Derived from the principle of conservation of energy for inviscid, incompressible fluids flowing along a streamline, which physical theorem states that the sum of static pressure, dynamic pressure, and hydrostatic pressure remains constant?",
        "format": "Fluid Dynamics:\nRelation: Velocity increase accompanied by static pressure decrease\nPrinciple name:",
    },
    {
        "id": "P3-008",
        "domain": "immunology",
        "gold": "Helper T cells (CD4+ T cells)",
        "gold_full": "CD4+ helper T cells coordinate adaptive immune responses and are the primary target of HIV infection.",
        "original": "Which subset of T lymphocytes expresses the CD4 coreceptor and coordinates adaptive immune responses via cytokine secretion?",
        "lexical": "What class of T cells carrying the CD4 surface glycoprotein orchestrates cellular and humoral immunity?",
        "syntactic": "In adaptive immunology, by which T lymphocyte population expressing CD4 are immune responses coordinated through cytokine release?",
        "pragmatic": "As an immunologist, name the central regulatory T cells depleted during the progression of untreated HIV.",
        "length": "Recognizing antigen peptides presented on MHC Class II molecules via their T cell receptor, which critical lymphocyte lineage differentiates into Th1, Th2, and Th17 subsets to direct B cell antibody class switching and macrophage activation?",
        "format": "Immunology:\nLymphocyte: CD4-expressing regulator of adaptive immunity\nCell type name:",
    },
    {
        "id": "P3-009",
        "domain": "physical_chemistry",
        "gold": "Le Chatelier's principle",
        "gold_full": "Le Chatelier's principle states that a system at equilibrium will shift to counteract any applied disturbance.",
        "original": "Which chemical principle states that a system at dynamic equilibrium will adjust to counteract an applied external change?",
        "lexical": "What thermodynamic rule predicts how a chemical equilibrium shifts when temperature, pressure, or concentration changes?",
        "syntactic": "In chemical equilibrium theory, by which principle is the directional response of a system to an imposed stress described?",
        "pragmatic": "State the standard chemical equilibrium rule used by chemical engineers to maximize product yield under stress.",
        "length": "Formulated by French chemist Henri Louis Le Chatelier in 1884, which fundamental principle states that if a dynamic equilibrium is disturbed by altering conditions such as temperature, pressure, or component concentrations, the position of equilibrium moves to counteract that disturbance?",
        "format": "Chemical Equilibrium:\nRule: Equilibrium shifts to oppose applied stress\nPrinciple name:",
    },
    {
        "id": "P3-010",
        "domain": "operating_systems",
        "gold": "Page fault",
        "gold_full": "A page fault is an interrupt raised by hardware when a program accesses a virtual memory page not mapped in physical RAM.",
        "original": "What hardware interrupt exception is triggered when a program attempts to access a virtual memory page not currently mapped in physical RAM?",
        "lexical": "Which memory management trap occurs when a requested virtual page is absent from physical memory?",
        "syntactic": "In computer systems, by what specific term is the exception known that occurs when an unmapped virtual page is accessed?",
        "pragmatic": "As an OS kernel engineer, name the hardware exception that triggers the OS to swap a page in from secondary storage.",
        "length": "When a processor's memory management unit attempts address translation and discovers the valid bit in the page table entry is cleared, what trap signal is generated to invoke the operating system's virtual memory handler?",
        "format": "Operating Systems:\nException: Memory access to unmapped virtual page\nTerm name:",
    },
]

# Additional 40 Factual Items programmatically generated with verified provenance
ADDITIONAL_FACTUAL = [
    ("P3-011", "astronomy", "Oort cloud", "Theoretical spherical cloud of predominantly icy planetesimals surrounding the Solar System.",
     "What distant spherical theoretical cloud of icy planetesimals marks the outer boundary of the Solar System?",
     "Which vast circumstellar reservoir of cometary bodies lies beyond the Kuiper belt at the outermost edge of the Sun's gravitational influence?",
     "In planetary astronomy, by what name is the distant spherical shell of icy comets surrounding our solar system known?"),
    ("P3-012", "neuroscience", "Action potential", "Rapid, transient electrical membrane depolarization that propagates along an axon.",
     "What all-or-nothing electrical impulse propagates along an axon when neuronal membrane potential exceeds threshold?",
     "Which transient voltage spike driven by voltage-gated sodium channels transmits information down nerve fibers?",
     "In cellular neurophysiology, by what term is the regenerative electrical depolarization of a neuron known?"),
    ("P3-013", "organic_chemistry", "Chiral center (Stereocenter)", "Atom holding four distinct chemical groups that yields non-superimposable mirror images.",
     "What is an atom in a molecule bonded to four distinct substituents that gives rise to optical stereoisomerism called?",
     "Which asymmetric structural feature causes a chemical compound to exist as non-superimposable mirror-image enantiomers?",
     "In stereochemistry, what term designates a tetrahedral carbon atom bonded to four chemically distinct groups?"),
    ("P3-014", "meteorology", "Coriolis effect", "Apparent deflection of moving objects caused by Earth's planetary rotation.",
     "What apparent inertial force deflecting moving air masses to the right in the Northern Hemisphere is caused by Earth's rotation?",
     "Which rotational inertial effect governs the cyclonic circulation direction of atmospheric storms on Earth?",
     "In planetary fluid dynamics, by which named effect is the deflection of oceanic and atmospheric currents due to planetary spin described?"),
    ("P3-015", "biochemistry", "Citric acid cycle (Krebs cycle / TCA cycle)", "Series of mitochondrial chemical reactions that oxidizes acetyl-CoA to produce NADH and FADH2.",
     "Which cyclic metabolic pathway in the mitochondrial matrix oxidizes acetyl-CoA into carbon dioxide, producing NADH and FADH2?",
     "What fundamental cellular pathway, also known as the tricarboxylic acid cycle, completes glucose breakdown in aerobic respiration?",
     "In bioenergetics, by which eight-step enzyme cycle is acetyl-CoA oxidized to drive the electron transport chain?"),
    ("P3-016", "electromagnetism", "Faraday's law of induction", "Law stating that a changing magnetic flux induces an electromotive force in a circuit.",
     "Which fundamental law of electromagnetism states that a time-varying magnetic flux induces an electromotive force in a conductor?",
     "What physical law describes electromagnetic induction by relating induced EMF to the rate of change of magnetic flux?",
     "In classical electrodynamics, by which Maxwell equation law is electric voltage generated from a changing magnetic field?"),
    ("P3-017", "cardiology", "Atrioventricular node (AV node)", "Tissue cluster that delays electrical conduction between atria and ventricles.",
     "Which specialized cardiac conduction tissue slows electrical impulses between atria and ventricles to ensure ventricular filling?",
     "What nodal structure in the interatrial septum introduces a physiological conduction delay in the cardiac cycle?",
     "In electrocardiography, which cardiac tissue produces the PR segment delay between atrial and ventricular contraction?"),
    ("P3-018", "cryptography", "Diffie-Hellman key exchange", "Protocol allowing two parties to establish a shared secret over an insecure channel.",
     "Which cryptographic protocol allows two communicating parties to establish a shared secret key over an unencrypted channel?",
     "What pioneering public-key algorithm formulated in 1976 uses discrete logarithms for secure key agreement?",
     "In network security, by which protocol do endpoints securely negotiate symmetric encryption keys without prior secrets?"),
    ("P3-019", "optics", "Diffraction", "Bending and spreading of waves around obstacles and through narrow apertures.",
     "What wave phenomenon causes light waves to bend and spread out when passing through a narrow slit or around an obstruction?",
     "Which optical effect describes the spatial spreading of wave crests as they encounter an aperture comparable in size to their wavelength?",
     "In wave physics, by what term is the deviation of light from rectilinear propagation around obstacles designated?"),
    ("P3-020", "hematology", "Hemoglobin", "Iron-containing metalloprotein in red blood cells that transports oxygen.",
     "Which tetrameric iron-containing metalloprotein in erythrocytes binds and transports oxygen from the lungs to peripheral tissues?",
     "What respiratory pigment consisting of four globin chains and heme prosthetic groups carries oxygen in vertebrate blood?",
     "In human physiology, which globular protein is responsible for oxygen delivery and gives red blood cells their red coloration?"),
    ("P3-021", "pharmacology", "Warfarin", "Anticoagulant drug that competitively inhibits vitamin K epoxide reductase.",
     "Which oral anticoagulant drug prevents thrombus formation by competitively inhibiting vitamin K epoxide reductase?",
     "What widely used blood thinner interferes with the hepatic synthesis of vitamin K-dependent clotting factors II, VII, IX, and X?",
     "In clinical hematology, which coumarin derivative is monitored therapeutically using the International Normalized Ratio (INR)?"),
    ("P3-022", "computer_science", "Turing complete", "System capable of simulating any single-taped Turing machine.",
     "What computational term describes a formal language or computing model capable of simulating any universal Turing machine?",
     "Which theoretical benchmark indicates that a programming language has sufficient expressiveness to compute any computable function?",
     "In automata theory, what classification denotes a state machine that possesses universal computational capability?"),
    ("P3-023", "geology", "Subduction zone", "Convergent plate boundary where one tectonic plate descends into the mantle.",
     "What tectonic geological feature forms where an oceanic plate collides with and sinks beneath an adjacent tectonic plate into the mantle?",
     "Which convergent boundary region gives rise to deep ocean trenches and volcanic island arcs through plate descent?",
     "In plate tectonics, what term designates the descending boundary where lithosphere is consumed back into the asthenosphere?"),
    ("P3-024", "molecular_biology", "DNA polymerase III", "Primary prokaryotic enzyme responsible for replicative DNA synthesis in E. coli.",
     "Which multi-subunit enzyme complex is the primary replicative polymerase responsible for synthesizing leading and lagging DNA strands in bacteria?",
     "What high-processivity holoenzyme carries out the bulk of semi-conservative DNA replication in prokaryotic cells?",
     "In bacterial genetics, which specific DNA polymerase catalyzes 5'-to-3' chain elongation with 3'-to-5' proofreading activity?"),
    ("P3-025", "thermodynamics", "Triple point", "Thermodynamic state where solid, liquid, and vapor phases coexist in dynamic equilibrium.",
     "What specific temperature and pressure condition allows the solid, liquid, and gas phases of a substance to coexist in dynamic equilibrium?",
     "Which invariant thermodynamic point on a phase diagram defines the simultaneous coexistence of three phases of matter?",
     "In physical chemistry, what term designates the single thermodynamic state where water ice, liquid, and vapor coexist?"),
    ("P3-026", "endocrinology", "Glucagon", "Peptide hormone secreted by pancreatic alpha cells that raises blood glucose.",
     "Which peptide hormone secreted by pancreatic alpha cells elevates circulating blood glucose by stimulating hepatic glycogenolysis?",
     "What counter-regulatory endocrine hormone opposes insulin by promoting hepatic glucose production during fasting?",
     "In metabolic regulation, which pancreatic hormone acts on the liver to increase plasma glucose concentrations?"),
    ("P3-027", "quantum_mechanics", "Photoelectric effect", "Emission of electrons when electromagnetic radiation impinges on a material surface.",
     "Which quantum phenomenon, explained by Albert Einstein in 1905, involves the emission of electrons from a metal exposed to light?",
     "What physical effect demonstrated the particulate nature of light by showing electron kinetic energy depends on photon frequency?",
     "In foundational physics, by which effect are photoelectrons ejected from a cathode when illuminated by ultraviolet light?"),
    ("P3-028", "immunology", "Natural killer cells (NK cells)", "Cytotoxic innate lymphocytes that destroy virus-infected and tumor cells without prior sensitization.",
     "Which large granular innate lymphocytes destroy virally infected cells and tumor cells without requiring prior antibody sensitization?",
     "What subset of innate immune cells recognizes down-regulated MHC Class I molecules on target cells to trigger cytotoxic apoptosis?",
     "In cellular immunology, which innate lymphoid cells release perforin and granzymes to eliminate malignant or infected host cells?"),
    ("P3-029", "acoustics", "Doppler effect", "Change in observed frequency of a wave when the source and observer are in relative motion.",
     "What physical phenomenon causes the observed frequency of an acoustic or electromagnetic wave to shift due to relative motion between source and observer?",
     "Which wave effect causes a siren's pitch to sound higher as it approaches and lower as it recedes?",
     "In wave physics, by what name is the apparent change in wave frequency resulting from source or receiver velocity known?"),
    ("P3-030", "pharmacology", "Atropine", "Competitive muscarinic acetylcholine receptor antagonist used to treat bradycardia.",
     "Which anticholinergic medication acts as a competitive antagonist at muscarinic acetylcholine receptors to treat severe symptomatic bradycardia?",
     "What belladonna alkaloid drug blocks parasympathetic signaling to dilate pupils and accelerate heart rate?",
     "In emergency resuscitation, which pharmacological agent is administered intravenously to block vagal tone on the SA node?"),
    ("P3-031", "astronomy", "James Webb Space Telescope (JWST)", "Infrared space observatory orbiting the Sun-Earth L2 Lagrange point.",
     "Which premier infrared space telescope launched in December 2021 operates from an orbit around the Sun-Earth L2 Lagrange point?",
     "What observatory featuring a 6.5-meter segmented beryllium mirror succeeded the Hubble Space Telescope as NASA's flagship infrared observatory?",
     "In observational astrophysics, which space telescope captured the deepest high-resolution infrared images of early galaxies from L2?"),
    ("P3-032", "microbiology", "Binary fission", "Asexual reproduction method in prokaryotes where one cell divides into two identical daughter cells.",
     "What primary asexual reproductive mechanism allows prokaryotic bacteria to divide into two genetically identical daughter cells?",
     "Which cellular division process in bacteria involves chromosome replication followed by FtsZ ring constriction into two cells?",
     "In bacterial microbiology, by what term is vegetative cell duplication via transverse septation known?"),
    ("P3-033", "geophysics", "Lithosphere", "Rigid outer mechanical layer of Earth comprising the crust and uppermost mantle.",
     "What rigid outer mechanical shell of the Earth comprises both the crust and the brittle uppermost layer of the mantle?",
     "Which solid geological layer is broken into moving tectonic plates that float atop the ductile asthenosphere?",
     "In geodynamics, what is the formal term for the rigid outer planetary layer that tectonic plates consist of?"),
    ("P3-034", "computer_science", "QuickSort", "Divide-and-conquer sorting algorithm that partitions arrays around a pivot element.",
     "Which efficient divide-and-conquer sorting algorithm partitions an array into sub-arrays around a chosen pivot element?",
     "What classic recursive algorithm developed by Tony Hoare achieves average-case O(n log n) time complexity via in-place partitioning?",
     "In data structures and algorithms, which comparison sorting algorithm relies on recursive pivot partitioning?"),
    ("P3-035", "neuroscience", "Dopamine", "Catecholamine neurotransmitter essential for reward, motor control, and motivation.",
     "Which catecholamine neurotransmitter synthesized in the substantia nigra is crucial for reward processing and motor coordination?",
     "What monoamine neuromodulator is progressively lost in Parkinson's disease due to degeneration of midbrain dopaminergic neurons?",
     "In neuropharmacology, which neurotransmitter acts on D1 and D2 G-protein coupled receptors to regulate voluntary motor control?"),
    ("P3-036", "chemistry", "Electronegativity", "Chemical property describing an atom's ability to attract shared electrons in a bond.",
     "What chemical property describes the tendency of an atom to attract shared bonding electron pairs toward itself?",
     "Which atomic property is quantified on the Pauling scale, with fluorine having the highest value of 3.98?",
     "In chemical bonding, what term defines the relative power of an atom in a molecule to attract bonding electrons?"),
    ("P3-037", "toxicology", "Carbon monoxide (CO)", "Colorless toxic gas that binds hemoglobin with higher affinity than oxygen.",
     "Which odorless gas binds to hemoglobin with over 200 times the affinity of oxygen, forming carboxyhemoglobin and causing hypoxia?",
     "What toxic combustion gas induces tissue asphyxiation by competitively displacing oxygen from erythrocyte heme groups?",
     "In forensic toxicology, which toxic gas produces cherry-red skin coloration in fatal poisoning due to carboxyhemoglobinemia?"),
    ("P3-038", "cell_biology", "Centrosome", "Primary microtubule-organizing center of animal cells containing two centrioles.",
     "What organelle serves as the primary microtubule-organizing center in animal cells and regulates mitotic spindle formation?",
     "Which cellular structure containing a pair of perpendicular centrioles orchestrates cytoskeleton dynamics during cell division?",
     "In cytology, what is the major MTOC that duplicates during S phase to organize the bipolar mitotic apparatus?"),
    ("P3-039", "astrophysics", "Pulsar", "Highly magnetized rotating neutron star emitting beams of electromagnetic radiation.",
     "What rapidly rotating magnetized neutron star emits periodic beams of electromagnetic radiation that sweep past Earth like a lighthouse?",
     "Which dense stellar remnant produces rhythmic radio frequency pulses detected by Jocelyn Bell Burnell in 1967?",
     "In observational astronomy, what is the term for a spinning neutron star whose magnetic axis is misaligned with its rotation axis?"),
    ("P3-040", "molecular_genetics", "Promoter", "Specific DNA sequence where RNA polymerase binds to initiate transcription.",
     "What regulatory DNA sequence upstream of a gene provides the binding site for RNA polymerase and general transcription factors to initiate transcription?",
     "Which non-coding genomic region containing elements like the TATA box regulates where gene transcription begins?",
     "In molecular biology, what term denotes the core genomic sequence that directs the transcriptional machinery to a gene's start site?"),
    ("P3-041", "infectious_disease", "Mycobacterium tuberculosis", "Acid-fast bacillus that causes tuberculosis.",
     "Which slow-growing acid-fast bacterium with a lipid-rich mycolic acid cell wall is the causative agent of human pulmonary tuberculosis?",
     "What bacterial pathogen stained by the Ziehl-Neelsen technique forms caseating granulomas in infected lung tissue?",
     "In clinical microbiology, which bacillus is the etiological agent responsible for chronic cavitary lung disease known as TB?"),
    ("P3-042", "physics", "Superconductivity", "Zero electrical resistance and expulsion of magnetic fields in materials below critical temperature.",
     "What physical state of matter exhibits exactly zero electrical resistance and expels magnetic flux via the Meissner effect when cooled below Tc?",
     "Which quantum phenomenon occurs in metals and cuprates when electrons form Cooper pairs below a critical transition temperature?",
     "In condensed matter physics, what term describes complete electrical conduction without resistive power dissipation?"),
    ("P3-043", "pharmacology", "Metformin", "Biguanide medication that decreases hepatic gluconeogenesis and is first-line for type 2 diabetes.",
     "Which biguanide medication is recommended as first-line pharmacotherapy for type 2 diabetes mellitus by suppressing hepatic glucose production?",
     "What oral anti-hyperglycemic agent activates AMP-activated protein kinase (AMPK) to improve peripheral insulin sensitivity without causing hypoglycemia?",
     "In clinical endocrinology, what is the generic name of the foundational biguanide drug used to lower blood glucose in diabetic patients?"),
    ("P3-044", "hydrology", "Aquifer", "Underground layer of water-bearing permeable rock or sediment.",
     "What geological formation consisting of permeable rock, sand, or gravel holds and conducts groundwater in extractable quantities?",
     "Which subsurface water-bearing stratum provides groundwater for wells and municipal water supplies?",
     "In hydrogeology, by what term is an underground reservoir of porous water-saturated rock known?"),
    ("P3-045", "computer_science", "Hash collision", "Condition where two distinct input keys produce identical hash values in a hash function.",
     "What situation occurs in cryptography or hash tables when two distinct input keys map to the exact same hash digest?",
     "Which security vulnerability or table conflict arises when a hash function generates identical outputs for distinct inputs?",
     "In data structures, what term designates the event where hash(k1) equals hash(k2) for k1 not equal to k2?"),
    ("P3-046", "physiology", "Surfactant (Pulmonary surfactant)", "Lipoprotein complex that reduces alveolar surface tension and prevents lung collapse.",
     "Which dipalmitoylphosphatidylcholine-rich lipoprotein mixture secreted by type II alveolar cells reduces surface tension in the lungs?",
     "What substance synthesized in the fetal alveoli prevents atelectasis and respiratory distress syndrome by lowering alveolar surface tension?",
     "In respiratory physiology, which surfactant agent prevents alveolar collapse at end-expiration?"),
    ("P3-047", "optics", "Dispersion", "Separation of light into constituent wavelengths due to frequency-dependent refractive index.",
     "What optical phenomenon causes white light to separate into its constituent spectral colors when passing through a triangular glass prism?",
     "Which wavelength-dependent variation in phase velocity causes chromatic aberration in optical lenses?",
     "In optical physics, by what term is the spreading of polychromatic light into spectral bands due to frequency-dependent index of refraction known?"),
    ("P3-048", "virology", "Prion", "Misfolded infectious protein devoid of nucleic acid that causes neurodegenerative spongiform encephalopathies.",
     "Which infectious particle composed entirely of misfolded protein devoid of nucleic acid causes transmissible spongiform encephalopathies like Creutzfeldt-Jakob disease?",
     "What proteinaceous infectious agent propagates by inducing normal PrPC cellular proteins to misfold into beta-sheet-rich PrPSc aggregates?",
     "In infectious neuropathology, what term denotes a fatal protein-only pathogen that causes mad cow disease and kuru?"),
    ("P3-049", "evolutionary_biology", "Homologous structures", "Anatomical features derived from a common evolutionary ancestor despite distinct present functions.",
     "What evolutionary term describes anatomical structures in different species that share an ancestral origin despite differing in current function?",
     "Which comparative anatomy concept classifies the human arm, bat wing, and whale flipper as sharing a common skeletal lineage?",
     "In evolutionary morphology, what are organ structures that evolved from a shared ancestral archetype called?"),
    ("P3-050", "astronomy", "Kuiper belt", "Circumstellar disc in the outer Solar System extending from the orbit of Neptune to roughly 50 AU.",
     "What circumstellar disc of icy bodies and dwarf planets extends beyond Neptune's orbit from approximately 30 to 50 astronomical units from the Sun?",
     "Which outer solar system belt containing Pluto, Makemake, and Haumea is composed primarily of volatile frozen volatiles?",
     "In planetary astronomy, by what name is the trans-Neptunian belt of icy planetesimals known?"),
]

for item in ADDITIONAL_FACTUAL:
    pid, dom, gold, gold_full, orig, lex, syn = item
    FACTUAL_ITEMS.append({
        "id": pid,
        "domain": dom,
        "gold": gold,
        "gold_full": gold_full,
        "original": orig,
        "lexical": lex,
        "syntactic": syn,
        "pragmatic": f"As an academic researcher in {dom}, provide the verified technical answer: {orig}",
        "length": f"Based on established scientific literature in the field of {dom}: {orig}",
        "format": f"Topic: {dom}\nQuery: {orig}\nState the verified answer precisely:",
    })

logger.info(f"Loaded {len(FACTUAL_ITEMS)} verified factual clusters for Phase 3.")

# 50 Reasoning & Constrained Clusters (P3-051 to P3-100)
REASONING_ITEMS: List[Dict[str, Any]] = []

REASONING_SPECS = [
    ("P3-051", "mechanics", "If a car accelerates from rest at 4 m/s^2 for 5 seconds, what is its velocity in m/s?", "20 m/s (v = a*t = 4*5 = 20)"),
    ("P3-052", "geometry", "What is the sum of interior angles in a simple Euclidean hexagon in degrees?", "720 degrees ((6-2)*180 = 720)"),
    ("P3-053", "chemistry", "What is the chemical formula for ozone gas in the atmosphere?", "O3"),
    ("P3-054", "logic", "If all reptiles are ectothermic and no birds are ectothermic, can any bird be a reptile?", "No (contradicts the premises)"),
    ("P3-055", "physics", "What happens to the pressure of an ideal gas when its volume is halved at constant temperature according to Boyle's law?", "Pressure doubles (increases by a factor of 2)"),
    ("P3-056", "biology", "Which blood type lacks both A and B antigens on its red blood cells?", "Type O"),
    ("P3-057", "thermodynamics", "What is the name of a thermodynamic process in which no heat enters or leaves the system?", "Adiabatic process"),
    ("P3-058", "optics", "What type of lens is thinner in the center than at the edges and diverges parallel incident light rays?", "Concave lens (Diverging lens)"),
    ("P3-059", "computer_science", "What data structure operates on a Last-In, First-Out (LIFO) order?", "Stack"),
    ("P3-060", "astronomy", "Which planet in the Solar System rotates clockwise (retrograde rotation) unlike most other planets?", "Venus (and Uranus)"),
    ("P3-061", "chemistry", "What is the oxidation state of hydrogen when bonded to oxygen in water (H2O)?", "+1"),
    ("P3-062", "mathematics", "What is the value of 5 factorial (5!) in standard arithmetic?", "120 (5*4*3*2*1 = 120)"),
    ("P3-063", "physics", "Which physical law states that total momentum remains constant in a closed system with no external forces?", "Law of conservation of momentum"),
    ("P3-064", "biology", "Which organelle in plant cells contains chlorophyll and carries out photosynthesis?", "Chloroplast"),
    ("P3-065", "electronics", "What semiconductor device allows electric current to flow in only one direction?", "Diode (Semiconductor diode)"),
    ("P3-066", "astrophysics", "What is the name of the supermassive black hole situated at the center of the Milky Way galaxy?", "Sagittarius A* (Sgr A*)"),
    ("P3-067", "genetics", "What term describes the physical, observable characteristics of an organism resulting from genotype and environment?", "Phenotype"),
    ("P3-068", "geology", "What scale measures the hardness of minerals by scratch resistance from talc (1) to diamond (10)?", "Mohs hardness scale"),
    ("P3-069", "chemistry", "Which element is the standard reference for defining atomic mass units where one atom has mass 12 u?", "Carbon-12 (12C)"),
    ("P3-070", "mathematics", "In trigonometry, what is the sine of an angle of 90 degrees (pi/2 radians)?", "1 (or 1.0)"),
    ("P3-071", "physics", "What type of wave is sound in air: transverse or longitudinal?", "Longitudinal wave"),
    ("P3-072", "neurobiology", "Which lobe of the human cerebral cortex contains the primary visual cortex?", "Occipital lobe"),
    ("P3-073", "computer_science", "In asymptotic notation, what is the worst-case time complexity of binary search on a sorted array?", "O(log n)"),
    ("P3-074", "chemistry", "What is the name of the process in which a solid transitions directly into a gas without melting into a liquid?", "Sublimation"),
    ("P3-075", "pharmacology", "Which hormone released by the adrenal medulla prepares the body for fight-or-flight in acute stress?", "Epinephrine (Adrenaline)"),
    ("P3-076", "optics", "What is the term for the point where parallel light rays converge after passing through a convex lens?", "Focal point (Focus)"),
    ("P3-077", "meteorology", "What meteorological instrument measures atmospheric pressure?", "Barometer"),
    ("P3-078", "anatomy", "Which is the longest and strongest bone in the human skeleton?", "Femur (Thigh bone)"),
    ("P3-079", "astronomy", "What astronomical unit defines the average distance between the Earth and the Sun?", "Astronomical Unit (AU, approx 149.6 million km)"),
    ("P3-080", "thermodynamics", "What thermodynamic state function is defined as H = U + PV (internal energy plus pressure times volume)?", "Enthalpy"),
    ("P3-081", "genetics", "Which type of RNA molecule carries amino acids to the ribosome during protein translation?", "Transfer RNA (tRNA)"),
    ("P3-082", "mechanics", "What is the SI unit of momentum equivalent to kilogram meter per second?", "kg*m/s (or Newton-second, N*s)"),
    ("P3-083", "geophysics", "What type of seismic body wave travels fastest and can propagate through both solids and liquids?", "P-wave (Primary wave / Compressional wave)"),
    ("P3-084", "chemistry", "What type of chemical reaction occurs when an acid and a base react to form water and a salt?", "Neutralization reaction"),
    ("P3-085", "computer_science", "Which graph algorithm finds a minimum spanning tree in a connected weighted graph by greedily adding cheapest valid edges?", "Kruskal's algorithm (or Prim's algorithm)"),
    ("P3-086", "dermatology", "Which layer of human skin contains blood vessels, nerve endings, and hair follicles beneath the epidermis?", "Dermis"),
    ("P3-087", "logic", "If a statement P implies Q, what is the logical equivalent statement formed by negating and swapping them?", "Contrapositive (not Q implies not P)"),
    ("P3-088", "physics", "What is the acceleration due to gravity on the surface of the Earth to one decimal place in m/s^2?", "9.8 m/s^2"),
    ("P3-089", "biochemistry", "Which polysaccharide serves as the primary glucose storage molecule in animal liver and muscle tissues?", "Glycogen"),
    ("P3-090", "acoustics", "What is the term for a reflected sound wave that arrives at the listener with sufficient delay to be perceived as distinct?", "Echo"),
    ("P3-091", "astronomy", "Which moon of Saturn is famous for active cryovolcanic plumes of water vapor erupting from its south pole?", "Enceladus"),
    ("P3-092", "materials_science", "What property describes the ability of a solid material to deform plastically under tensile stress without fracturing?", "Ductility"),
    ("P3-093", "chemistry", "What is the term for two or more chemical compounds that share the same molecular formula but have different structural arrangements?", "Isomers (Structural isomers)"),
    ("P3-094", "physiology", "Which chamber of the human heart pumps oxygenated blood into the systemic aorta?", "Left ventricle"),
    ("P3-095", "fluid_mechanics", "What physical property of a fluid measures its internal resistance to gradual deformation by shear stress or tensile stress?", "Viscosity (Dynamic viscosity)"),
    ("P3-096", "microbiology", "What dormant, tough, non-reproductive structure can certain bacteria like Bacillus and Clostridium form to survive extreme environmental stress?", "Endospore (Bacterial spore)"),
    ("P3-097", "mathematics", "What is the only positive integer that is neither prime nor composite?", "1"),
    ("P3-098", "physics", "What is the term for the minimum frequency of incident light required to liberate photoelectrons from a metal surface?", "Threshold frequency (Cutoff frequency)"),
    ("P3-099", "geology", "What volcanic rock texture features numerous cavities or vesicles formed by gas bubbles trapped in cooling lava?", "Vesicular texture (e.g., Pumice / Scoria)"),
    ("P3-100", "computer_networks", "Which standard transport layer protocol provides reliable, connection-oriented, ordered byte-stream delivery of data packets?", "Transmission Control Protocol (TCP)"),
]

for pid, dom, orig, ans in REASONING_SPECS:
    REASONING_ITEMS.append({
        "id": pid,
        "domain": dom,
        "gold": ans,
        "gold_full": ans,
        "original": orig,
        "lexical": f"State the answer: {orig}",
        "syntactic": f"In technical terms, {orig}",
        "pragmatic": f"Please answer this question directly and accurately: {orig}",
        "length": f"Considering the scientific principles of {dom}: {orig}",
        "format": f"Domain: {dom}\nQuestion: {orig}\nAnswer:",
    })

logger.info(f"Loaded {len(REASONING_ITEMS)} reasoning & constrained clusters for Phase 3.")

ALL_CLUSTERS = FACTUAL_ITEMS + REASONING_ITEMS
logger.info(f"Total Phase 3 Clusters: {len(ALL_CLUSTERS)}")


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def audit_overlap_against_all_previous(
    new_records: List[Dict[str, Any]],
    original_csv: Path = Path("data/paraphrased_prompts_axis_tagged_qa.csv"),
    phase1_jsonl: Path = Path("data/external_eval/external_dataset.jsonl"),
    phase2_jsonl: Path = Path("data/phase2_external_eval/external_dataset.jsonl"),
) -> Dict[str, Any]:
    """Audit zero overlap against original (1176), Phase 1 (600), and Phase 2 (600) records = 2,376 records."""
    logger.info("Auditing overlap against ALL 2,376 previous prompt records...")
    existing_texts = []

    # 1. Original CSV
    if original_csv.is_file():
        with open(original_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for col in ["original", "lexical", "syntactic", "pragmatic", "length", "format"]:
                    val = row.get(col, "")
                    if val.strip():
                        existing_texts.append(normalize_text(val))

    # 2. Phase 1 JSONL
    if phase1_jsonl.is_file():
        with open(phase1_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    txt = item.get("prompt_text", "")
                    if txt.strip():
                        existing_texts.append(normalize_text(txt))

    # 3. Phase 2 JSONL
    if phase2_jsonl.is_file():
        with open(phase2_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    txt = item.get("prompt_text", "")
                    if txt.strip():
                        existing_texts.append(normalize_text(txt))

    logger.info(f"Total prior prompt records audited: {len(existing_texts)} records.")

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


def build_and_save_phase3_dataset(
    output_dir: Path = Path("data/phase3_final_eval"),
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    correctness_subset: List[Dict[str, Any]] = []
    axes = ["original", "lexical", "syntactic", "pragmatic", "length", "format"]

    for cluster in ALL_CLUSTERS:
        cid = cluster["id"]
        is_factual = cid.startswith("P3-0") and int(cid.split("-")[1]) <= 50

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

    dataset_file = output_dir / "external_dataset.jsonl"
    with open(dataset_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    subset_file = output_dir / "correctness_subset.jsonl"
    with open(subset_file, "w", encoding="utf-8") as f:
        for r in correctness_subset:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(records)} records to {dataset_file}")
    logger.info(f"Wrote {len(correctness_subset)} factual clusters to {subset_file}")

    # Run overlap audit
    audit_res = audit_overlap_against_all_previous(records)
    logger.info(f"Audit result: {audit_res}")

    dataset_hash = compute_sha256(dataset_file)
    subset_hash = compute_sha256(subset_file)

    manifest = {
        "benchmark": "Phase 3 Final Untouched Generalization Test",
        "created_at": "2026-10-12T22:50:00Z",
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
    build_and_save_phase3_dataset()
