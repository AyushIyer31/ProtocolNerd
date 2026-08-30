#!/usr/bin/env python3
"""
Fetch and cache public protocols from protocols.io into data/protocols/.

Each protocol is saved as data/protocols/<id>.json.
A combined index is written to data/protocols_index.json.

Usage:
    python fetch_protocols.py
    python fetch_protocols.py --keywords "RNA extraction" "PCR" "western blot"
    python fetch_protocols.py --max-per-keyword 50 --output-dir data/protocols

Requires PROTOCOLS_IO_TOKEN in protocolnerd-backend/variables.env (or as env var).
Get a free CLIENT_ACCESS_TOKEN at: https://www.protocols.io/developers
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import urllib.request
import urllib.parse
import urllib.error

# Verify TLS against certifi's CA bundle — the system store fails in some
# managed/sandboxed environments (CERTIFICATE_VERIFY_FAILED).
try:
    import ssl
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:  # pragma: no cover
    _SSL_CTX = None

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Load token from variables.env
load_dotenv(Path(__file__).parent.parent / "protocolnerd-backend" / "variables.env", override=False)

API_BASE = "https://www.protocols.io/api/v3"
RATE_LIMIT_DELAY = 0.65  # ~92 req/min, safely under the 100/min limit
PAGE_SIZE = 50  # protocols per list request; small = reliable on flaky networks
ORDER_FIELD = "activity"  # 'id' gives a STABLE order (no drift during long crawls)
ORDER_DIR = "desc"

# Comprehensive biology keyword list — covers all major protocol categories
DEFAULT_KEYWORDS = [
    # -----------------------------------------------------------------------
    # CRITICAL: Prof. Shasha's 3 benchmark query topics (currently under-covered)
    # -----------------------------------------------------------------------
    "drought tolerance",
    "drought stress",
    "drought tolerance rice",
    "drought stress plant",
    "drought stress Arabidopsis",
    "osmotic stress tolerance",
    "water deficit plant",
    "dehydration tolerance",
    "salt stress tolerance",
    "heat stress plant",
    "abiotic stress tolerance",
    "rice drought",
    "rice stress",
    "Oryza sativa stress",
    "Oryza sativa transformation",
    "rice transformation",
    "rice CRISPR",
    "rice gene editing",
    "rice RNA extraction",
    "rice protein extraction",
    "rice phenotyping",
    "rice seed germination",
    "rice callus",
    "agrobacterium infiltration",
    "agrobacterium transformation plant",
    "agroinfiltration",
    "in planta transformation",
    "floral dip transformation",
    "vacuum infiltration plant",
    "transient expression plant",
    "in planta gene editing",
    "multiplex CRISPR",
    "multiplex gene editing",
    "multiplex genome editing",
    "combinatorial CRISPR",
    "simultaneous gene knockout",
    "multiple gene knockout",
    "multiplex knockout",
    "CRISPR multiplex mouse",
    "multiplex transcription factor",
    "transcription factor binding",
    "transcription factor mouse",
    "transcription factor plant",
    "gene regulation mouse",
    # -----------------------------------------------------------------------
    # RNA — molecular biology core
    # -----------------------------------------------------------------------
    "RNA extraction",
    "total RNA extraction",
    "RNA isolation",
    "RNA isolation plant",
    "RNA isolation tissue",
    "RNA isolation bacteria",
    "RNA isolation yeast",
    "RNA isolation blood",
    "RNA isolation FFPE",
    "mRNA isolation",
    "mRNA extraction",
    "small RNA isolation",
    "microRNA extraction",
    "RNA purification",
    "RNA quality assessment",
    "TRIzol RNA extraction",
    "CTAB RNA extraction",
    "RNeasy RNA extraction",
    "RNA-seq library preparation",
    "single cell RNA sequencing",
    "bulk RNA sequencing",
    "cDNA synthesis",
    "reverse transcription",
    "RT-PCR",
    "quantitative RT-PCR",
    # -----------------------------------------------------------------------
    # DNA
    # -----------------------------------------------------------------------
    "DNA extraction",
    "genomic DNA extraction",
    "DNA isolation plant",
    "DNA isolation tissue",
    "DNA isolation blood",
    "DNA isolation bacteria",
    "CTAB DNA extraction",
    "phenol chloroform extraction",
    "plasmid extraction",
    "plasmid purification",
    "miniprep",
    "maxiprep",
    "DNA gel electrophoresis",
    "agarose gel electrophoresis",
    "DNA quantification",
    "DNA library preparation",
    "bisulfite sequencing",
    "whole genome sequencing",
    "ChIP-seq",
    "ATAC-seq",
    "Hi-C",
    # -----------------------------------------------------------------------
    # PCR variants
    # -----------------------------------------------------------------------
    "PCR amplification",
    "qPCR",
    "quantitative PCR",
    "colony PCR",
    "site-directed mutagenesis",
    "overlap extension PCR",
    "digital PCR",
    "droplet digital PCR",
    "long range PCR",
    # -----------------------------------------------------------------------
    # CRISPR and gene editing
    # -----------------------------------------------------------------------
    "CRISPR Cas9",
    "CRISPR Cas12a",
    "CRISPR knockout",
    "CRISPR knockin",
    "CRISPR guide RNA",
    "guide RNA design",
    "base editing",
    "prime editing",
    "CRISPRa",
    "CRISPRi",
    "CRISPR screen",
    "CRISPR mammalian cells",
    "CRISPR plant",
    "CRISPR zebrafish",
    "CRISPR mouse",
    "homology directed repair",
    "HDR template",
    "electroporation CRISPR",
    # -----------------------------------------------------------------------
    # Protein work
    # -----------------------------------------------------------------------
    "protein extraction",
    "protein extraction plant",
    "protein extraction tissue",
    "protein purification",
    "recombinant protein expression",
    "protein expression E. coli",
    "protein expression yeast",
    "His-tag purification",
    "GST purification",
    "affinity chromatography",
    "size exclusion chromatography",
    "gel filtration",
    "protein concentration",
    "western blot",
    "western blotting",
    "SDS-PAGE",
    "2D gel electrophoresis",
    "co-immunoprecipitation",
    "immunoprecipitation",
    "pull-down assay",
    "protein interaction",
    "ELISA",
    "sandwich ELISA",
    "protein quantification",
    "Bradford assay",
    "BCA assay",
    "mass spectrometry proteomics",
    "proteomics sample preparation",
    # -----------------------------------------------------------------------
    # Cell biology
    # -----------------------------------------------------------------------
    "cell culture",
    "mammalian cell culture",
    "primary cell culture",
    "cell line maintenance",
    "cell passaging",
    "cell counting",
    "cell viability",
    "MTT assay",
    "cell proliferation assay",
    "apoptosis assay",
    "flow cytometry",
    "FACS sorting",
    "cell cycle analysis",
    "transfection",
    "lipofection",
    "calcium phosphate transfection",
    "electroporation",
    "lentiviral transduction",
    "adeno-associated virus",
    "retroviral transduction",
    "stable cell line",
    "cell freezing",
    "cryopreservation",
    "thawing cells",
    # -----------------------------------------------------------------------
    # Microscopy and imaging
    # -----------------------------------------------------------------------
    "immunofluorescence",
    "immunohistochemistry",
    "confocal microscopy",
    "fluorescence microscopy",
    "live cell imaging",
    "super resolution microscopy",
    "TIRF microscopy",
    "electron microscopy",
    "transmission electron microscopy",
    "scanning electron microscopy",
    "cryo-EM",
    "cryo-EM sample preparation",
    "calcium imaging",
    "GFP imaging",
    # -----------------------------------------------------------------------
    # Cloning and molecular tools
    # -----------------------------------------------------------------------
    "molecular cloning",
    "restriction cloning",
    "Gibson assembly",
    "Golden Gate cloning",
    "Gateway cloning",
    "ligation",
    "restriction digest",
    "bacterial transformation",
    "competent cells",
    "E. coli expression",
    "yeast transformation",
    "yeast two-hybrid",
    "two-hybrid assay",
    "reporter assay",
    "luciferase assay",
    # -----------------------------------------------------------------------
    # Plant biology — comprehensive
    # -----------------------------------------------------------------------
    "plant transformation",
    "plant cell culture",
    "plant tissue culture",
    "plant regeneration",
    "callus induction",
    "shoot regeneration",
    "Arabidopsis",
    "Arabidopsis thaliana",
    "Arabidopsis transformation",
    "Arabidopsis RNA",
    "Arabidopsis protein",
    "Arabidopsis CRISPR",
    "Arabidopsis phenotyping",
    "Arabidopsis seedling",
    "Nicotiana benthamiana",
    "tobacco transformation",
    "maize transformation",
    "wheat transformation",
    "soybean transformation",
    "tomato transformation",
    "potato transformation",
    "barley transformation",
    "plant RNA extraction",
    "plant protein extraction",
    "plant DNA extraction",
    "plant phenotyping",
    "plant hormone",
    "auxin",
    "cytokinin",
    "gibberellin",
    "plant pathogen",
    "plant immunity",
    "plant defense",
    "chloroplast isolation",
    "chloroplast transformation",
    "mitochondria isolation plant",
    "plant protoplast",
    "plant cell wall",
    "stomata measurement",
    "photosynthesis measurement",
    "chlorophyll extraction",
    "seed germination",
    "root growth",
    "plant hormone measurement",
    # -----------------------------------------------------------------------
    # Model organisms
    # -----------------------------------------------------------------------
    "zebrafish protocol",
    "zebrafish embryo",
    "zebrafish CRISPR",
    "zebrafish injection",
    "Drosophila protocol",
    "Drosophila CRISPR",
    "Drosophila genetics",
    "C. elegans protocol",
    "C. elegans CRISPR",
    "C. elegans genetics",
    "mouse protocol",
    "mouse dissection",
    "mouse tissue collection",
    "mouse genotyping",
    "mouse behavior",
    "mouse knockout",
    "rat protocol",
    "yeast Saccharomyces",
    "yeast genetics",
    "yeast protein expression",
    # -----------------------------------------------------------------------
    # Sequencing and genomics
    # -----------------------------------------------------------------------
    "next generation sequencing",
    "Illumina sequencing",
    "Oxford Nanopore sequencing",
    "PacBio sequencing",
    "library preparation sequencing",
    "adapter ligation",
    "ChIP-seq library",
    "ATAC-seq library",
    "CUT&RUN",
    "CUT&TAG",
    "spatial transcriptomics",
    "single cell sequencing",
    "10X Genomics",
    "single nucleus RNA",
    "Sanger sequencing",
    "genome assembly",
    # -----------------------------------------------------------------------
    # Neuroscience
    # -----------------------------------------------------------------------
    "neuron culture",
    "primary neuron culture",
    "brain slice",
    "patch clamp",
    "electrophysiology",
    "brain tissue dissection",
    "synaptic protein",
    "neural differentiation",
    "iPSC neuron",
    "calcium imaging neuron",
    # -----------------------------------------------------------------------
    # Immunology
    # -----------------------------------------------------------------------
    "antibody production",
    "antibody purification",
    "ELISA cytokine",
    "cytokine measurement",
    "T cell isolation",
    "B cell isolation",
    "PBMC isolation",
    "dendritic cell",
    "macrophage culture",
    "neutrophil isolation",
    "NK cell",
    "immune cell",
    "flow cytometry immune",
    "intracellular staining",
    # -----------------------------------------------------------------------
    # Biochemistry
    # -----------------------------------------------------------------------
    "enzyme activity assay",
    "kinase assay",
    "phosphorylation assay",
    "ubiquitination assay",
    "FRET assay",
    "surface plasmon resonance",
    "isothermal titration calorimetry",
    "electrophoretic mobility shift",
    "EMSA",
    "chromatin immunoprecipitation",
    "metabolite extraction",
    "lipid extraction",
    "metabolomics",
    "lipidomics",
    "GC-MS metabolomics",
    "LC-MS proteomics",
    # -----------------------------------------------------------------------
    # Stem cells and development
    # -----------------------------------------------------------------------
    "iPSC reprogramming",
    "iPSC differentiation",
    "embryoid body",
    "organoid culture",
    "brain organoid",
    "intestinal organoid",
    "stem cell culture",
    "hematopoietic stem cell",
    "mesenchymal stem cell",
    "cardiac differentiation",
    # -----------------------------------------------------------------------
    # Microbiology
    # -----------------------------------------------------------------------
    "bacterial culture",
    "bacterial growth",
    "biofilm formation",
    "minimal inhibitory concentration",
    "antibiotic susceptibility",
    "16S rRNA sequencing",
    "microbiome analysis",
    "phage transduction",
    "bacterial genetics",
    "gram staining",
    # -----------------------------------------------------------------------
    # Structural biology
    # -----------------------------------------------------------------------
    "protein crystallization",
    "X-ray crystallography",
    "NMR spectroscopy protein",
    "negative stain electron microscopy",
    "cryo-EM grid preparation",
    "protein structure",
    # -----------------------------------------------------------------------
    # Drug discovery and toxicology
    # -----------------------------------------------------------------------
    "IC50 assay",
    "cytotoxicity assay",
    "drug treatment",
    "drug screening",
    "high throughput screening",
    "cell-based assay",
    "genotoxicity assay",
    # -----------------------------------------------------------------------
    # Clinical and diagnostic
    # -----------------------------------------------------------------------
    "blood collection",
    "plasma isolation",
    "serum preparation",
    "tissue biopsy",
    "FFPE tissue",
    "immunohistochemistry tissue",
    "clinical sample",
    "biobank protocol",
    # -----------------------------------------------------------------------
    # Histology and tissue processing
    # -----------------------------------------------------------------------
    "tissue fixation",
    "paraffin embedding",
    "cryosectioning",
    "H&E staining",
    "tissue sectioning",
    "histology staining",
    "antigen retrieval",
    # -----------------------------------------------------------------------
    # Molecular & cellular methods (expansion — codex-suggested, 2026-07-25)
    # -----------------------------------------------------------------------
    # RNA biology and nucleic-acid processing
    "DNA cleanup",
    "DNA gel extraction",
    "PCR purification",
    "nucleic acid precipitation",
    "DNase treatment",
    "RNase treatment",
    "poly(A) selection",
    "ribosomal RNA depletion",
    "Northern blot",
    "Southern blot",
    "in situ hybridization",
    "fluorescence in situ hybridization",
    "DNA FISH",
    "RNA FISH",
    "single molecule RNA FISH",
    "RNAscope",
    "RACE PCR",
    "5 prime RACE",
    "3 prime RACE",
    "RNA immunoprecipitation",
    "RIP-seq",
    "CLIP-seq",
    "eCLIP",
    "RNA pull-down",
    "RNA stability assay",
    "actinomycin D chase",
    "polysome profiling",
    "ribosome profiling",
    "nascent RNA sequencing",
    "GRO-seq",
    "PRO-seq",
    "small RNA sequencing",
    "direct RNA sequencing",
    "microRNA assay",
    # PCR, genotyping and cytogenetics
    "multiplex PCR",
    "nested PCR",
    "inverse PCR",
    "touchdown PCR",
    "hot start PCR",
    "LAMP assay",
    "genotyping PCR",
    "allele-specific PCR",
    "methylation-specific PCR",
    "restriction fragment length polymorphism",
    "telomere length assay",
    "DNA fiber assay",
    "chromosome spread",
    "metaphase chromosome preparation",
    "karyotyping",
    # Genomics and epigenomics
    "whole exome sequencing",
    "targeted sequencing",
    "amplicon sequencing",
    "shotgun metagenomic sequencing",
    "long read library preparation",
    "exome library preparation",
    "target enrichment sequencing",
    "single cell ATAC sequencing",
    "single cell multiome",
    "CITE-seq",
    "cell hashing",
    "MNase-seq",
    "DNase-seq",
    "FAIRE-seq",
    "whole genome bisulfite sequencing",
    "reduced representation bisulfite sequencing",
    "MeDIP-seq",
    "ChIP-qPCR",
    "HiChIP",
    "4C-seq",
    "Capture-C",
    "chromosome conformation capture",
    # Gene perturbation and editing
    "RNA interference",
    "siRNA transfection",
    "shRNA knockdown",
    "antisense oligonucleotide",
    "gene knockdown",
    "gene overexpression",
    "inducible gene expression",
    "Cre-lox recombination",
    "CRISPR RNP delivery",
    "CRISPR editing validation",
    "CRISPR off-target analysis",
    "T7 endonuclease assay",
    "Cas13 RNA editing",
    "CRISPR epigenome editing",
    "transposon mutagenesis",
    "piggyBac transfection",
    "Sleeping Beauty transposon",
    # Cloning, delivery and viral vectors
    "plasmid cloning",
    "vector construction",
    "TOPO cloning",
    "TA cloning",
    "blunt-end cloning",
    "recombineering",
    "BAC cloning",
    "oligonucleotide annealing",
    "plasmid linearization",
    "nucleofection",
    "microinjection",
    "lentivirus production",
    "lentivirus titration",
    "AAV production",
    "AAV purification",
    "AAV titration",
    "adenovirus production",
    "viral vector production",
    "pseudovirus production",
    # Protein analysis and purification
    "native PAGE",
    "blue native PAGE",
    "isoelectric focusing",
    "dot blot",
    "slot blot",
    "far-western blot",
    "capillary western blot",
    "western blot stripping",
    "membrane protein extraction",
    "nuclear protein extraction",
    "cytoplasmic protein extraction",
    "protein precipitation",
    "TCA precipitation",
    "ammonium sulfate precipitation",
    "protein dialysis",
    "buffer exchange",
    "protein refolding",
    "ion exchange chromatography",
    "hydrophobic interaction chromatography",
    "density gradient centrifugation",
    "ultracentrifugation",
    "thermal shift assay",
    "differential scanning fluorimetry",
    "pulse-chase assay",
    "cycloheximide chase",
    "protein degradation assay",
    "protein turnover assay",
    "crosslinking mass spectrometry",
    # Proteomics and immunoassays
    "phosphoproteomics sample preparation",
    "TMT labeling",
    "SILAC labeling",
    "label-free proteomics",
    "in-gel digestion",
    "in-solution digestion",
    "trypsin digestion proteomics",
    "peptide desalting",
    "antibody conjugation",
    "antibody labeling",
    "antibody validation",
    "ELISPOT",
    "Luminex assay",
    "multiplex immunoassay",
    "proximity ligation assay",
    "enzyme kinetics",
    "phosphatase assay",
    "ATPase assay",
    "GTPase assay",
    "zymography",
    "glycosylation analysis",
    # Cell-culture preparation and quality control
    "aseptic technique",
    "cell culture media preparation",
    "mycoplasma testing",
    "cell line authentication",
    "STR profiling",
    "limiting dilution cloning",
    "single cell cloning",
    "clonal cell line generation",
    "primary cell isolation",
    "tissue dissociation",
    "enzymatic tissue dissociation",
    "single cell suspension",
    "red blood cell lysis",
    "dead cell removal",
    "magnetic cell separation",
    "MACS cell separation",
    "Ficoll density gradient",
    "co-culture",
    "transwell co-culture",
    "3D cell culture",
    "spheroid culture",
    "Matrigel culture",
    "air-liquid interface culture",
    "explant culture",
    "feeder cell culture",
    "serum starvation",
    "cell synchronization",
    "cell immortalization",
    "organoid passaging",
    "organoid dissociation",
    # Additional cell types
    "epithelial cell culture",
    "endothelial cell culture",
    "fibroblast culture",
    "cancer cell culture",
    "hepatocyte culture",
    "cardiomyocyte culture",
    "myoblast culture",
    "adipocyte differentiation",
    "insect cell culture",
    # Viability, proliferation and death
    "trypan blue exclusion",
    "resazurin assay",
    "ATP viability assay",
    "clonogenic assay",
    "colony formation assay",
    "EdU incorporation assay",
    "BrdU incorporation assay",
    "Ki-67 staining",
    "TUNEL assay",
    "Annexin V assay",
    "caspase activity assay",
    "LDH release assay",
    "live dead staining",
    # Stress, autophagy and mitochondrial function
    "mitochondrial membrane potential assay",
    "JC-1 assay",
    "reactive oxygen species assay",
    "oxidative stress assay",
    "autophagy assay",
    "LC3 turnover assay",
    "mitophagy assay",
    "senescence assay",
    "SA-beta-gal staining",
    "metabolic flux analysis",
    "Seahorse assay",
    "oxygen consumption rate",
    "extracellular acidification rate",
    "glucose uptake assay",
    "ATP measurement",
    "lipid droplet staining",
    # Migration and cellular function
    "wound healing assay",
    "scratch assay",
    "transwell migration assay",
    "cell invasion assay",
    "cell adhesion assay",
    "chemotaxis assay",
    "phagocytosis assay",
    "endocytosis assay",
    "exocytosis assay",
    "receptor internalization assay",
    "calcium flux assay",
    "cAMP assay",
    "dual luciferase reporter assay",
    "cell signaling assay",
    "cell mechanics assay",
    "traction force microscopy",
    # Subcellular fractionation and organelles
    "subcellular fractionation",
    "nuclei isolation",
    "nuclear cytoplasmic fractionation",
    "mitochondrial isolation",
    "mitochondrial purification",
    "lysosome isolation",
    "peroxisome isolation",
    "Golgi isolation",
    "endosome isolation",
    "plasma membrane isolation",
    "membrane fractionation",
    "microsome preparation",
    "ribosome isolation",
    "chromatin fractionation",
    "cytoskeleton fractionation",
    "secretome analysis",
    # Extracellular vesicles
    "extracellular vesicle isolation",
    "extracellular vesicle purification",
    "exosome isolation",
    "exosome characterization",
    "exosome uptake assay",
    # Imaging and image analysis
    "brightfield microscopy",
    "phase contrast microscopy",
    "differential interference contrast microscopy",
    "high-content imaging",
    "high-content screening",
    "time-lapse microscopy",
    "FRAP",
    "fluorescence recovery after photobleaching",
    "FLIM",
    "fluorescence lifetime imaging",
    "light sheet microscopy",
    "multiphoton microscopy",
    "expansion microscopy",
    "correlative light electron microscopy",
    "immunogold labeling",
    "cell segmentation",
    "fluorescence quantification",
    "colocalization analysis",
    # Flow and mass cytometry
    "flow cytometry staining",
    "flow cytometry panel design",
    "flow cytometry compensation",
    "spectral flow cytometry",
    "mass cytometry",
    "CyTOF",
    "phospho-flow cytometry",
    # Tissue and spatial methods
    "tissue clearing",
    "whole mount staining",
    "whole mount immunofluorescence",
    "perfusion fixation",
    "tissue decalcification",
    "cytospin preparation",
    "tissue microarray",
    "laser capture microdissection",
    "multiplex immunofluorescence",
    # Cellular immunology
    "T cell activation",
    "T cell proliferation",
    "B cell activation",
    "macrophage polarization",
    "dendritic cell differentiation",
    "NK cell cytotoxicity",
    "mixed lymphocyte reaction",
    "cytokine bead array",
    "tetramer staining",
    "neutrophil extracellular trap assay",
    "complement assay",
    "antibody-dependent cellular cytotoxicity",
    # Virology and microbial cellular methods
    "viral infection assay",
    "virus titration",
    "viral plaque assay",
    "TCID50 assay",
    "virus neutralization assay",
    "pseudovirus neutralization assay",
    "viral RNA extraction",
    "bacterial conjugation",
    "bacterial electroporation",
    "phage plaque assay",
    "fungal culture",
    "anaerobic bacterial culture",
    # -----------------------------------------------------------------------
    # Molecular & cellular methods, batch 2 (expansion — codex-suggested, 2026-07-25)
    # -----------------------------------------------------------------------
    # DNA damage and repair
    "DNA damage assay",
    "comet assay",
    "gamma H2AX staining",
    "DNA repair assay",
    "homologous recombination assay",
    "non-homologous end joining assay",
    "replication fork assay",
    "micronucleus assay",
    # RNA processing and epitranscriptomics
    "RNA splicing assay",
    "alternative splicing",
    "minigene splicing assay",
    "RNA editing",
    "RNA structure probing",
    "m6A sequencing",
    "nonsense-mediated decay assay",
    # Cytoskeleton and cell organization
    "actin staining",
    "phalloidin staining",
    "microtubule staining",
    "tubulin polymerization assay",
    "intermediate filament staining",
    "centrosome analysis",
    "cell polarity assay",
    "cytokinesis assay",
    # Cell junctions and barriers
    "tight junction staining",
    "adherens junction staining",
    "epithelial barrier assay",
    "permeability assay",
    "blood-brain barrier model",
    # Protein interactions and localization
    "proximity labeling",
    "BioID",
    "TurboID",
    "APEX labeling",
    "proximity proteomics",
    "bimolecular fluorescence complementation",
    "BiFC",
    # Protein modification and quality control
    "protein acetylation",
    "SUMOylation assay",
    "palmitoylation assay",
    "chaperone assay",
    "unfolded protein response",
    "ER stress assay",
    # Additional cell-death pathways
    "ferroptosis assay",
    "necroptosis assay",
    "anoikis assay",
    # Extracellular matrix and mechanobiology
    "extracellular matrix extraction",
    "collagen gel culture",
    "hydrogel cell culture",
    "decellularization",
    "matrix stiffness",
    "atomic force microscopy",
    "micropipette aspiration",
    # Metabolic tracing
    "stable isotope tracing",
    "metabolic labeling",
    "glucose flux assay",
    "glycolysis assay",
    "fatty acid oxidation assay",
    "amino acid tracing",
    "carbon-13 tracing",
    # Development and lineage
    "lineage tracing",
    "cell barcoding",
    "fate mapping",
    "embryo culture",
    "embryonic microinjection",
    "gastruloid culture",
    # Biophysical protein characterization
    "circular dichroism spectroscopy",
    "dynamic light scattering",
    "SEC-MALS",
    "biolayer interferometry",
    "microscale thermophoresis",
    "fluorescence anisotropy",
    # Synthetic biology
    "synthetic gene circuit",
    "optogenetics",
    "inducible promoter",
    "biosensor assay",
    "synthetic promoter",
    "cell-free expression",
]


def _get_token_from_client_credentials(client_id: str, client_secret: str) -> str:
    """
    Exchange client_id + client_secret for a CLIENT_ACCESS_TOKEN.
    This only needs to be done once — the token is then saved to variables.env.
    """
    url = "https://www.protocols.io/api/v3/oauth/token"
    payload = urllib.parse.urlencode({
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials",
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": BROWSER_UA,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            token = body.get("access_token") or body.get("client_access_token") or ""
            if not token:
                raise SystemExit(f"[ERROR] Token exchange succeeded but no token in response: {body}")
            log.info("Token obtained successfully.")
            return token
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise SystemExit(f"[ERROR] Token exchange failed (HTTP {e.code}): {body}")


def _save_token_to_env(token: str):
    """Write the obtained token back into variables.env so future runs skip this step."""
    env_path = Path(__file__).parent.parent / "protocolnerd-backend" / "variables.env"
    text = env_path.read_text(encoding="utf-8")
    if "PROTOCOLS_IO_TOKEN=" in text:
        lines = []
        for line in text.splitlines():
            if line.startswith("PROTOCOLS_IO_TOKEN="):
                lines.append(f"PROTOCOLS_IO_TOKEN={token}")
            else:
                lines.append(line)
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        env_path.write_text(text.rstrip() + f"\nPROTOCOLS_IO_TOKEN={token}\n", encoding="utf-8")
    log.info(f"Token saved to {env_path}")


def _get_token() -> str:
    token = os.getenv("PROTOCOLS_IO_TOKEN", "").strip().strip('"')
    if token:
        return token

    # Fall back: check for client_id / client_secret
    client_id = os.getenv("PROTOCOLS_IO_CLIENT_ID", "").strip().strip('"')
    client_secret = os.getenv("PROTOCOLS_IO_CLIENT_SECRET", "").strip().strip('"')

    if client_id and client_secret:
        log.info("No token found — exchanging client_id/client_secret for access token...")
        token = _get_token_from_client_credentials(client_id, client_secret)
        _save_token_to_env(token)
        return token

    raise SystemExit(
        "\n[ERROR] No credentials found. Do one of the following:\n\n"
        "  OPTION A — paste your client_id and client_secret (from protocols.io/developers):\n"
        "    Add to protocolnerd-backend/variables.env:\n"
        "      PROTOCOLS_IO_CLIENT_ID=your_client_id\n"
        "      PROTOCOLS_IO_CLIENT_SECRET=your_client_secret\n"
        "    Then re-run. The script will fetch and save the token automatically.\n\n"
        "  OPTION B — paste the token directly if you already have it:\n"
        "    Add to protocolnerd-backend/variables.env:\n"
        "      PROTOCOLS_IO_TOKEN=your_token_here\n"
    )


BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _api_get(path: str, params: Dict[str, Any], token: str, retries: int = 5) -> Dict[str, Any]:
    """
    Make an authenticated GET request to the protocols.io API, retrying on
    transient failures (timeouts, connection resets, 5xx). A single network
    hiccup must NOT be mistaken for "no more results" during a long crawl.

    Returns the parsed JSON, or {} only after all retries are exhausted (or on
    a genuine 4xx, which won't be retried).
    """
    url = f"{API_BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": BROWSER_UA,
        },
    )
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            # 4xx (except 429 rate-limit) are genuine — don't retry.
            if 400 <= e.code < 500 and e.code != 429:
                log.warning(f"HTTP {e.code} for {url}: {body[:200]}")
                return {}
            log.warning(f"HTTP {e.code} (attempt {attempt}/{retries}) for {url}: {body[:120]}")
        except Exception as e:
            log.warning(f"Request failed (attempt {attempt}/{retries}) for {url}: {e}")
        if attempt < retries:
            time.sleep(min(2 ** attempt, 30))  # exponential backoff, capped at 30s
    return {}


def _fetch_protocols_page(
    keyword: str,
    page_id: int,
    page_size: int,
    token: str,
    order_field: str = ORDER_FIELD,
    order_dir: str = ORDER_DIR,
) -> tuple[List[Dict], bool, bool]:
    """
    Fetch one page of protocols for a keyword.
    Returns (list_of_protocols, has_more_pages, ok).

    `ok` is False when the request itself failed (e.g. all retries timed out),
    which is distinct from a genuine empty page — so the caller can retry the
    same page instead of assuming it reached the end of the results.
    """
    params = {
        "filter": "public",
        "key": keyword,
        "order_field": order_field,
        "order_dir": order_dir,
        "page_id": page_id,
        "page_size": page_size,
    }
    data = _api_get("/protocols", params, token)

    if not data or data.get("status_code") not in (0, None):
        log.warning(f"API error for keyword='{keyword}' page={page_id}: {data.get('status_message', 'request failed')}")
        return [], False, False

    items = data.get("items", []) or []
    pagination = data.get("pagination", {}) or {}
    total_pages = pagination.get("total_pages", 1)
    has_more = page_id < total_pages

    return items, has_more, True


def _strip_html(text: str) -> str:
    """Remove HTML tags from a string."""
    import re
    if not text:
        return ""
    return re.sub(r"<[^>]+>", " ", str(text)).strip()


def _fetch_full_protocol(protocol_id: int, token: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a single protocol by ID to get its full step data.
    The list endpoint omits steps; this individual endpoint includes them.
    """
    data = _api_get(f"/protocols/{protocol_id}", {}, token)
    # The individual-protocol response nests the protocol under the top-level
    # 'protocol' key (NOT 'payload').
    return data.get("protocol") or (data if data.get("id") else None)


def _extract_steps(raw: Dict[str, Any]) -> List[str]:
    """
    Extract readable plain-text steps from a full protocol object.

    Prefers the step-level `step` field (the instruction HTML). Falls back to
    description components, skipping type_id=6 section headers.
    """
    steps = []
    for s in (raw.get("steps") or []):
        text = _strip_html(s.get("step") or "")
        if not text:
            parts = []
            for comp in (s.get("components") or []):
                if comp.get("type_id") == 6:  # section header, not an instruction
                    continue
                source = comp.get("source") or {}
                t = _strip_html(
                    source.get("description")
                    or source.get("body")
                    or source.get("title")
                    or ""
                )
                if t and t.lower() not in {"note", "warning", "tip"}:
                    parts.append(t)
            text = " ".join(parts).strip()
        if text:
            steps.append(text)
    return steps


def _extract_protocol_fields(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Pull out the fields most useful for RAG indexing."""

    steps = _extract_steps(raw)

    # Authors
    authors_raw = raw.get("authors") or []
    authors = [
        a.get("name") or f"{a.get('fname', '')} {a.get('lname', '')}".strip()
        for a in authors_raw
        if isinstance(a, dict)
    ]

    creator = raw.get("creator") or {}
    if isinstance(creator, dict):
        creator_name = creator.get("name") or f"{creator.get('fname', '')} {creator.get('lname', '')}".strip()
    else:
        creator_name = ""

    return {
        "id": raw.get("id"),
        "title": raw.get("title") or "",
        "uri": raw.get("uri") or "",
        "doi": raw.get("doi") or "",
        "description": raw.get("description") or "",
        "guidelines": raw.get("guidelines") or "",
        "before_start": raw.get("before_start") or "",
        "warning": raw.get("warning") or "",
        "materials_text": raw.get("materials_text") or "",
        "steps": steps,
        "authors": [a for a in authors if a],
        "creator": creator_name,
        "published_on": raw.get("published_on") or "",
        "created_on": raw.get("created_on") or "",
        "keywords": [
            kw.get("name") or kw if isinstance(kw, str) else ""
            for kw in (raw.get("keywords") or [])
        ],
    }


def fetch_and_cache(
    keywords: List[str],
    output_dir: Path,
    max_per_keyword: int,
    token: str,
    skip_steps: bool = False,
    incremental: bool = False,
) -> List[Dict[str, Any]]:
    """
    For each keyword, paginate through protocols.io and save each protocol to disk.
    Deduplicates across keywords by protocol ID.
    Returns the full list of saved protocol metadata (for the index).

    incremental=True: order by id DESC and stop each keyword the moment an id drops
    to/below the highest id already cached (the "watermark"). Because ids are
    monotonic with creation, this reliably fetches ONLY protocols created since the
    last run — no misses, and it reads just the top page(s) per keyword.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_ids: Set[int] = set()
    index_entries: List[Dict[str, Any]] = []

    # Resume: load already-fetched IDs from existing index
    index_path = output_dir.parent / "protocols_index.json"
    if index_path.exists():
        try:
            existing = json.loads(index_path.read_text(encoding="utf-8"))
            for entry in existing:
                seen_ids.add(entry["id"])
                index_entries.append(entry)
            log.info(f"Resuming — {len(seen_ids)} protocols already cached.")
        except Exception:
            pass

    # Ordering: default is activity (used for the initial crawl). Incremental mode
    # switches to a stable id DESC order with a global max-id watermark.
    order_field, order_dir = ORDER_FIELD, ORDER_DIR
    watermark = 0
    if incremental:
        order_field, order_dir = "id", "desc"
        watermark = max(seen_ids) if seen_ids else 0
        log.info(f"Incremental mode: order by id DESC, watermark (max cached id) = {watermark}")

    # protocols.io accepts up to page_size=200, but big pages are large downloads
    # that time out on flaky networks. Smaller pages are far more reliable; the
    # caller can tune this via PAGE_SIZE.
    page_size = min(PAGE_SIZE, max_per_keyword)
    total_new = 0

    for keyword in keywords:
        log.info(f"--- Fetching keyword: '{keyword}' (max {max_per_keyword}) ---")
        fetched_this_kw = 0
        page_id = 0  # protocols.io pagination is 0-indexed; page_id=0 is the
                     # first (most relevant) page. Starting at 1 skips it.
        page_failures = 0  # consecutive failures on the CURRENT page
        reached_watermark = False  # incremental: hit an id we already have

        while fetched_this_kw < max_per_keyword and not reached_watermark:
            items, has_more, ok = _fetch_protocols_page(
                keyword, page_id, page_size, token, order_field, order_dir
            )
            time.sleep(RATE_LIMIT_DELAY)

            if not ok:
                # Transient failure (not a real end-of-results). Retry the same
                # page a few times before giving up on this keyword, so one
                # flaky request doesn't truncate the whole enumeration.
                page_failures += 1
                if page_failures >= 6:
                    log.warning(f"  giving up on '{keyword}' at page {page_id} after {page_failures} failures")
                    break
                time.sleep(min(2 ** page_failures, 30))
                continue
            page_failures = 0

            if not items:
                break

            for raw in items:
                pid = raw.get("id")
                if not pid:
                    continue
                # Incremental: ids arrive DESC, so the first id at/below the
                # watermark means every remaining protocol is already cached/older.
                if incremental and pid <= watermark:
                    reached_watermark = True
                    break
                if pid in seen_ids:
                    continue
                if fetched_this_kw >= max_per_keyword:
                    break

                # Optionally fetch full protocol to get steps (doubles API calls)
                if not skip_steps:
                    full = _fetch_full_protocol(pid, token)
                    time.sleep(RATE_LIMIT_DELAY)
                    protocol = _extract_protocol_fields(full if full else raw)
                else:
                    protocol = _extract_protocol_fields(raw)

                # Save individual protocol file
                dest = output_dir / f"{pid}.json"
                dest.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")

                seen_ids.add(pid)
                fetched_this_kw += 1
                total_new += 1

                # Lightweight index entry (no full steps — kept in individual files)
                index_entries.append({
                    "id": pid,
                    "title": protocol["title"],
                    "uri": protocol["uri"],
                    "doi": protocol["doi"],
                    "description": protocol["description"][:300],
                    "keywords": protocol["keywords"],
                    "authors": protocol["authors"],
                    "published_on": protocol["published_on"],
                    "file": str(dest.relative_to(output_dir.parent)),
                })

                log.info(f"  [{total_new:>4}] {pid}: {protocol['title'][:70]}")

            if not has_more:
                break
            page_id += 1

        log.info(f"  Fetched {fetched_this_kw} new protocols for '{keyword}'")

    # Write / overwrite the index
    index_path.write_text(json.dumps(index_entries, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"\nDone. {total_new} new protocols cached. Total in index: {len(index_entries)}")
    log.info(f"Index: {index_path}")
    log.info(f"Protocols dir: {output_dir}")
    return index_entries


def main():
    global PAGE_SIZE, ORDER_FIELD, ORDER_DIR
    parser = argparse.ArgumentParser(description="Fetch protocols from protocols.io into local cache.")
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=DEFAULT_KEYWORDS,
        help="Search keywords to use. Defaults to a broad biology set.",
    )
    parser.add_argument(
        "--max-per-keyword",
        type=int,
        default=50,
        help="Max protocols to fetch per keyword (default: 50).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/protocols"),
        help="Directory to write protocol JSON files (default: data/protocols).",
    )
    parser.add_argument(
        "--skip-steps",
        action="store_true",
        help="Skip the per-protocol full fetch (halves API calls / time). "
             "The list endpoint omits step text anyway, so this matches the "
             "existing cached corpus while covering ~2x more protocols.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=PAGE_SIZE,
        help=f"Protocols per list request, max 200 (default {PAGE_SIZE}). "
             "Smaller is more reliable on flaky networks.",
    )
    parser.add_argument(
        "--order-field",
        default=ORDER_FIELD,
        help="Sort field. Use 'id' for a stable, drift-free full enumeration.",
    )
    parser.add_argument(
        "--order-dir",
        default=ORDER_DIR,
        choices=["asc", "desc"],
        help="Sort direction (default desc; use asc with --order-field id).",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Delta run: order by id DESC and stop each keyword at the highest "
             "already-cached id, so only protocols created since the last run are "
             "fetched (reliable + fast).",
    )
    args = parser.parse_args()

    PAGE_SIZE = max(1, min(200, args.page_size))
    ORDER_FIELD = args.order_field
    ORDER_DIR = args.order_dir

    token = _get_token()

    log.info(f"Output dir: {args.output_dir.resolve()}")
    log.info(f"Keywords ({len(args.keywords)}): {args.keywords[:5]}{'...' if len(args.keywords) > 5 else ''}")
    log.info(f"Max per keyword: {args.max_per_keyword}")
    log.info(f"Estimated max protocols: {len(args.keywords) * args.max_per_keyword} (before deduplication)")

    fetch_and_cache(
        keywords=args.keywords,
        output_dir=args.output_dir,
        max_per_keyword=args.max_per_keyword,
        token=token,
        skip_steps=args.skip_steps,
        incremental=args.incremental,
    )


if __name__ == "__main__":
    main()
