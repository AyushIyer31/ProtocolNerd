from __future__ import annotations

import re
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

from biology_intents import (
    INTENT_FAMILY_LABELS,
    SUB_INTENT_LABELS,
    controlled_intent_payload,
    family_for_sub_intent,
    normalize_intent_family,
    normalize_sub_intent,
)
from query_operators import detect_operator
import ranking_config as rank_cfg
import profile_schema


PROFILE_FIELDS = [
    "intent_family",
    "sub_intent",
    "organism",
    "sample_type",
    "tissue_or_cell_type",
    "target",
    "gene_or_construct",
    "modification_type",
    "method",
    "experimental_method",
    "delivery_method",
    "expression_type",
    "readout",
    "readout_assay",
    "condition",
    "timeline",
    "equipment",
    "required_equipment",
    "difficulty",
    "protocol_difficulty",
    "constraints",
    "intent_specific",
]


DEFAULT_PROFILE: Dict[str, Any] = {
    "intent_family": None,
    "sub_intent": None,
    "organism": None,
    "sample_type": None,
    "tissue_or_cell_type": None,
    "target": None,
    "gene_or_construct": None,
    "modification_type": None,
    "method": None,
    "experimental_method": None,
    "delivery_method": None,
    "expression_type": None,
    "readout": None,
    "readout_assay": None,
    "condition": None,
    "timeline": None,
    "equipment": [],
    "required_equipment": [],
    "difficulty": None,
    "protocol_difficulty": None,
    "constraints": [],
    "intent_specific": {},
}


INTENT_LABELS = {
    **SUB_INTENT_LABELS,
    "gene_modification": "gene modification",
    "gene_overexpression": "gene overexpression",
    "gene_knockdown": "gene knockdown",
    "protein_purification": "protein purification",
    "pcr_qpcr": "PCR/qPCR",
    "transformation": "transformation",
    "microscopy": "microscopy",
    "sequencing_prep": "sequencing prep",
    "unknown": "unknown",
}


_INTENT_PATTERNS = [
    ("multiplex_gene_modification", r"\b(?:more than one|multiple|multiplex|several|at the same time|simultaneously)\b.{0,80}\b(?:genes?|transcription factors?|targets?)\b.{0,80}\b(?:modify|modified|modification|edit|editing|alter|altered)\b|\b(?:modify|modified|modification|edit|editing|alter|altered)\b.{0,80}\b(?:more than one|multiple|multiplex|several|at the same time|simultaneously)\b"),
    ("stress_tolerance_assay", r"\b(drought|salt|salinity|heat|cold|osmotic|stress)\b.{0,60}\b(tolerance|resistance|response|assay|test|testing|phenotype|phenotyping)\b|\b(tolerance|resistance|response)\b.{0,60}\b(drought|salt|salinity|heat|cold|osmotic|stress)\b"),
    ("gene_modification", r"\b(gene modification|modify genes?|modified genes?|genes? (?:are |is |being )?modified|gene editing|genome editing|crispr|cas9|mutagenesis|mutagenize|mutation)\b"),
    ("gene_overexpression", r"\b(overexpress|overexpressing|overexpressed|overexpression|ectopic expression|forced expression)\b"),
    ("gene_knockdown", r"\b(knockdown|knock down|silenc(?:e|ing)|sirna|shrna|rnai)\b"),
    ("protein_purification", r"\b(protein purification|purify protein|recombinant protein|affinity purification|his-tag|his tag|protein expression)\b"),
    ("protein_extraction", r"\b(protein extraction|extract protein|total protein extraction)\b"),
    ("western_blot", r"\b(western blot|western blotting|immunoblot)\b"),
    ("pcr_qpcr", r"\b(qpcr|q-pcr|rt-qpcr|rt pcr|real[- ]time pcr|pcr|polymerase chain reaction)\b"),
    ("transformation", r"\b(transformation|transform|agrobacterium|floral dip|transfection|transduction|electroporation|biolistic|gene gun)\b"),
    ("microscopy", r"\b(microscopy|microscope|imaging|confocal|fluorescence imaging|brightfield|electron microscopy|localization)\b"),
    ("sequencing_prep", r"\b(sequencing|rna-seq|rnaseq|chip-seq|amplicon|library prep|ngs|sanger)\b"),
]

_OVEREXPRESSION_GOAL_RE = re.compile(
    r"\b(overexpress|overexpressing|overexpressed|overexpression|transgene expression|"
    r"transgenic overexpression|ectopic expression|forced expression)\b",
    flags=re.IGNORECASE,
)

_ORGANISMS = {
    "arabidopsis": "Arabidopsis thaliana",
    "arabidopsis thaliana": "Arabidopsis thaliana",
    "tobacco": "tobacco",
    "nicotiana benthamiana": "Nicotiana benthamiana",
    "nicotiana": "Nicotiana",
    "rice": "rice",
    "oryza sativa": "Oryza sativa",
    "maize": "maize",
    "corn": "maize",
    "wheat": "wheat",
    "tomato": "tomato",
    "soybean": "soybean",
    "bean": "bean",
    "beans": "bean",
    "common bean": "bean",
    "phaseolus vulgaris": "bean",
    "pumpkin": "pumpkin",
    "pumpkins": "pumpkin",
    "cucurbita pepo": "pumpkin",
    "cucurbita maxima": "pumpkin",
    "squash": "squash",
    "cucumber": "cucumber",
    "cucumis sativus": "cucumber",
    "potato": "potato",
    "solanum tuberosum": "potato",
    "pea": "pea",
    "peas": "pea",
    "pisum sativum": "pea",
    "plant": "plant",
    "plants": "plant",
    "human": "human",
    "mouse": "mouse",
    "mice": "mouse",
    "rat": "rat",
    "yeast": "yeast",
    "e. coli": "E. coli",
    "e coli": "E. coli",
    "ecoli": "E. coli",
    "bacteria": "bacteria",
    "bacterial": "bacteria",
    "zebrafish": "zebrafish",
    "drosophila": "Drosophila",
}

_GENERIC_ORGANISMS = {
    "plant", "plants", "cell", "cells", "tissue", "organism", "bacteria",
    "not sure", "not sure / flexible", "unsure", "flexible",
}

_TISSUE_TERMS = [
    ("in planta / whole plant", r"\b(whole plant|whole-plant|entire plant|in planta|in-planta)\b"),
    ("whole animal / in vivo", r"\b(whole animal|whole-animal|in vivo|in-vivo)\b"),
    ("cotyledons / explants", r"\b(cotyledon|cotyledons|explant|explants)\b"),
    ("leaf", r"\b(leaf|leaves)\b"),
    ("root", r"\b(root|roots)\b"),
    ("seed", r"\b(seed|seeds)\b"),
    ("flower", r"\b(flower|flowers|floral)\b"),
    ("immature embryos", r"\b(immature embryo|immature embryos)\b"),
    ("embryo", r"\b(embryo|embryos)\b"),
    ("primary cells", r"\b(primary cell|primary cells)\b"),
    ("cell line", r"\b(cell line|cell lines)\b"),
    ("specific tissue or organ", r"\b(specific tissue|organ|organs|tissue or organ)\b"),
    ("organoid", r"\b(organoid|organoids)\b"),
    ("bacterial culture", r"\b(bacterial culture|bacterial cultures)\b"),
    ("plasmid-bearing cells", r"\b(plasmid-bearing cells|plasmid bearing cells)\b"),
    ("yeast culture", r"\b(yeast culture|yeast cultures)\b"),
    ("colony", r"\b(colony|colonies)\b"),
    ("protoplast", r"\b(protoplast|protoplasts)\b"),
    ("callus", r"\b(callus)\b"),
    ("cell culture", r"\b(cell culture|cultured cells|cells)\b"),
    ("tissue", r"\b(tissue|tissues)\b"),
    ("not sure / flexible", r"\b(not sure|unsure|no preference|flexible)\b"),
]

_EXPRESSION_TYPES = [
    ("stable or transient expression", r"\b(not sure|either|both|both stable and transient)\b"),
    ("stable transformation", r"\b(stable|stably|transgenic|floral dip)\b"),
    ("transient expression", r"\b(transient|transiently|agroinfiltration|agro-infiltration|protoplast)\b"),
    ("tissue-specific expression", r"\b(tissue specific|tissue-specific|cell specific|cell-specific)\b"),
    ("inducible expression", r"\b(inducible|induced|estradiol|dexamethasone)\b"),
    ("constitutive expression", r"\b(constitutive|35s promoter|camv 35s|ubiquitin promoter)\b"),
]

_MODIFICATION_TYPES = [
    ("CRISPR / genome editing", r"\b(crispr|cas9|cas12|genome editing|gene editing|base editing|prime editing)\b"),
    ("overexpression", r"\b(overexpress|overexpressing|overexpressed|overexpression|transgene expression|ectopic expression|forced expression)\b"),
    ("knockdown / silencing", r"\b(knockdown|knock down|silencing|silence|rnai|sirna|shrna|vigs)\b"),
    ("mutation / mutagenesis", r"\b(mutation|mutations|mutagenesis|mutagenize|mutant|knockout|knock out)\b"),
    ("stable transformation", r"\b(stable transformation|stable transform|stably transformed|transgenic)\b"),
]

_DELIVERY_METHODS = [
    ("stable transformation", r"\b(stable transformation|stable transform|stably transformed|transgenic|floral dip)\b"),
    ("transient expression", r"\b(transient expression|transient transformation|transiently|agroinfiltration|agro-infiltration)\b"),
    ("Agrobacterium-mediated delivery", r"\b(agrobacterium|agrobacterium-mediated|agrobacterium mediated)\b"),
    ("biolistic / gene gun", r"\b(biolistic|particle bombardment|gene gun)\b"),
    ("protoplast transfection", r"\b(protoplast transfection|peg transfection|transfection)\b"),
]

_READOUTS = [
    ("phenotype", r"\b(phenotype|phenotypic|growth|morphology|trait|traits)\b"),
    ("RNA level / qPCR", r"\b(qpcr|q-pcr|rt-qpcr|rna level|transcript|mrna|gene expression)\b"),
    ("protein level", r"\b(protein level|western|western blot|immunoblot|elisa)\b"),
    ("stress response", r"\b(stress|drought|salt|salinity|heat|cold|tolerance|resistance)\b"),
    ("localization", r"\b(localization|localisation|gfp|fluorescence|confocal|microscopy|imaging)\b"),
    ("genotyping / sequencing", r"\b(genotyping|sequencing|sanger|amplicon|mutation confirmation)\b"),
    ("sequencing", r"\b(sequencing|rna-seq|rnaseq|ngs)\b"),
    ("reporter assay", r"\b(reporter|luciferase|gus|beta glucuronidase)\b"),
]

_EQUIPMENT = [
    ("qPCR machine", r"\b(qpcr|q-pcr|real[- ]time pcr)\b"),
    ("PCR thermocycler", r"\b(pcr|thermocycler)\b"),
    ("confocal microscope", r"\b(confocal)\b"),
    ("fluorescence microscope", r"\b(fluorescence microscope|fluorescent microscope|fluorescence imaging|gfp)\b"),
    ("growth chamber", r"\b(growth chamber|plant growth room|greenhouse)\b"),
    ("electroporator", r"\b(electroporator|electroporation)\b"),
    ("gene gun", r"\b(gene gun|biolistic)\b"),
    ("sequencer", r"\b(sequencer|sequencing|ngs)\b"),
]

_ORGANISM_SEARCH_VARIANTS = {
    "Arabidopsis thaliana": ["Arabidopsis thaliana", "arabidopsis"],
    "Nicotiana benthamiana": ["Nicotiana benthamiana", "tobacco"],
    "rice": ["rice", "Oryza sativa"],
    "maize": ["maize", "Zea mays", "corn"],
    "wheat": ["wheat", "Triticum aestivum"],
    "tomato": ["tomato", "Solanum lycopersicum"],
    "bean": ["bean", "beans", "common bean", "Phaseolus vulgaris"],
    "pumpkin": ["pumpkin", "Cucurbita pepo", "Cucurbita maxima"],
    "squash": ["squash", "Cucurbita"],
    "cucumber": ["cucumber", "Cucumis sativus"],
    "potato": ["potato", "Solanum tuberosum"],
    "pea": ["pea", "Pisum sativum"],
    "mouse": ["mouse", "Mus musculus"],
    "human": ["human", "Homo sapiens"],
    "E. coli": ["E. coli", "Escherichia coli"],
}

_METHOD_SEARCH_VARIANTS = {
    "gene modification": ["gene modification", "gene editing", "genome modification"],
    "multiplex gene modification": ["multiplex gene modification", "simultaneous gene modification"],
    "genome editing": ["CRISPR", "genome editing", "gene editing"],
    "gene overexpression": ["gene overexpression", "overexpression", "transgene expression"],
    "gene knockdown": ["gene knockdown", "RNAi", "gene silencing"],
    "protein purification": ["protein purification", "recombinant protein purification"],
    "protein extraction": ["protein extraction", "total protein extraction"],
    "western blot": ["western blot", "immunoblot"],
    "protein detection": ["protein detection", "western blot", "immunoblot"],
    "PCR/qPCR": ["PCR", "qPCR", "RT-qPCR"],
    "transformation": ["transformation", "genetic transformation"],
    "microscopy": ["microscopy", "imaging"],
    "sequencing prep": ["sequencing prep", "library preparation"],
    "stress tolerance assay": ["stress tolerance assay", "phenotyping", "stress assay"],
}

_MODIFICATION_SEARCH_VARIANTS = {
    "CRISPR / genome editing": ["CRISPR", "genome editing", "gene editing"],
    "overexpression": ["overexpression", "gene overexpression", "transgene expression"],
    "knockdown / silencing": ["gene knockdown", "RNAi", "gene silencing"],
    "mutation / mutagenesis": ["mutation", "mutagenesis", "knockout"],
    "stable transformation": ["stable transformation", "transgenic transformation"],
}

_DELIVERY_SEARCH_VARIANTS = {
    "stable transformation": ["stable transformation", "transgenic"],
    "transient expression": ["transient expression", "transient transformation"],
    "Agrobacterium-mediated delivery": ["Agrobacterium", "Agrobacterium-mediated"],
    "biolistic / gene gun": ["biolistic", "gene gun", "particle bombardment"],
    "protoplast transfection": ["protoplast transfection", "PEG transfection"],
}

_EXPRESSION_SEARCH_VARIANTS = {
    "stable transformation": ["stable transformation", "transgenic"],
    "transient expression": ["transient expression", "transient overexpression", "transient transformation", "agroinfiltration", "agro-infiltration"],
    "stable or transient expression": ["overexpression", "transient expression", "stable transformation"],
    "tissue-specific expression": ["tissue specific expression"],
    "inducible expression": ["inducible expression"],
    "constitutive expression": ["constitutive expression"],
}

_READOUT_SEARCH_VARIANTS = {
    "phenotype": ["phenotype", "phenotyping", "trait analysis"],
    "drought tolerance / phenotype": ["drought tolerance", "phenotyping", "drought stress assay"],
    "salt tolerance / phenotype": ["salt tolerance", "phenotyping", "salt stress assay"],
    "stress tolerance / phenotype": ["stress tolerance", "phenotyping", "stress assay"],
    "RNA level / qPCR": ["qPCR", "RT-qPCR", "RNA level", "gene expression"],
    "protein level": ["protein level", "protein detection", "western blot", "protein assay"],
    "stress response": ["stress response", "stress tolerance"],
    "localization": ["localization", "GFP", "fluorescence microscopy"],
    "genotyping / sequencing": ["genotyping", "Sanger sequencing", "amplicon sequencing"],
    "sequencing": ["sequencing", "RNA-seq"],
    "reporter assay": ["reporter assay", "luciferase", "GUS"],
}

_PLANT_CONTEXT_TERMS = [
    "plant", "plants", "leaf", "leaves", "seedling", "mesophyll",
    "protoplast", "agrobacterium", "agroinfiltration", "agro-infiltration",
    "nicotiana", "arabidopsis", "rice", "oryza", "maize", "zea mays",
    "bean", "beans", "phaseolus", "pumpkin", "cucurbita", "squash",
    "immature embryo", "callus", "regeneration", "somatic embryogenesis",
    "tobacco leaf", "tobacco leaves",
]

_TEV_FALSE_POSITIVE_TERMS = [
    "tev protease",
    "tobacco etch virus",
    "affinity tag",
    "affinity tags",
    "cleavage of the fusion protein",
    "removal of affinity tags",
]

_GENERIC_PROTEIN_PURIFICATION_TERMS = [
    "protein expression and purification",
    "protein purification",
    "recombinant protein expression",
    "his-tagged protein expression",
    "his6-tagged",
    "e. coli",
    "bacterial expression",
]

_RNA_QPCR_OFF_TARGET_TERMS = [
    "protein expression and purification",
    "protein purification",
    "recombinant protein expression",
    "recombinant protein purification",
    "ribosome footprint",
    "ribosome footprinting",
    "ribo-seq",
    "riboseq",
    "metabolomics",
    "metabolomic",
    "mass spectrometry",
    "lc-ms",
    "library prep",
    "library preparation",
    "sequencing library",
    "rna-seq",
    "rnaseq",
    "ngs",
]

_STABLE_TRANSFORMATION_DELIVERY_TERMS = [
    "stable transformation",
    "stably transformed",
    "plant transformation",
    "genetic transformation",
    "agrobacterium-mediated transformation",
    "agrobacterium mediated transformation",
    "agrobacterium transformation",
    "biolistic transformation",
    "biolistics-mediated transformation",
    "particle bombardment",
    "gene gun",
    "immature embryo transformation",
    "somatic embryogenesis transformation",
    "transgenic plant",
    "transgenic plants",
    "transgenic line",
    "transgenic lines",
    "transgenic overexpression",
    "transformant selection",
]

_STABLE_TRANSFORMATION_OFF_TARGET_TERMS = [
    "hydroponics",
    "rna-seq",
    "rnaseq",
    "bsa-seq",
    "sam-seq",
    "sequencing",
    "atlas selection",
    "genotyping assay",
    "phenotyping",
]


def empty_profile() -> Dict[str, Any]:
    return deepcopy(DEFAULT_PROFILE)


def normalize_profile(candidate: Any, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Validate an LLM-produced profile and fill any missing fields from fallback.
    """
    profile = empty_profile()
    sources = [fallback, candidate]
    for source in sources:
        if not isinstance(source, dict):
            continue
        for field in PROFILE_FIELDS:
            value = source.get(field)
            if _emptyish(value):
                continue
            if field in {"required_equipment", "equipment", "constraints"}:
                values = value if isinstance(value, list) else [value]
                profile[field] = _merge_list(profile.get(field), [str(v) for v in values if not _emptyish(v)])
            elif field == "intent_specific":
                if isinstance(value, dict):
                    profile[field] = _merge_dict(profile.get(field), value)
            elif field == "intent_family":
                normalized = normalize_intent_family(value)
                if normalized != "unknown":
                    profile[field] = normalized
            elif field == "sub_intent":
                normalized = normalize_sub_intent(value)
                if normalized != "unknown":
                    profile[field] = normalized
            else:
                profile[field] = str(value).strip()
    return _sync_intent_specific_fields(profile)


def merge_profiles(primary: Optional[Dict[str, Any]], fallback: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Prefer primary profile values, but preserve fallback fields the LLM omitted.
    """
    return normalize_profile(primary or {}, fallback=fallback or {})


def normalize_experiment_goal(
    source_query: str,
    intent: Dict[str, Any],
    profile: Dict[str, Any],
) -> tuple:
    """
    Keep the user's primary experimental goal stable across clarifications.

    A later answer like "stable transformation" describes delivery context for
    overexpression; it should not downgrade the goal to plain transformation.
    """
    normalized_intent = dict(intent or {})
    normalized_profile = normalize_profile(profile or {})
    if not _has_overexpression_goal(source_query):
        return normalized_intent, normalized_profile

    raw_intent = str(normalized_intent.get("intent") or "").lower()
    method = str(normalized_profile.get("experimental_method") or "").lower()
    if raw_intent != "gene_overexpression":
        try:
            confidence = float(normalized_intent.get("confidence") or 0.0)
        except Exception:
            confidence = 0.0
        normalized_intent = {
            **normalized_intent,
            "intent": "gene_overexpression",
            "label": "gene overexpression",
            "intent_family": "gene_nucleic_acid_manipulation",
            "intent_family_label": INTENT_FAMILY_LABELS["gene_nucleic_acid_manipulation"],
            "sub_intent": "gene_overexpression",
            "sub_intent_label": "gene overexpression",
            "confidence": max(confidence, 0.82),
        }
    if (
        not method
        or method in {"transformation", "plant transformation", "genetic transformation", "transgenic transformation"}
        or ("transformation" in method and "overexpression" not in method and "transgene expression" not in method)
    ):
        normalized_profile["experimental_method"] = "gene overexpression"
    normalized_profile["method"] = "gene overexpression"
    normalized_profile["intent_family"] = "gene_nucleic_acid_manipulation"
    normalized_profile["sub_intent"] = "gene_overexpression"
    normalized_profile["intent_specific"] = _merge_dict(
        normalized_profile.get("intent_specific"),
        {"modification_type": "overexpression"},
    )
    if not normalized_profile.get("modification_type"):
        normalized_profile["modification_type"] = "overexpression"
    return normalized_intent, normalized_profile


def validate_biology_profile(
    source_query: str,
    intent: Dict[str, Any],
    profile: Dict[str, Any],
) -> tuple:
    """
    Deterministic validation layer for LLM-first extraction.

    The LLM can propose the profile, but controlled intent names and critical
    biology distinctions are enforced here before clarification/query logic runs.
    """
    normalized_profile = normalize_profile(profile or {})
    normalized_intent = controlled_intent_payload(
        (intent or {}).get("sub_intent") or (intent or {}).get("intent") or normalized_profile.get("sub_intent"),
        intent_family=(intent or {}).get("intent_family") or normalized_profile.get("intent_family"),
        confidence=(intent or {}).get("confidence", 0.7),
        alternatives=(intent or {}).get("alternatives", []),
    )
    q = str(source_query or "").lower()
    sub_intent = normalized_intent.get("sub_intent") or normalized_intent.get("intent") or "unknown"

    if _has_stress_tolerance_goal(q):
        sub_intent = "stress_tolerance_assay"
    elif _has_multiplex_gene_modification_goal(q):
        sub_intent = "multiplex_gene_modification"
    elif _has_overexpression_goal(q):
        sub_intent = "gene_overexpression"
    elif _has_knockdown_goal(q):
        sub_intent = "gene_knockdown"
    elif _has_explicit_genome_editing_goal(q):
        sub_intent = "genome_editing"
    elif _has_ambiguous_gene_modification_goal(q):
        sub_intent = "gene_modification"
    else:
        sub_intent = normalize_sub_intent(sub_intent)

    persisted_modification_sub_intent = _sub_intent_from_modification_type(
        _profile_value(normalized_profile, "modification_type")
    )
    if (
        sub_intent == "gene_modification"
        and persisted_modification_sub_intent
        and _has_ambiguous_gene_modification_goal(q)
    ):
        sub_intent = persisted_modification_sub_intent

    if sub_intent == "unknown" and not _emptyish(normalized_profile.get("experimental_method")):
        sub_intent = normalize_sub_intent(normalized_profile.get("experimental_method"))

    intent_family = family_for_sub_intent(sub_intent)
    label = INTENT_LABELS.get(sub_intent, sub_intent.replace("_", " "))
    normalized_intent = {
        **normalized_intent,
        "intent": sub_intent,
        "label": label,
        "intent_family": intent_family,
        "intent_family_label": INTENT_FAMILY_LABELS.get(intent_family, intent_family.replace("_", " ")),
        "sub_intent": sub_intent,
        "sub_intent_label": label,
    }
    normalized_profile["intent_family"] = intent_family
    normalized_profile["sub_intent"] = sub_intent
    if sub_intent not in {"unknown", "chitchat"}:
        normalized_profile["method"] = label
        normalized_profile["experimental_method"] = label

    organism = _extract_organism(q)
    if organism:
        current = normalized_profile.get("organism")
        current_str = str(current or "")
        keep_current = (
            # Rule-extracted value is generic but the current one is specific.
            (_is_generic(organism) and current and not _is_generic(current_str))
            # Current value already contains the rule-extracted organism, i.e. it
            # is more complete (e.g. "rice or tomato" contains "rice"). The rule
            # extractor only returns the first organism, so overwriting here would
            # drop the others — keep the fuller value.
            or (current_str and organism.lower() in current_str.lower())
        )
        if not keep_current:
            normalized_profile["organism"] = organism

    target = _extract_target(source_query)
    if target:
        normalized_profile["target"] = target

    if normalized_profile.get("gene_or_construct") and not normalized_profile.get("target"):
        normalized_profile["target"] = normalized_profile["gene_or_construct"]

    tissue = _first_pattern_match(q, _TISSUE_TERMS)
    if tissue:
        normalized_profile["tissue_or_cell_type"] = tissue
    if normalized_profile.get("tissue_or_cell_type") and not normalized_profile.get("sample_type"):
        normalized_profile["sample_type"] = normalized_profile["tissue_or_cell_type"]

    if normalized_profile.get("readout_assay") and not normalized_profile.get("readout"):
        normalized_profile["readout"] = normalized_profile["readout_assay"]
    if normalized_profile.get("readout") and not normalized_profile.get("readout_assay"):
        normalized_profile["readout_assay"] = normalized_profile["readout"]

    if normalized_profile.get("required_equipment"):
        normalized_profile["equipment"] = _merge_list(
            normalized_profile.get("equipment"),
            normalized_profile.get("required_equipment") or [],
        )
    if normalized_profile.get("equipment"):
        normalized_profile["required_equipment"] = _merge_list(
            normalized_profile.get("required_equipment"),
            normalized_profile.get("equipment") or [],
        )
    if normalized_profile.get("protocol_difficulty") and not normalized_profile.get("difficulty"):
        normalized_profile["difficulty"] = normalized_profile["protocol_difficulty"]
    if normalized_profile.get("difficulty") and not normalized_profile.get("protocol_difficulty"):
        normalized_profile["protocol_difficulty"] = normalized_profile["difficulty"]

    intent_specific = normalized_profile.get("intent_specific") if isinstance(normalized_profile.get("intent_specific"), dict) else {}

    stress_type = _extract_stress_type(q)
    if sub_intent == "stress_tolerance_assay":
        stress_type = stress_type or intent_specific.get("stress_type") or "stress"
        normalized_profile["condition"] = normalized_profile.get("condition") or f"{stress_type} stress"
        if _emptyish(normalized_profile.get("readout_assay")) or str(normalized_profile.get("readout_assay")).lower() == "stress response":
            normalized_profile["readout_assay"] = f"{stress_type} tolerance / phenotype"
        if _emptyish(normalized_profile.get("readout")) or str(normalized_profile.get("readout")).lower() == "stress response":
            normalized_profile["readout"] = normalized_profile["readout_assay"]
        intent_specific = _merge_dict(intent_specific, {"stress_type": stress_type})
        growth_stage = _extract_growth_stage(q)
        if growth_stage:
            intent_specific = _merge_dict(intent_specific, {"growth_stage": growth_stage})
        treatment_condition = _extract_treatment_condition(q)
        if treatment_condition:
            normalized_profile["condition"] = normalized_profile.get("condition") or treatment_condition
            intent_specific = _merge_dict(intent_specific, {"treatment_condition": treatment_condition})

    if sub_intent == "multiplex_gene_modification":
        intent_specific = _merge_dict(intent_specific, {"multiplex": True})

    if sub_intent == "gene_overexpression":
        normalized_profile["modification_type"] = "overexpression"
        intent_specific = _merge_dict(intent_specific, {"modification_type": "overexpression"})
        expression = normalized_profile.get("expression_type")
        if _is_stable_transformation(expression):
            intent_specific = _merge_dict(intent_specific, {"delivery_mode": "plant transformation / transgenic transformation"})
            if not normalized_profile.get("delivery_method"):
                normalized_profile["delivery_method"] = "plant transformation / transgenic transformation"

    if sub_intent == "gene_knockdown":
        normalized_profile["modification_type"] = normalized_profile.get("modification_type") or "knockdown / silencing"
        intent_specific = _merge_dict(intent_specific, {"modification_type": normalized_profile["modification_type"]})

    if sub_intent == "genome_editing":
        normalized_profile["modification_type"] = normalized_profile.get("modification_type") or "CRISPR / genome editing"
        intent_specific = _merge_dict(intent_specific, {"modification_type": normalized_profile["modification_type"]})

    if sub_intent in {"gene_modification", "multiplex_gene_modification"}:
        if _ambiguous_modification_without_specific_type(q):
            normalized_profile["modification_type"] = None
        if normalized_profile.get("modification_type"):
            intent_specific = _merge_dict(intent_specific, {"modification_type": normalized_profile["modification_type"]})
        if normalized_profile.get("delivery_method"):
            intent_specific = _merge_dict(intent_specific, {"delivery_method": normalized_profile["delivery_method"]})

    if sub_intent in {"protein_purification", "protein_extraction", "western_blot", "protein_detection"}:
        intent_specific = _merge_dict(intent_specific, _extract_protein_specifics(q))

    normalized_profile["intent_specific"] = intent_specific
    normalized_profile = _sync_intent_specific_fields(normalized_profile)
    normalized_profile = _clear_incompatible_profile_fields(normalized_profile)
    return normalized_intent, normalized_profile


def classify_organism_system(profile: Dict[str, Any]) -> str:
    """
    Coarse organism/system class for clarification templates.

    Classes: plant, mammalian_in_vivo, mammalian_cells, insect, fish, worm,
    amphibian, bacteria, yeast, fungi, algae, cell_free, unknown.
    """
    organism = str((profile or {}).get("organism") or "").strip().lower()
    tissue = str((profile or {}).get("tissue_or_cell_type") or (profile or {}).get("sample_type") or "").strip().lower()
    text = " ".join([organism, tissue])

    plant_terms = [
        "plant", "arabidopsis", "nicotiana", "tobacco", "rice", "oryza", "maize",
        "zea mays", "corn", "wheat", "tomato", "soybean", "bean", "phaseolus",
        "pumpkin", "cucurbita", "squash", "cucumber", "potato", "pea", "in planta",
        "leaf", "leaves", "callus", "protoplast", "immature embryo",
    ]
    insect_terms = ["drosophila", "fruit fly", "mosquito", "anopheles", "aedes", "silkworm", "bombyx", "insect"]
    fish_terms = ["zebrafish", "danio rerio", "medaka", "fish"]
    worm_terms = ["c. elegans", "caenorhabditis", "planaria", "nematode", "worm"]
    amphibian_terms = ["xenopus", "frog", "axolotl", "amphibian", "newt"]
    mammalian_in_vivo_terms = ["mouse", "mice", "rat", "animal", "in vivo", "whole animal", "primate", "rabbit", "hamster", "pig"]
    mammalian_cell_terms = ["human", "homo sapiens", "mammalian", "mammalian cells", "human cells", "hela", "hek293", "jurkat"]
    bacteria_terms = ["bacteria", "bacterial", "e. coli", "escherichia coli", "ecoli", "agrobacterium"]
    yeast_terms = ["yeast", "saccharomyces", "pichia", "schizosaccharomyces"]
    fungi_terms = ["aspergillus", "neurospora", "fungus", "fungi", "fungal", "trichoderma", "penicillium"]
    algae_terms = ["chlamydomonas", "algae", "algal", "microalgae", "chlorella", "spirulina", "cyanobacteria"]
    cell_free_terms = ["cell-free", "cell free", "in vitro transcription", "in vitro translation", "lysate"]

    if _has_any_term(text, plant_terms):
        return "plant"
    if _has_any_term(text, insect_terms):
        return "insect"
    if _has_any_term(text, fish_terms):
        return "fish"
    if _has_any_term(text, worm_terms):
        return "worm"
    if _has_any_term(text, amphibian_terms):
        return "amphibian"
    if _has_any_term(text, mammalian_in_vivo_terms):
        return "mammalian_in_vivo"
    if _has_any_term(text, mammalian_cell_terms):
        return "mammalian_cells"
    if _has_any_term(text, bacteria_terms):
        return "bacteria"
    if _has_any_term(text, yeast_terms):
        return "yeast"
    if _has_any_term(text, fungi_terms):
        return "fungi"
    if _has_any_term(text, algae_terms):
        return "algae"
    if _has_any_term(text, cell_free_terms):
        return "cell_free"
    return "unknown"


def organism_aware_system_clarification(profile: Dict[str, Any]) -> Dict[str, Any]:
    organism_class = classify_organism_system(profile)
    templates = {
        "plant": {
            "question": "Where should the modification be tested?",
            "options": [
                "in planta / whole plant",
                "leaf tissue",
                "callus / tissue culture",
                "immature embryos",
                "protoplasts",
                "not sure",
            ],
        },
        "mammalian_in_vivo": {
            "question": "What animal system should be modified or tested?",
            "options": [
                "whole animal / in vivo",
                "embryo",
                "primary cells",
                "cell line",
                "specific tissue or organ",
                "organoid",
                "not sure",
            ],
        },
        "mammalian_cells": {
            "question": "What cell system or sample type should be modified?",
            "options": ["cell line", "primary cells", "organoid", "tissue sample", "not sure"],
        },
        "insect": {
            "question": "What insect system should be modified or tested?",
            "options": ["whole organism", "embryo", "larva", "cell line (S2, Sf9)", "tissue", "not sure"],
        },
        "fish": {
            "question": "What fish system should be modified or tested?",
            "options": ["whole organism / embryo", "larva", "adult tissue", "cell line", "not sure"],
        },
        "worm": {
            "question": "What worm system should be modified or tested?",
            "options": ["whole organism", "embryo", "larva / adult", "not sure"],
        },
        "amphibian": {
            "question": "What amphibian system should be modified or tested?",
            "options": ["embryo / oocyte", "tadpole", "whole organism", "tissue explant", "not sure"],
        },
        "bacteria": {
            "question": "What sample or culture system should be modified?",
            "options": ["bacterial culture", "colony", "plasmid-bearing cells", "not sure"],
        },
        "yeast": {
            "question": "What yeast system should be modified?",
            "options": ["yeast culture", "colony", "cells", "not sure"],
        },
        "fungi": {
            "question": "What fungal system should be modified?",
            "options": ["mycelium / culture", "spores", "protoplasts", "not sure"],
        },
        "algae": {
            "question": "What algal system should be modified?",
            "options": ["liquid culture", "colonies / plates", "single cells", "not sure"],
        },
        "cell_free": {
            "question": "What cell-free system are you using?",
            "options": ["in vitro transcription/translation", "cell lysate", "reconstituted system", "not sure"],
        },
        "unknown": {
            "question": "What system should be modified or tested?",
            "options": ["whole organism", "tissue", "primary cells", "cell line", "isolated cells/protoplasts", "not sure"],
        },
    }
    template = templates.get(organism_class, templates["unknown"])
    return {
        "field": "tissue_or_cell_type",
        "question": template["question"],
        "options": template["options"],
    }


def _organism_clarification_for_context(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Return organism options filtered by what the profile already implies."""
    system_class = classify_organism_system(profile)
    templates = {
        "plant": {
            "question": "What plant species are you working with?",
            "options": ["Arabidopsis thaliana", "Nicotiana benthamiana", "rice", "maize", "wheat", "tomato"],
        },
        "mammalian_in_vivo": {
            "question": "What animal model are you working with?",
            "options": ["mouse", "rat", "zebrafish", "Drosophila"],
        },
        "mammalian_cells": {
            "question": "What cell system are you working with?",
            "options": ["human cells", "mouse cells", "primary cells", "cell line"],
        },
        "insect": {
            "question": "What insect species are you working with?",
            "options": ["Drosophila", "mosquito (Anopheles/Aedes)", "silkworm (Bombyx mori)"],
        },
        "fish": {
            "question": "What fish species are you working with?",
            "options": ["zebrafish", "medaka"],
        },
        "worm": {
            "question": "What worm species are you working with?",
            "options": ["C. elegans", "planaria"],
        },
        "amphibian": {
            "question": "What amphibian species are you working with?",
            "options": ["Xenopus laevis", "Xenopus tropicalis", "axolotl"],
        },
        "bacteria": {
            "question": "What bacterial system are you working with?",
            "options": ["E. coli", "Agrobacterium", "other bacteria"],
        },
        "yeast": {
            "question": "What yeast species are you working with?",
            "options": ["Saccharomyces cerevisiae", "Pichia pastoris", "Schizosaccharomyces pombe"],
        },
        "fungi": {
            "question": "What fungal species are you working with?",
            "options": ["Aspergillus", "Neurospora", "Trichoderma"],
        },
        "algae": {
            "question": "What algal species are you working with?",
            "options": ["Chlamydomonas", "Chlorella", "cyanobacteria"],
        },
        "cell_free": {
            "question": "What cell-free system are you using?",
            "options": ["in vitro transcription/translation", "cell lysate", "reconstituted system"],
        },
        "unknown": {
            "question": "What organism or experimental system are you working with?",
            "options": ["plant", "mouse / animal", "human cells", "bacteria", "yeast", "other"],
        },
    }
    template = templates.get(system_class, templates["unknown"])
    return {
        "field": "organism",
        "question": template["question"],
        "options": template["options"],
    }


def next_biology_clarification(
    profile: Dict[str, Any],
    intent: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    profile = _sync_intent_specific_fields(dict(profile or {}))
    sub_intent = normalize_sub_intent(
        (profile or {}).get("sub_intent")
        or (intent or {}).get("sub_intent")
        or (intent or {}).get("intent")
        or (profile or {}).get("experimental_method")
    )
    intent_specific = profile.get("intent_specific") if isinstance(profile.get("intent_specific"), dict) else {}

    if sub_intent in {"gene_modification", "genome_editing", "multiplex_gene_modification"}:
        modification_type = _profile_value(profile, "modification_type")
        delivery_method = _profile_value(profile, "delivery_method")
        if needs_clarification(profile, "modification_type"):
            return profile_schema.field_clarification("modification_type")
        if needs_clarification(profile, "organism"):
            return _organism_clarification_for_context(profile)
        if sub_intent == "multiplex_gene_modification" and needs_clarification(profile, "target"):
            return {
                "field": "target",
                "question": "What gene targets or target class should be modified?",
                "options": ["transcription factors", "two named genes", "gene family", "not sure"],
            }
        if needs_clarification(profile, "tissue_or_cell_type"):
            return organism_aware_system_clarification(profile)
        if needs_clarification(profile, "delivery_method") and not _has_delivery_context(modification_type):
            return profile_schema.field_clarification("delivery_method")
        if needs_clarification(profile, "readout_assay"):
            return {
                "field": "readout_assay",
                "question": "What readout do you care about?",
                "options": ["phenotype", "genotyping / sequencing", "RNA level / qPCR", "protein level"],
            }

    if sub_intent == "stress_tolerance_assay":
        if needs_clarification(profile, "organism"):
            return _organism_clarification_for_context(profile)
        if (
            not intent_specific.get("growth_stage")
            and needs_clarification(profile, "growth_stage")
            and needs_clarification(profile, "tissue_or_cell_type")
        ):
            return {
                "field": "growth_stage",
                "question": "What growth stage or sample should be tested?",
                "options": ["seedlings", "adult plants", "leaf tissue", "not sure"],
            }
        if (
            not intent_specific.get("treatment_condition")
            and needs_clarification(profile, "treatment_condition")
            and needs_clarification(profile, "condition")
        ):
            return {
                "field": "treatment_condition",
                "question": "What stress treatment condition should the protocol use?",
                "options": ["drought stress", "PEG/osmotic stress", "withholding water", "not sure"],
            }
        if needs_clarification(profile, "readout_assay"):
            return {
                "field": "readout_assay",
                "question": "How should tolerance be measured?",
                "options": ["survival rate", "growth/biomass", "water loss", "phenotype score"],
            }

    if sub_intent == "protein_purification":
        if (
            not intent_specific.get("expression_host")
            and needs_clarification(profile, "expression_host")
            and needs_clarification(profile, "organism")
        ):
            return {
                "field": "expression_host",
                "question": "What expression host or protein source are you using?",
                "options": ["E. coli", "yeast", "mammalian cells", "plant tissue"],
            }
        if needs_clarification(profile, "target"):
            return {
                "field": "target",
                "question": "What protein or tag are you purifying?",
                "options": ["His-tagged protein", "GST-tagged protein", "native protein", "not sure"],
            }
        if not intent_specific.get("purification_method") and needs_clarification(profile, "purification_method"):
            return {
                "field": "purification_method",
                "question": "What purification approach do you need?",
                "options": ["His-tag affinity", "GST affinity", "native purification", "not sure"],
            }

    return None


def can_generate_search_queries(profile: Dict[str, Any]) -> bool:
    profile = _sync_intent_specific_fields(dict(profile or {}))
    sub_intent = normalize_sub_intent(profile.get("sub_intent") or profile.get("experimental_method"))
    intent_specific = profile.get("intent_specific") if isinstance(profile.get("intent_specific"), dict) else {}
    modification_type = _profile_value(profile, "modification_type")

    if sub_intent in {"gene_modification", "genome_editing"}:
        return bool(
            not needs_clarification(profile, "modification_type")
            and not needs_clarification(profile, "organism")
            and not needs_clarification(profile, "tissue_or_cell_type")
        )
    if sub_intent == "multiplex_gene_modification":
        return bool(
            not needs_clarification(profile, "modification_type")
            and not needs_clarification(profile, "organism")
            and not needs_clarification(profile, "tissue_or_cell_type")
            and not needs_clarification(profile, "target")
        )
    if sub_intent == "stress_tolerance_assay":
        return bool(
            not needs_clarification(profile, "organism")
            and intent_specific.get("stress_type")
            and (intent_specific.get("growth_stage") or not needs_clarification(profile, "tissue_or_cell_type") or not needs_clarification(profile, "readout_assay"))
        )
    if sub_intent == "protein_purification":
        return bool(
            profile.get("target")
            or profile.get("gene_or_construct")
            or intent_specific.get("protein_source")
            or intent_specific.get("expression_host")
            or not _missing_or_generic(profile.get("organism"))
        )
    if sub_intent in {"unknown", "chitchat"}:
        return False
    return bool(
        not _emptyish(profile.get("method"))
        or not _emptyish(profile.get("experimental_method"))
        or not _emptyish(profile.get("target"))
        or not _emptyish(profile.get("gene_or_construct"))
    )


def profile_source_query_for_request(
    query: str,
    conversation_query: str = "",
    search_confirmed: bool = False,
    experiment_profile: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Decide which text is safe to use for profile extraction.

    Once a user confirms a suggested search query, that query is search input
    only. It should not mutate the structured experiment profile.
    """
    clean_query = (query or "").strip()
    clean_conversation = (conversation_query or "").strip()
    has_profile = isinstance(experiment_profile, dict) and any(
        not _emptyish(value) for value in experiment_profile.values()
    )
    if search_confirmed and has_profile:
        return clean_conversation
    return "\n".join(part for part in [clean_conversation, clean_query] if part) or clean_query


def detect_experiment_intent(query: str) -> Dict[str, Any]:
    q = query.lower()
    matches: List[Dict[str, Any]] = []
    for intent, pattern in _INTENT_PATTERNS:
        if re.search(pattern, q):
            matches.append(controlled_intent_payload(intent, confidence=0.9 if not matches else 0.72))
    if not matches:
        return controlled_intent_payload("unknown", confidence=0.2)
    primary = matches[0]
    return {
        "intent": primary["intent"],
        "label": primary["label"],
        "intent_family": primary["intent_family"],
        "intent_family_label": primary["intent_family_label"],
        "sub_intent": primary["sub_intent"],
        "sub_intent_label": primary["sub_intent_label"],
        "confidence": primary["confidence"],
        "alternatives": matches[1:],
    }


def build_experiment_profile(
    query: str,
    previous_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    profile = empty_profile()
    if isinstance(previous_profile, dict):
        for field in PROFILE_FIELDS:
            value = previous_profile.get(field)
            if value not in (None, "", []):
                profile[field] = deepcopy(value)

    q = query.lower()
    intent = detect_experiment_intent(query)
    if intent["intent"] != "unknown":
        label = INTENT_LABELS[intent["intent"]]
        profile["intent_family"] = intent.get("intent_family") or family_for_sub_intent(intent["intent"])
        profile["sub_intent"] = intent["intent"]
        profile["method"] = label
        profile["experimental_method"] = label

    organism = _extract_organism(q)
    if organism:
        current = profile.get("organism")
        if not (_is_generic(organism) and current and not _is_generic(str(current))):
            profile["organism"] = organism
    if _missing_or_generic(profile.get("organism")):
        freeform_organism = _extract_freeform_organism_answer(query)
        if freeform_organism:
            profile["organism"] = freeform_organism

    gene = _extract_gene_or_construct(query)
    if gene:
        profile["gene_or_construct"] = gene
        profile["target"] = profile.get("target") or gene

    target = _extract_target(query)
    if target:
        profile["target"] = target

    for field, patterns in (
        ("modification_type", _MODIFICATION_TYPES),
        ("delivery_method", _DELIVERY_METHODS),
        ("tissue_or_cell_type", _TISSUE_TERMS),
        ("expression_type", _EXPRESSION_TYPES),
        ("readout_assay", _READOUTS),
    ):
        value = _first_pattern_match(q, patterns)
        if value:
            profile[field] = value

    if profile.get("tissue_or_cell_type") and not profile.get("sample_type"):
        profile["sample_type"] = profile["tissue_or_cell_type"]
    if profile.get("readout_assay") and not profile.get("readout"):
        profile["readout"] = profile["readout_assay"]

    timeline = _extract_timeline(query)
    if timeline:
        profile["timeline"] = timeline

    equipment = _extract_many(q, _EQUIPMENT)
    profile["required_equipment"] = _merge_list(
        profile.get("required_equipment"),
        equipment,
    )
    profile["equipment"] = _merge_list(
        profile.get("equipment"),
        equipment,
    )
    profile["constraints"] = _merge_list(
        profile.get("constraints"),
        _extract_constraints(query),
    )

    difficulty = _extract_difficulty(q)
    if difficulty:
        profile["protocol_difficulty"] = difficulty
        profile["difficulty"] = difficulty

    stress_type = _extract_stress_type(q)
    if stress_type:
        profile["condition"] = profile.get("condition") or f"{stress_type} stress"
        if not profile.get("readout_assay"):
            profile["readout_assay"] = f"{stress_type} tolerance / phenotype"
            profile["readout"] = profile["readout_assay"]
        profile["intent_specific"] = _merge_dict(profile.get("intent_specific"), {"stress_type": stress_type})

    growth_stage = _extract_growth_stage(q)
    if growth_stage:
        profile["intent_specific"] = _merge_dict(profile.get("intent_specific"), {"growth_stage": growth_stage})

    treatment_condition = _extract_treatment_condition(q)
    if treatment_condition:
        profile["condition"] = profile.get("condition") or treatment_condition
        profile["intent_specific"] = _merge_dict(profile.get("intent_specific"), {"treatment_condition": treatment_condition})

    profile["intent_specific"] = _merge_dict(
        profile.get("intent_specific"),
        _extract_protein_specifics(q),
    )

    _, validated_profile = validate_biology_profile(query, intent, profile)
    return validated_profile


def next_clarification(
    profile: Dict[str, Any],
    intent: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    profile = _sync_intent_specific_fields(dict(profile or {}))
    biology_clarification = next_biology_clarification(profile, intent)
    if biology_clarification:
        return biology_clarification

    intent_name = intent.get("intent", "unknown")
    method = profile.get("experimental_method") or INTENT_LABELS.get(intent_name, "")

    if intent_name == "gene_modification" or method == "gene modification":
        if needs_clarification(profile, "modification_type"):
            return profile_schema.field_clarification("modification_type")
        if needs_clarification(profile, "organism"):
            return _organism_clarification_for_context(profile)
        if needs_clarification(profile, "tissue_or_cell_type"):
            return organism_aware_system_clarification(profile)
        if needs_clarification(profile, "delivery_method") and not _has_delivery_context(_profile_value(profile, "modification_type")):
            return profile_schema.field_clarification("delivery_method")
        if needs_clarification(profile, "readout_assay"):
            return {
                "field": "readout_assay",
                "question": "What readout do you care about?",
                "options": ["phenotype", "genotyping / sequencing", "RNA level / qPCR", "protein level"],
            }

    if intent_name == "gene_overexpression" or method == "gene overexpression":
        if needs_clarification(profile, "organism"):
            return _organism_clarification_for_context(profile)
        if needs_clarification(profile, "expression_type"):
            return {
                "field": "expression_type",
                "question": "Do you want stable whole-plant overexpression or transient expression?",
                "options": ["stable transformation", "transient expression", "either / not sure"],
            }
        if needs_clarification(profile, "tissue_or_cell_type"):
            if _is_stable_transformation(profile.get("expression_type")):
                return {
                    "field": "tissue_or_cell_type",
                    "question": "What starting material or transformation system should be used?",
                    "options": [
                        "cotyledons / explants",
                        "callus / tissue culture",
                        "immature embryos",
                        "floral dip / whole plant",
                        "not sure / flexible",
                    ],
                }
            return {
                "field": "tissue_or_cell_type",
                "question": "Should overexpression happen in the whole plant, a tissue such as leaves, or isolated cells/protoplasts?",
                "options": ["whole plant", "leaf tissue", "protoplasts", "cell culture"],
            }
        if needs_clarification(profile, "readout_assay"):
            return {
                "field": "readout_assay",
                "question": "What readout do you care about?",
                "options": ["phenotype", "RNA level / qPCR", "protein level", "localization"],
            }

    if intent_name in {"gene_knockdown", "transformation", "pcr_qpcr", "microscopy", "sequencing_prep", "protein_purification"}:
        if needs_clarification(profile, "organism"):
            return _organism_clarification_for_context(profile)
        if intent_name == "pcr_qpcr" and needs_clarification(profile, "readout_assay"):
            return {
                "field": "readout_assay",
                "question": "What kind of PCR result are you trying to get?",
                "options": ["standard PCR", "qPCR quantification", "RT-qPCR", "colony PCR"],
            }
        if intent_name == "microscopy" and needs_clarification(profile, "tissue_or_cell_type"):
            return {
                "field": "tissue_or_cell_type",
                "question": "What sample will you image?",
                "options": ["fixed cells", "live cells", "plant tissue", "tissue section"],
            }
        if intent_name == "sequencing_prep" and needs_clarification(profile, "readout_assay"):
            return {
                "field": "readout_assay",
                "question": "What sequencing workflow are you looking for?",
                "options": ["RNA-seq", "amplicon sequencing", "Sanger sequencing", "ChIP-seq"],
            }

    return None


def profile_missing_fields(profile: Dict[str, Any]) -> List[str]:
    missing = []
    for field in PROFILE_FIELDS:
        if field in {
            "intent_family",
            "sub_intent",
            "method",
            "readout",
            "sample_type",
            "target",
            "condition",
            "equipment",
            "required_equipment",
            "difficulty",
            "constraints",
            "intent_specific",
        }:
            continue
        if profile.get(field) in (None, "", []):
            missing.append(field)
    return missing


def profile_to_search_query(profile: Dict[str, Any], fallback_query: str) -> str:
    profile = _sync_intent_specific_fields(dict(profile or {}))
    terms: List[str] = []
    # Field order for the structured search query is declared in the schema
    # (profile_schema.QUERY_FIELD_ORDER).
    for field in profile_schema.QUERY_FIELD_ORDER:
        value = profile.get(field)
        if value and not _is_generic(value):
            terms.append(str(value))
    intent_specific = profile.get("intent_specific") if isinstance(profile.get("intent_specific"), dict) else {}
    for value in intent_specific.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, list):
            terms.extend(str(item) for item in value if not _emptyish(item))
        elif not _emptyish(value):
            terms.append(str(value))
    terms += [str(x) for x in profile.get("constraints") or []]

    structured = " ".join(_dedup(terms))
    if structured:
        return structured
    return fallback_query


def generate_candidate_search_queries(
    profile: Dict[str, Any],
    fallback_query: str = "",
    max_queries: int = 5,
) -> List[str]:
    """
    Rule-based fallback for the LLM candidate-query step.
    """
    profile = _sync_intent_specific_fields(dict(profile or {}))
    if not _profile_has_search_concepts(profile):
        fallback_query = " ".join(str(fallback_query or "").split())
        if fallback_query and detect_experiment_intent(fallback_query).get("intent") != "unknown":
            return [fallback_query][:max_queries]
        return []
    if not can_generate_search_queries(profile):
        return []

    organism = profile.get("organism")
    method = profile.get("experimental_method") or profile.get("method")
    sub_intent = normalize_sub_intent(profile.get("sub_intent") or method)
    modification_type = _profile_value(profile, "modification_type")
    delivery_method = _profile_value(profile, "delivery_method")
    expression = profile.get("expression_type")
    tissue = profile.get("tissue_or_cell_type")
    readout = profile.get("readout_assay")
    target = profile.get("target") or profile.get("gene_or_construct")
    condition = profile.get("condition")
    intent_specific = profile.get("intent_specific") if isinstance(profile.get("intent_specific"), dict) else {}

    org_terms = _query_variants(organism, _ORGANISM_SEARCH_VARIANTS)
    method_terms = _query_variants(method, _METHOD_SEARCH_VARIANTS)
    modification_terms = _query_variants(modification_type, _MODIFICATION_SEARCH_VARIANTS)
    delivery_terms = _query_variants(delivery_method, _DELIVERY_SEARCH_VARIANTS)
    expression_terms = _query_variants(expression, _EXPRESSION_SEARCH_VARIANTS)
    readout_terms = _query_variants(readout, _READOUT_SEARCH_VARIANTS)
    tissue_term = None if _emptyish(tissue) else str(tissue)

    org = _first(org_terms)
    org_alt = org_terms[1] if len(org_terms) > 1 else org
    target_term = None if _emptyish(target) else str(target)

    if sub_intent in {"gene_modification", "genome_editing", "multiplex_gene_modification"}:
        if _emptyish(modification_type) or str(modification_type).strip().lower() == "not sure":
            return []
        mod = _first(modification_terms) or str(modification_type)
        mod_alt = modification_terms[1] if len(modification_terms) > 1 else mod
        delivery = _first(delivery_terms)
        delivery_alt = delivery_terms[1] if len(delivery_terms) > 1 else delivery
        multiplex = "multiplex" if sub_intent == "multiplex_gene_modification" or intent_specific.get("multiplex") else None
        candidates = [
            _join_terms([org, multiplex, mod, target_term, tissue_term, "protocol"]),
            _join_terms([org_alt, "simultaneous" if multiplex else None, target_term, mod_alt, delivery]),
            _join_terms([org, mod, delivery_alt, readout]),
            _join_terms([org, tissue_term, mod_alt, target_term]),
            profile_to_search_query(profile, fallback_query),
        ]
        if fallback_query and candidate_query_preserves_required_concepts(profile, fallback_query):
            candidates.append(fallback_query)
        cleaned = [" ".join(str(candidate or "").split()) for candidate in candidates]
        return _dedup([candidate for candidate in cleaned if len(candidate) > 2])[:max_queries]

    if sub_intent == "stress_tolerance_assay":
        stress_type = intent_specific.get("stress_type") or _extract_stress_type(str(condition or "").lower()) or "stress"
        growth_stage = intent_specific.get("growth_stage") or tissue_term
        candidates = [
            _join_terms([org, stress_type, "tolerance phenotyping protocol"]),
            _join_terms([org_alt, stress_type, "stress assay"]),
            _join_terms([org, growth_stage, stress_type, "tolerance protocol"]),
            _join_terms([org, condition, "phenotype assay"]),
            _join_terms([org_alt, stress_type, "treatment phenotype"]),
            profile_to_search_query(profile, fallback_query),
        ]
        cleaned = [" ".join(str(candidate or "").split()) for candidate in candidates]
        return _dedup([candidate for candidate in cleaned if len(candidate) > 2])[:max_queries]

    if sub_intent == "protein_purification":
        expression_host = intent_specific.get("expression_host")
        purification_method = intent_specific.get("purification_method")
        tag = intent_specific.get("tag")
        protein_source = intent_specific.get("protein_source")
        candidates = [
            _join_terms([target_term, expression_host or organism, purification_method, "protocol"]),
            _join_terms([expression_host or organism, tag, "protein purification"]),
            _join_terms([protein_source or organism, target_term, "protein extraction purification"]),
            _join_terms([target_term, "recombinant protein purification protocol"]),
            profile_to_search_query(profile, fallback_query),
        ]
        cleaned = [" ".join(str(candidate or "").split()) for candidate in candidates]
        return _dedup([candidate for candidate in cleaned if len(candidate) > 2])[:max_queries]

    if _is_gene_overexpression_method(method) and _is_transient_expression(expression):
        read = _first(readout_terms)
        read_alt = readout_terms[1] if len(readout_terms) > 1 else read
        read_assay = readout_terms[2] if len(readout_terms) > 2 else read_alt
        candidates = [
            _join_terms([org, "transient overexpression", read]),
            _join_terms([org_alt, "transient transgene expression", read_alt]),
            _join_terms([org, tissue_term, "transient overexpression", read_assay or "gene expression assay"]),
            _join_terms([org, "transient transformation overexpression", "RNA level" if _is_rna_qpcr_readout(readout) else read]),
            _join_terms([org, "transient transgene expression", read or "qPCR", "protocol"]),
            profile_to_search_query(profile, fallback_query),
        ]
        if fallback_query and candidate_query_preserves_required_concepts(profile, fallback_query):
            candidates.append(fallback_query)
        cleaned = [" ".join(str(candidate or "").split()) for candidate in candidates]
        return _dedup([candidate for candidate in cleaned if len(candidate) > 2])[:max_queries]

    if _is_gene_modification_method(method):
        if _emptyish(modification_type) or str(modification_type).strip().lower() == "not sure":
            return []
        mod = _first(modification_terms) or str(modification_type)
        mod_alt = modification_terms[1] if len(modification_terms) > 1 else mod
        delivery = _first(delivery_terms)
        delivery_alt = delivery_terms[1] if len(delivery_terms) > 1 else delivery
        candidates = [
            _join_terms([org, mod, tissue_term, "protocol"]),
            _join_terms([org_alt, mod_alt, tissue_term, delivery]),
            _join_terms([org, mod, delivery_alt, readout]),
            _join_terms([org, tissue_term, mod_alt]),
            profile_to_search_query(profile, fallback_query),
        ]
        if fallback_query:
            candidates.append(fallback_query)
        cleaned = [" ".join(str(candidate or "").split()) for candidate in candidates]
        return _dedup([candidate for candidate in cleaned if len(candidate) > 2])[:max_queries]

    if _is_gene_overexpression_method(method) and _is_stable_transformation(expression):
        candidates = [
            _join_terms([org, "stable transformation", "gene overexpression"]),
            _join_terms([org_alt, "transgenic overexpression protocol"]),
            _join_terms([org, "Agrobacterium transformation overexpression"]),
            _join_terms([org, "immature embryo transformation transgene expression"]),
            _join_terms([org, "biolistic transformation gene overexpression"]),
            _join_terms([org, "callus regeneration transgenic overexpression"]),
            profile_to_search_query(profile, fallback_query),
        ]
        if fallback_query:
            candidates.append(fallback_query)
        cleaned = [" ".join(str(candidate or "").split()) for candidate in candidates]
        return _dedup([candidate for candidate in cleaned if len(candidate) > 2])[:max_queries]

    method_main = _first(method_terms)
    method_alt = method_terms[1] if len(method_terms) > 1 else method_main
    expr = _first(expression_terms)
    expr_alt = expression_terms[1] if len(expression_terms) > 1 else expr
    method_after_expr = None
    if method_alt and not (expr_alt and method_alt.lower() in expr_alt.lower()):
        method_after_expr = method_alt
    read = _first(readout_terms)
    read_alt = readout_terms[1] if len(readout_terms) > 1 else read
    read_assay = readout_terms[2] if len(readout_terms) > 2 else read_alt

    candidates = [
        _join_terms([org, expr, tissue_term, read]),
        _join_terms([org, tissue_term, expr_alt, method_after_expr, read_alt]),
        _join_terms([org, method_main, tissue_term, read_assay]),
        _join_terms([org_alt, expr, read_alt]),
        _join_terms([org, method_alt, expr, tissue_term, "protocol"]),
        profile_to_search_query(profile, fallback_query),
    ]

    constraints = profile.get("constraints") or []
    if constraints:
        candidates.append(_join_terms([profile_to_search_query(profile, fallback_query), *constraints]))

    if fallback_query:
        candidates.append(fallback_query)

    cleaned = []
    for candidate in candidates:
        candidate = " ".join(str(candidate or "").split())
        if len(candidate) > 2:
            cleaned.append(candidate)
    return _dedup(cleaned)[:max_queries]


def candidate_query_preserves_required_concepts(profile: Dict[str, Any], query: str) -> bool:
    """
    Validate LLM-generated query chips for cases where broad rewrites are risky.
    """
    profile = _sync_intent_specific_fields(dict(profile or {}))
    sub_intent = normalize_sub_intent(profile.get("sub_intent") or profile.get("experimental_method"))
    text = str(query or "").lower()

    if sub_intent == "stress_tolerance_assay":
        intent_specific = profile.get("intent_specific") if isinstance(profile.get("intent_specific"), dict) else {}
        organism_terms = _query_variants(profile.get("organism"), _ORGANISM_SEARCH_VARIANTS)
        organism_ok = not organism_terms or any(_has_term(text, term) for term in organism_terms)
        stress_type = intent_specific.get("stress_type") or _extract_stress_type(str(profile.get("condition") or "").lower())
        stress_ok = not stress_type or _has_term(text, str(stress_type)) or _has_term(text, "stress")
        assay_ok = _has_any_term(text, ["tolerance", "phenotyping", "phenotype", "assay", "treatment"])
        return organism_ok and stress_ok and assay_ok

    if sub_intent == "protein_purification":
        target = profile.get("target") or profile.get("gene_or_construct")
        target_ok = _emptyish(target) or _has_term(text, str(target))
        protein_ok = _has_any_term(text, ["protein", "purification", "purify", "affinity"])
        return target_ok and protein_ok

    if _is_gene_modification_method(profile.get("experimental_method")):
        modification_terms = _query_variants(_profile_value(profile, "modification_type"), _MODIFICATION_SEARCH_VARIANTS)
        if not modification_terms:
            return False
        organism_terms = _query_variants(profile.get("organism"), _ORGANISM_SEARCH_VARIANTS)
        tissue = profile.get("tissue_or_cell_type")
        target = profile.get("target")
        organism_ok = not organism_terms or any(_has_term(text, term) for term in organism_terms)
        modification_ok = any(_has_term(text, term) for term in modification_terms)
        tissue_ok = _emptyish(tissue) or _is_generic(str(tissue)) or _has_term(text, str(tissue)) or (
            str(tissue).lower() == "in planta / whole plant" and _has_any_term(text, ["in planta", "whole plant"])
        )
        target_ok = _emptyish(target) or _has_term(text, str(target))
        return organism_ok and modification_ok and tissue_ok and target_ok

    if _is_gene_overexpression_method(profile.get("experimental_method")) and not _is_stable_transformation(profile.get("expression_type")):
        organism_terms = _query_variants(profile.get("organism"), _ORGANISM_SEARCH_VARIANTS)
        organism_ok = not organism_terms or any(_has_term(text, term) for term in organism_terms)
        overexpression_ok = _has_any_term(text, ["overexpression", "overexpress", "transgene expression"])
        expression_terms = _query_variants(profile.get("expression_type"), _EXPRESSION_SEARCH_VARIANTS)
        expression_ok = not expression_terms or any(_has_term(text, term) for term in expression_terms)
        readout_terms = _query_variants(profile.get("readout_assay") or profile.get("readout"), _READOUT_SEARCH_VARIANTS)
        readout_ok = not readout_terms or any(_has_term(text, term) for term in readout_terms)
        tissue = profile.get("tissue_or_cell_type")
        tissue_ok = _emptyish(tissue) or _is_generic(str(tissue)) or _has_term(text, str(tissue)) or (
            str(tissue).lower() == "in planta / whole plant" and _has_any_term(text, ["in planta", "whole plant", "plant"])
        )
        return organism_ok and overexpression_ok and expression_ok and readout_ok and tissue_ok

    if not (_is_gene_overexpression_method(profile.get("experimental_method")) and _is_stable_transformation(profile.get("expression_type"))):
        return True

    organism_terms = _query_variants(profile.get("organism"), _ORGANISM_SEARCH_VARIANTS)
    organism_ok = not organism_terms or any(_has_term(text, term) for term in organism_terms)
    overexpression_ok = _has_overexpression_goal(text)
    delivery_ok = _has_any_term(text, [
        "stable transformation",
        "transformation",
        "transgenic",
        "agrobacterium",
        "biolistic",
        "gene gun",
        "immature embryo",
        "callus",
        "regeneration",
    ])
    return organism_ok and overexpression_ok and delivery_ok


def should_respond_as_chitchat(
    has_active_experiment_context: bool,
    intent: Dict[str, Any],
    next_action: str = "",
) -> bool:
    """
    Short clarification answers can look like general words to an LLM.
    Once a protocol-search conversation is active, never answer with chitchat.
    """
    if has_active_experiment_context:
        return False
    return intent.get("intent") == "chitchat" or next_action == "respond_chitchat"


# blend_score is a cosine similarity in [0, ~0.3] in practice, while the
# field-match bonus below sums to ~1–4. This multiplier lifts text relevance onto
# the same scale so it competes as a primary signal for both sources.
# All ranking weights/thresholds are centralized in ranking_config (env-overridable).
_TEXT_RELEVANCE_WEIGHT = rank_cfg.TEXT_RELEVANCE_WEIGHT


def apply_profile_ranking(
    profile: Dict[str, Any],
    results: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    ranked = []
    for result in results:
        signals = _profile_match_signals(profile, result)
        # Cross-source relevance base. `blend_score` (title>body cosine on a shared
        # vocabulary — see blend_ranking.py) is computed identically for BOTH
        # protocols.io and PubMed, so it puts both sources on one comparable axis.
        # The old local-index `score` is protocols.io-only (PubMed had base 0),
        # which structurally sank PubMed regardless of relevance. Fall back to the
        # local score only when blend_score is absent (non-blended/single-source
        # paths). Weighted up (`_TEXT_RELEVANCE_WEIGHT`) so text relevance is a
        # primary ranking term, not a rounding-error tiebreaker next to the
        # field-match bonus below.
        blend = result.get("blend_score")
        base = float(blend if blend is not None else (result.get("score") or result.get("combined_score") or 0.0))
        # Profile-match bonus is driven by the declarative ranking schema
        # (profile_schema.RANKING_SCHEMA) — one place lists every signal + weight.
        bonus = profile_schema.ranking_bonus(signals)
        annotated = dict(result)
        raw_score = _TEXT_RELEVANCE_WEIGHT * base + bonus
        penalty = signals.get("false_positive_penalty", 1.0)
        if _has_required_concept_groups(profile) and signals.get("required_concept_match", 0.0) < rank_cfg.T_REQUIRED_CONCEPT_MIN:
            penalty *= rank_cfg.P_REQUIRED_CONCEPT
        if _is_strict_overexpression_readout_profile(profile):
            coverage = signals.get("required_concept_match", 0.0)
            if coverage < 0.6:
                penalty *= rank_cfg.P_STRICT_OX_LOW
            elif coverage < 0.8:
                penalty *= rank_cfg.P_STRICT_OX_MID
        annotated["profile_score"] = round(raw_score * penalty, 3)
        annotated["profile_signals"] = signals
        annotated.update(explain_protocol_match(profile, annotated))
        ranked.append(annotated)

    ranked.sort(key=lambda item: item.get("profile_score", item.get("score", 0)), reverse=True)
    severe_off_target = [
        item for item in ranked
        if item.get("profile_signals", {}).get("false_positive_penalty", 1.0) <= rank_cfg.T_SEVERE_OFF_TARGET
    ]
    acceptable = [item for item in ranked if item not in severe_off_target]
    if _requires_strict_required_concepts(profile):
        return acceptable[:top_k]
    if len(acceptable) >= top_k:
        return acceptable[:top_k]
    return (acceptable + severe_off_target)[:top_k]


def explain_protocol_match(profile: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    signals = result.get("profile_signals") or _profile_match_signals(profile, result)
    why = []
    if result.get("why"):
        why.append(result["why"])
    if signals["organism_match"]:
        why.append(f"matches organism/system: {profile.get('organism')}")
    if signals["method_match"]:
        why.append(f"matches method: {profile.get('experimental_method')}")
    if signals["readout_match"]:
        why.append(f"matches readout: {profile.get('readout_assay')}")
    if signals["expression_match"]:
        why.append(f"matches expression type: {profile.get('expression_type')}")
    if signals.get("tissue_match"):
        why.append(f"matches tissue/cell type: {profile.get('tissue_or_cell_type')}")
    if signals.get("combined_match"):
        why.append("matches multiple required profile fields together")

    assumptions = []
    missing = []
    for field in profile_missing_fields(profile):
        if field in {"gene_or_construct", "timeline", "protocol_difficulty"}:
            missing.append(_human_field(field))
        elif field == "required_equipment":
            continue

    may_not_fit = []
    text = _result_text(result)
    expression = str(profile.get("expression_type") or "").lower()
    if expression and expression not in text and not signals["expression_match"]:
        may_not_fit.append(f"The result does not clearly state {profile.get('expression_type')}.")
    readout = str(profile.get("readout_assay") or "").lower()
    if readout and not signals["readout_match"]:
        may_not_fit.append(f"The result may not cover the requested readout: {profile.get('readout_assay')}.")
    tissue = str(profile.get("tissue_or_cell_type") or "").lower()
    if tissue and not _is_generic(tissue) and not signals.get("tissue_match"):
        may_not_fit.append(f"The result may not use the requested tissue/cell type: {profile.get('tissue_or_cell_type')}.")
    organism = profile.get("organism")
    if organism and not _is_generic(str(organism)) and not signals.get("organism_match"):
        may_not_fit.append(f"The result does not clearly match the requested organism/system: {organism}.")
    if signals.get("protoplast_mismatch"):
        may_not_fit.append("Uses protoplasts, while the profile asks for leaf tissue.")
    if signals.get("tev_false_positive"):
        may_not_fit.append("Looks like TEV/protein-processing rather than plant transient expression.")
    if signals.get("generic_protein_purification"):
        may_not_fit.append("Looks like a generic protein expression/purification protocol, not plant transient overexpression.")
    if signals.get("stable_off_target"):
        may_not_fit.append("Looks like a maize-only assay or profiling protocol, not stable transformation for overexpression.")
    if signals.get("rna_qpcr_off_target"):
        may_not_fit.append("Looks off-target for RNA/qPCR overexpression readout, such as protein purification, footprinting, metabolomics, or generic sequencing prep.")
    if _missing_or_generic(profile.get("organism")):
        assumptions.append("Assumes the protocol can be adapted to your exact organism.")

    return {
        "why_it_matches": "; ".join(_dedup(why)) or "Matched by protocol title, description, or keyword overlap.",
        "assumptions": assumptions,
        "may_not_fit": may_not_fit,
        "missing_information": missing[:4],
    }


def _extract_organism(q: str) -> Optional[str]:
    matches = []
    for key in sorted(_ORGANISMS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", q):
            matches.append(_ORGANISMS[key])
    if not matches:
        return None

    specific = [value for value in matches if not _is_generic(value)]
    return specific[0] if specific else matches[0]


def _has_multiplex_gene_modification_goal(text: Any) -> bool:
    q = str(text or "").lower()
    has_multiplex = re.search(r"\b(more than one|multiple|multiplex|several|simultaneously|at the same time)\b", q)
    has_target = re.search(r"\b(genes?|targets?|transcription factors?|loci|constructs?)\b", q)
    has_modification = re.search(r"\b(modify|modified|modification|edit|editing|alter|altered|knock(?: |-)?out|knock(?: |-)?down|overexpress)\b", q)
    return bool(has_multiplex and has_target and has_modification)


def _has_stress_tolerance_goal(text: Any) -> bool:
    q = str(text or "").lower()
    has_stress = re.search(r"\b(drought|salt|salinity|heat|cold|osmotic|stress)\b", q)
    has_assay = re.search(r"\b(tolerance|resistance|response|assay|test|testing|phenotype|phenotyping)\b", q)
    return bool(has_stress and has_assay)


def _has_knockdown_goal(text: Any) -> bool:
    return bool(re.search(r"\b(knockdown|knock down|silenc(?:e|ing)|sirna|shrna|rnai|vigs)\b", str(text or ""), flags=re.IGNORECASE))


def _has_explicit_genome_editing_goal(text: Any) -> bool:
    return bool(re.search(r"\b(crispr|cas9|cas12|base editing|prime editing|grna|guide rna)\b", str(text or ""), flags=re.IGNORECASE))


def _has_ambiguous_gene_modification_goal(text: Any) -> bool:
    return bool(re.search(
        r"\b(gene modification|modify genes?|modified genes?|genes? (?:are |is |being )?modified|gene editing|genome editing)\b",
        str(text or ""),
        flags=re.IGNORECASE,
    ))


def _ambiguous_modification_without_specific_type(text: Any) -> bool:
    q = str(text or "").lower()
    if not _has_ambiguous_gene_modification_goal(q):
        return False
    return not (
        _has_overexpression_goal(q)
        or _has_knockdown_goal(q)
        or _has_explicit_genome_editing_goal(q)
        or re.search(r"\b(mutagenesis|mutagenize|mutation|knockout|knock out)\b", q)
    )


def _extract_target(query: str) -> Optional[str]:
    q = str(query or "").lower()
    if re.search(r"\btranscription factors?\b", q):
        return "transcription factors"
    if re.search(r"\bgene family\b", q):
        return "gene family"
    patterns = [
        r"\bmore than one\s+([a-z][a-z\s-]{2,40}?)\s+(?:to be |being )?(?:modified|edited|altered)\b",
        r"\bmultiple\s+([a-z][a-z\s-]{2,40}?)\s+(?:to be |being )?(?:modified|edited|altered)\b",
        r"\btarget(?:ing)?\s+([A-Za-z0-9_.-]{2,})\b",
    ]
    reject = {"gene", "genes", "target", "targets", "protocol", "modified", "editing"}
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        value = " ".join(match.group(1).split()).strip(" .,:;")
        if value.lower() not in reject:
            return value
    return None


def _extract_stress_type(q: str) -> Optional[str]:
    for label, pattern in (
        ("drought", r"\b(drought|water deficit|withholding water)\b"),
        ("salt", r"\b(salt|salinity|nacl)\b"),
        ("heat", r"\b(heat|high temperature)\b"),
        ("cold", r"\b(cold|chilling|freezing)\b"),
        ("osmotic", r"\b(osmotic|peg)\b"),
    ):
        if re.search(pattern, q):
            return label
    if re.search(r"\bstress\b", q):
        return "stress"
    return None


def _extract_growth_stage(q: str) -> Optional[str]:
    for label, pattern in (
        ("seedlings", r"\b(seedling|seedlings)\b"),
        ("adult plants", r"\b(adult plant|adult plants|mature plant|mature plants)\b"),
        ("germination", r"\b(germination|germinating seeds?)\b"),
        ("leaf stage", r"\b(leaf stage|leaves|leaf tissue)\b"),
    ):
        if re.search(pattern, q):
            return label
    return None


def _extract_treatment_condition(q: str) -> Optional[str]:
    if re.search(r"\bwithholding water\b", q):
        return "withholding water"
    if re.search(r"\bpeg\b", q):
        return "PEG/osmotic stress"
    stress_type = _extract_stress_type(q)
    if stress_type and stress_type != "stress":
        return f"{stress_type} stress"
    return None


def _extract_protein_specifics(q: str) -> Dict[str, Any]:
    specific: Dict[str, Any] = {}
    if re.search(r"\be\.?\s*coli|escherichia coli\b", q):
        specific["expression_host"] = "E. coli"
    elif re.search(r"\byeast\b", q):
        specific["expression_host"] = "yeast"
    elif re.search(r"\bmammalian cells?\b", q):
        specific["expression_host"] = "mammalian cells"
    elif re.search(r"\bplant tissue|leaf|leaves\b", q):
        specific["protein_source"] = "plant tissue"

    if re.search(r"\bhis[- ]?tag|his6\b", q):
        specific["tag"] = "His-tag"
        specific["purification_method"] = "His-tag affinity"
    elif re.search(r"\bgst[- ]?tag|gst\b", q):
        specific["tag"] = "GST-tag"
        specific["purification_method"] = "GST affinity"
    elif re.search(r"\baffinity purification\b", q):
        specific["purification_method"] = "affinity purification"
    return specific


def _extract_freeform_organism_answer(query: str) -> Optional[str]:
    """
    Accept short typed answers to the organism clarification question.

    This keeps the organism field open-ended: the quick options are examples,
    not a closed vocabulary.
    """
    lines = [line.strip() for line in str(query or "").splitlines() if line.strip()]
    if not lines:
        return None
    answer = re.sub(r"[?!.,;:]+$", "", lines[-1]).strip()
    if not answer or len(answer) > 48:
        return None

    lowered = answer.lower()
    if lowered in _GENERIC_ORGANISMS:
        return None

    known = _extract_organism(lowered)
    if known and not _is_generic(known):
        return known

    if len(answer.split()) > 4:
        return None

    reject_patterns = [
        r"\b(how|what|which|should|does|do|can|want|happen|happens)\b",
        r"\b(overexpress|overexpression|gene|construct|plasmid)\b",
        r"\b(stable|transient|transformation|expression|transgenic)\b",
        r"\b(phenotype|protein|western|immunoblot|rna|qpcr|pcr|localization|microscopy)\b",
        r"\b(whole plant|leaf|leaves|tissue|protoplast|callus|cell culture)\b",
        r"\b(either|not sure|unsure|unknown|protocol|search)\b",
    ]
    if any(re.search(pattern, lowered) for pattern in reject_patterns):
        return None
    if detect_experiment_intent(answer).get("intent") != "unknown":
        return None
    if not re.fullmatch(r"[A-Za-z][A-Za-z .'-]*", answer):
        return None

    cleaned = re.sub(r"\b(plants?|seedlings?)\b", "", answer, flags=re.IGNORECASE)
    cleaned = " ".join(cleaned.split())
    return cleaned or None


def _extract_gene_or_construct(query: str) -> Optional[str]:
    patterns = [
        r"\b(?:gene|construct|plasmid)\s+([A-Za-z0-9_.-]{2,})\b",
        r"\b([A-Za-z0-9_.-]{2,})\s+(?:gene|construct|plasmid)\b",
    ]
    generic = {
        "a", "an", "the", "target", "reporter", "gene", "construct", "plasmid",
        "in", "on", "for", "from", "with", "and", "or", "of", "to", "what",
        "happens", "overexpressing", "overexpress", "overexpression",
        "expression", "transgene", "transgenic", "transformation",
        "stable", "transient", "plant", "protocol", "method",
    }
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if value.lower() not in generic:
                return value
    return None


def _extract_timeline(query: str) -> Optional[str]:
    match = re.search(r"\b(\d+\s*(?:hour|hours|day|days|week|weeks|month|months))\b", query, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    q = query.lower()
    if re.search(r"\b(quick|fast|rapid|same day|overnight)\b", q):
        return "quick / short timeline"
    return None


def _extract_difficulty(q: str) -> Optional[str]:
    if re.search(r"\b(simple|easy|beginner|low cost|cheap)\b", q):
        return "simple"
    if re.search(r"\b(advanced|high throughput|high-throughput|complex)\b", q):
        return "advanced"
    return None


def _extract_constraints(query: str) -> List[str]:
    q = query.lower()
    constraints = []
    patterns = [
        (r"\b(no|without|avoid)\s+([^,.]+)", "avoid {}"),
        (r"\bonly\s+([^,.]+)", "only {}"),
        (r"\bdo not\s+([^,.]+)", "avoid {}"),
    ]
    for pattern, template in patterns:
        for match in re.finditer(pattern, q):
            value = " ".join(match.group(2).split()[:5])
            if value:
                constraints.append(template.format(value))
    return constraints


def _first_pattern_match(q: str, patterns: List[tuple]) -> Optional[str]:
    for label, pattern in patterns:
        if re.search(pattern, q):
            return label
    return None


def _extract_many(q: str, patterns: List[tuple]) -> List[str]:
    return [label for label, pattern in patterns if re.search(pattern, q)]


def _merge_list(existing: Any, incoming: List[str]) -> List[str]:
    base = existing if isinstance(existing, list) else []
    return _dedup([str(x) for x in base + incoming if x])


def _merge_dict(existing: Any, incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing) if isinstance(existing, dict) else {}
    for key, value in incoming.items():
        if _emptyish(value):
            continue
        clean_key = str(key).strip()
        if not clean_key:
            continue
        if isinstance(value, list):
            merged[clean_key] = [str(v) for v in value if not _emptyish(v)]
        elif isinstance(value, dict):
            merged[clean_key] = _merge_dict(merged.get(clean_key), value)
        else:
            merged[clean_key] = value if isinstance(value, bool) else str(value).strip()
    return merged


def _sync_intent_specific_fields(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep intent-specific source-of-truth fields visible at top level.

    The UI and older ranking code read top-level fields, while the newer profile
    model stores intent-specific values in intent_specific. This mirrors known
    intent-specific fields both ways, with intent_specific winning when both are
    populated.
    """
    if not isinstance(profile, dict):
        return profile

    intent_specific = profile.get("intent_specific")
    if not isinstance(intent_specific, dict):
        intent_specific = {}

    for field in ("modification_type", "delivery_method"):
        specific_value = intent_specific.get(field)
        top_value = profile.get(field)
        if not _emptyish(specific_value):
            profile[field] = str(specific_value).strip()
        elif not _emptyish(top_value):
            intent_specific[field] = str(top_value).strip()

    delivery_mode = intent_specific.get("delivery_mode")
    if _emptyish(profile.get("delivery_method")) and not _emptyish(delivery_mode):
        profile["delivery_method"] = str(delivery_mode).strip()

    if _emptyish(profile.get("readout")) and not _emptyish(profile.get("readout_assay")):
        profile["readout"] = profile["readout_assay"]
    if _emptyish(profile.get("readout_assay")) and not _emptyish(profile.get("readout")):
        profile["readout_assay"] = profile["readout"]

    if _emptyish(profile.get("sample_type")) and not _emptyish(profile.get("tissue_or_cell_type")):
        profile["sample_type"] = profile["tissue_or_cell_type"]
    if _emptyish(profile.get("tissue_or_cell_type")) and not _emptyish(profile.get("sample_type")):
        profile["tissue_or_cell_type"] = profile["sample_type"]

    if _emptyish(profile.get("target")) and not _emptyish(profile.get("gene_or_construct")):
        profile["target"] = profile["gene_or_construct"]

    if _emptyish(profile.get("difficulty")) and not _emptyish(profile.get("protocol_difficulty")):
        profile["difficulty"] = profile["protocol_difficulty"]
    if _emptyish(profile.get("protocol_difficulty")) and not _emptyish(profile.get("difficulty")):
        profile["protocol_difficulty"] = profile["difficulty"]

    equipment = _merge_list(profile.get("equipment"), profile.get("required_equipment") or [])
    if equipment:
        profile["equipment"] = equipment
        profile["required_equipment"] = _merge_list(profile.get("required_equipment"), equipment)

    delivery = str(profile.get("delivery_method") or "").strip().lower()
    expression = str(profile.get("expression_type") or "").strip().lower()
    if _emptyish(profile.get("expression_type")) and delivery in {"transient expression", "stable transformation"}:
        profile["expression_type"] = profile["delivery_method"]
    if _emptyish(profile.get("delivery_method")) and expression in {"transient expression", "stable transformation"}:
        profile["delivery_method"] = profile["expression_type"]

    profile["intent_specific"] = intent_specific
    return profile


def _profile_value(profile: Dict[str, Any], field: str) -> Any:
    if not isinstance(profile, dict):
        return None
    alias_fields = {
        "tissue_or_cell_type": ["tissue_or_cell_type", "sample_type", "system"],
        "sample_type": ["sample_type", "tissue_or_cell_type", "system"],
        "system": ["tissue_or_cell_type", "sample_type", "system"],
        "readout_assay": ["readout_assay", "readout"],
        "readout": ["readout", "readout_assay"],
        "target": ["target", "gene_or_construct"],
        "gene_or_construct": ["gene_or_construct", "target"],
        "equipment": ["equipment", "required_equipment"],
        "required_equipment": ["required_equipment", "equipment"],
        "difficulty": ["difficulty", "protocol_difficulty"],
        "protocol_difficulty": ["protocol_difficulty", "difficulty"],
    }
    intent_specific = profile.get("intent_specific") if isinstance(profile.get("intent_specific"), dict) else {}
    specific_value = intent_specific.get(field)
    if not _emptyish(specific_value):
        return specific_value
    if field == "delivery_method":
        value = profile.get("delivery_method")
        if not _emptyish(value):
            return value
        expression = profile.get("expression_type")
        if not _emptyish(expression) and str(expression).strip().lower() in {"transient expression", "stable transformation"}:
            return expression
        return None
    if field == "expression_type":
        value = profile.get("expression_type")
        if not _emptyish(value):
            return value
        delivery = profile.get("delivery_method")
        if not _emptyish(delivery) and str(delivery).strip().lower() in {"transient expression", "stable transformation"}:
            return delivery
        return None
    for candidate_field in alias_fields.get(field, [field]):
        value = profile.get(candidate_field)
        if not _emptyish(value):
            return value
    return None


def needs_clarification(profile: Dict[str, Any], field: str, context: Optional[Dict[str, Any]] = None) -> bool:
    """
    True only when a structured profile field still needs user input.

    This centralizes the clarification gate so values from previous turns,
    aliases, and intent_specific are treated consistently.
    """
    synced = _sync_intent_specific_fields(dict(profile or {}))
    canonical = _canonical_profile_field(field)
    # User explicitly said "not sure" for this field — never re-ask it, even if
    # it's required. The skip list is maintained by the chat endpoint.
    skipped = synced.get("_skipped_fields") or []
    if canonical in skipped or field in skipped:
        return False
    value = _profile_value(synced, canonical)

    if _field_has_conflict(synced, canonical, context=context):
        return True
    if _emptyish(value):
        return True
    if _is_uncertain_value(value):
        return True
    if canonical == "organism":
        return _missing_or_generic(value)
    if canonical in {"tissue_or_cell_type", "sample_type", "system"}:
        return _is_generic_system_value(value)
    if canonical in {"target", "gene_or_construct"}:
        return _is_generic_target_value(value)
    return False


def _canonical_profile_field(field: str) -> str:
    aliases = {
        "system": "tissue_or_cell_type",
        "sample_type": "tissue_or_cell_type",
        "readout": "readout_assay",
        "gene_or_construct": "target",
        "required_equipment": "equipment",
        "protocol_difficulty": "difficulty",
    }
    return aliases.get(field, field)


def _is_uncertain_value(value: Any) -> bool:
    if isinstance(value, list):
        return not value or all(_is_uncertain_value(item) for item in value)
    text = str(value or "").strip().lower()
    return text in {
        "",
        "none",
        "null",
        "unknown",
        "not specified",
        "n/a",
        "not sure",
        "unsure",
        "not sure / flexible",
        "flexible",
        "either / not sure",
    }


def _is_generic_system_value(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"tissue", "tissues", "cells", "sample", "samples", "system", "organism"}


def _is_generic_target_value(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"gene", "genes", "target", "targets", "construct", "constructs", "not sure"}


def _field_has_conflict(profile: Dict[str, Any], field: str, context: Optional[Dict[str, Any]] = None) -> bool:
    if field not in {"tissue_or_cell_type", "sample_type", "system"}:
        return False

    system_value = _profile_value(profile, "tissue_or_cell_type")
    if _emptyish(system_value):
        return False

    organism_class = classify_organism_system({"organism": profile.get("organism")})
    system_class = classify_organism_system({"tissue_or_cell_type": system_value})
    if organism_class == "unknown" or system_class == "unknown":
        return False
    if organism_class == system_class:
        return False

    # Mammalian-cell systems are compatible with mouse/animal organisms.
    compatible = {
        ("mouse_animal", "mammalian_cells"),
        ("mammalian_cells", "mouse_animal"),
    }
    return (organism_class, system_class) not in compatible


def _clear_incompatible_profile_fields(profile: Dict[str, Any]) -> Dict[str, Any]:
    if _field_has_conflict(profile, "tissue_or_cell_type"):
        profile["tissue_or_cell_type"] = None
        profile["sample_type"] = None
    return profile


def _emptyish(value: Any) -> bool:
    if value in (None, "", []):
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null", "unknown", "not specified", "n/a", "not sure", "unsure"}
    return False


def _query_variants(value: Any, mapping: Dict[str, List[str]]) -> List[str]:
    if _emptyish(value):
        return []
    text = str(value).strip()
    if _is_generic(text):
        return []
    if text in mapping:
        return mapping[text]
    lower_map = {k.lower(): v for k, v in mapping.items()}
    return lower_map.get(text.lower(), [text])


def _first(values: List[str]) -> Optional[str]:
    return values[0] if values else None


def _join_terms(values: List[Any]) -> str:
    return " ".join(str(v).strip() for v in values if not _emptyish(v))


def _dedup(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _missing_or_generic(value: Any) -> bool:
    if value in (None, "", []):
        return True
    return _is_generic(str(value))


def _profile_has_search_concepts(profile: Dict[str, Any]) -> bool:
    if not isinstance(profile, dict):
        return False
    organism = profile.get("organism")
    if not _emptyish(organism) and not _is_generic(str(organism)):
        return True
    for field in (
        "method",
        "experimental_method",
        "sub_intent",
        "target",
        "modification_type",
        "delivery_method",
        "expression_type",
        "sample_type",
        "tissue_or_cell_type",
        "readout",
        "readout_assay",
        "condition",
        "gene_or_construct",
    ):
        if not _emptyish(profile.get(field)):
            return True
    intent_specific = profile.get("intent_specific") if isinstance(profile.get("intent_specific"), dict) else {}
    return any(not _emptyish(value) for value in intent_specific.values() if not isinstance(value, bool))


def _is_generic(value: str) -> bool:
    return value.strip().lower() in _GENERIC_ORGANISMS


def _has_overexpression_goal(text: Any) -> bool:
    return bool(_OVEREXPRESSION_GOAL_RE.search(str(text or "")))


def _sub_intent_from_modification_type(value: Any) -> Optional[str]:
    if _emptyish(value):
        return None
    text = str(value).strip().lower()
    if "overexpression" in text or "overexpress" in text:
        return "gene_overexpression"
    if "knockdown" in text or "silencing" in text or "rnai" in text:
        return "gene_knockdown"
    if "crispr" in text or "genome editing" in text or "gene editing" in text:
        return "genome_editing"
    return None


def _is_gene_modification_method(value: Any) -> bool:
    if _emptyish(value):
        return False
    text = str(value).strip().lower()
    return "gene modification" in text or "gene editing" in text or "genome editing" in text


def _is_gene_overexpression_method(value: Any) -> bool:
    if _emptyish(value):
        return False
    text = str(value).strip().lower()
    return "overexpression" in text or "overexpress" in text or "transgene expression" in text


def _is_stable_transformation(value: Any) -> bool:
    if _emptyish(value):
        return False
    text = str(value).strip().lower()
    return "stable transformation" in text or "stable" == text or "transgenic" in text


def _is_transient_expression(value: Any) -> bool:
    if _emptyish(value):
        return False
    text = str(value).strip().lower()
    return "transient expression" in text or "transient transformation" in text or "transient" == text or "agroinfiltration" in text


def _is_rna_qpcr_readout(value: Any) -> bool:
    if _emptyish(value):
        return False
    text = str(value).strip().lower()
    return any(term in text for term in ("rna", "qpcr", "q-pcr", "rt-qpcr", "transcript", "gene expression"))


def _has_delivery_context(value: Any) -> bool:
    if _emptyish(value):
        return False
    text = str(value).strip().lower()
    return any(term in text for term in ("stable transformation", "transient", "agrobacterium", "biolistic", "gene gun", "transfection"))


def _requires_strict_required_concepts(profile: Dict[str, Any]) -> bool:
    return (
        _is_gene_overexpression_method(profile.get("experimental_method"))
        and _is_stable_transformation(profile.get("expression_type"))
        and not _missing_or_generic(profile.get("organism"))
    )


def _is_strict_overexpression_readout_profile(profile: Dict[str, Any]) -> bool:
    return (
        _is_gene_overexpression_method(profile.get("experimental_method"))
        and not _missing_or_generic(profile.get("organism"))
        and (
            _is_transient_expression(profile.get("expression_type"))
            or _is_rna_qpcr_readout(profile.get("readout_assay") or profile.get("readout"))
        )
    )


def _result_text(result: Dict[str, Any]) -> str:
    parts = [
        result.get("title") or "",
        result.get("description") or "",
        " ".join(str(k) for k in result.get("keywords") or []),
        result.get("materials_text") or "",
        " ".join(str(s) for s in result.get("steps_preview") or []),
    ]
    return " ".join(parts).lower()


def _has_required_concept_groups(profile: Dict[str, Any]) -> bool:
    return len(_required_concept_groups(profile)) >= 2


def _required_concept_coverage(profile: Dict[str, Any], text: str) -> float:
    groups = _required_concept_groups(profile)
    if not groups:
        return 0.0
    hits = 0
    for terms in groups:
        if _has_any_term(text, terms):
            hits += 1
    return hits / len(groups)


def _required_concept_groups(profile: Dict[str, Any]) -> List[List[str]]:
    groups: List[List[str]] = []
    organism_terms = _query_variants(profile.get("organism"), _ORGANISM_SEARCH_VARIANTS)
    if organism_terms:
        groups.append(organism_terms)

    sub_intent = normalize_sub_intent(profile.get("sub_intent") or profile.get("experimental_method"))
    method = profile.get("experimental_method") or profile.get("method")
    modification = profile.get("modification_type")
    expression = profile.get("expression_type")
    tissue = profile.get("tissue_or_cell_type") or profile.get("sample_type")
    readout = profile.get("readout_assay") or profile.get("readout")
    target = profile.get("target") or profile.get("gene_or_construct")
    intent_specific = profile.get("intent_specific") if isinstance(profile.get("intent_specific"), dict) else {}

    if sub_intent == "stress_tolerance_assay":
        stress_type = intent_specific.get("stress_type") or _extract_stress_type(str(profile.get("condition") or "").lower())
        if stress_type:
            groups.append([str(stress_type), f"{stress_type} stress"])
        groups.append(["tolerance", "phenotyping", "phenotype", "stress assay"])
        return groups

    if sub_intent in {"gene_modification", "multiplex_gene_modification"}:
        modification_terms = _query_variants(modification, _MODIFICATION_SEARCH_VARIANTS)
        if modification_terms:
            groups.append(modification_terms)
        if target and not _is_generic(str(target)):
            groups.append([str(target)])
        if tissue and not _is_generic(str(tissue)):
            groups.append(_tissue_terms_for_matching(str(tissue)))
        if profile.get("delivery_method"):
            groups.append(_query_variants(profile.get("delivery_method"), _DELIVERY_SEARCH_VARIANTS))
        return [group for group in groups if group]

    if sub_intent == "gene_overexpression" or _is_gene_overexpression_method(method):
        groups.append(["overexpression", "overexpress", "transgene expression"])
        expression_terms = _query_variants(expression, _EXPRESSION_SEARCH_VARIANTS)
        if expression_terms:
            groups.append(expression_terms)
        if tissue and not _is_generic(str(tissue)):
            groups.append(_tissue_terms_for_matching(str(tissue)))
        if readout:
            groups.append(_query_variants(readout, _READOUT_SEARCH_VARIANTS) or [str(readout)])
        return [group for group in groups if group]

    if sub_intent in {"protein_purification", "protein_extraction", "western_blot", "protein_detection"}:
        groups.append(["protein"])
        groups.append(_query_variants(method, _METHOD_SEARCH_VARIANTS) or [str(method)])
        if target:
            groups.append([str(target)])
        return [group for group in groups if group]

    method_terms = _query_variants(method, _METHOD_SEARCH_VARIANTS)
    if method_terms:
        groups.append(method_terms)
    if tissue and not _is_generic(str(tissue)):
        groups.append(_tissue_terms_for_matching(str(tissue)))
    if readout:
        groups.append(_query_variants(readout, _READOUT_SEARCH_VARIANTS) or [str(readout)])
    return [group for group in groups if group]


def _tissue_terms_for_matching(value: str) -> List[str]:
    raw = str(value or "").lower()
    if raw == "in planta / whole plant":
        return ["in planta", "whole plant", "whole organism"]
    if raw in {"leaf", "leaf tissue", "leaves"}:
        return ["leaf", "leaves", "leaf tissue"]
    return [value]


def _profile_match_signals(profile: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, float]:
    text = _result_text(result)
    title = str(result.get("title") or "").lower()

    # Per-field match scores are driven by the declarative schema: each field's
    # matcher + which profile key it reads is declared in profile_schema.FIELD_SIGNALS.
    _fs = profile_schema.field_scores(profile, text)
    organism_match = _fs["organism_match"]
    method_match = _fs["method_match"]
    readout_match = _fs["readout_match"]
    expression_match = _fs["expression_match"]
    tissue_match = _fs["tissue_match"]
    protoplast_mismatch = _protoplast_leaf_mismatch(profile, text, title)
    if protoplast_mismatch:
        tissue_match = min(tissue_match, 0.45)

    title_context = _title_context_match_score(profile, title)
    coverage_scores = [
        score for value, score in (
            (profile.get("organism"), organism_match),
            (profile.get("experimental_method"), method_match),
            (profile.get("expression_type"), expression_match),
            (profile.get("tissue_or_cell_type"), tissue_match),
            (profile.get("readout_assay"), readout_match),
        )
        if not _emptyish(value) and not _is_generic(str(value))
    ]
    profile_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0
    combined_match = _combined_match_score(
        organism_match=organism_match,
        method_match=method_match,
        expression_match=expression_match,
        tissue_match=tissue_match,
        readout_match=readout_match,
    )
    # Gate the "matches multiple required fields together" bonus on organism.
    # When the user explicitly named an organism but this result doesn't match it,
    # a broad method/expression/tissue match must not mask the wrong organism —
    # organism is a required field here, so the combined bonus isn't earned.
    organism_specified = not _emptyish(profile.get("organism")) and not _is_generic(str(profile.get("organism")))
    if organism_specified and organism_match < rank_cfg.T_STRONG_MATCH:
        combined_match = 0.0
    required_concept_match = _required_concept_coverage(profile, text)

    step_count = int(result.get("step_count") or len(result.get("steps_preview") or []))
    completeness = 0.0
    if result.get("description"):
        completeness += 0.35
    if step_count:
        completeness += 0.35
    if result.get("doi") or result.get("url") or result.get("uri"):
        completeness += 0.3

    published = result.get("published_on") or result.get("updated_on") or result.get("created_on")
    recency = _recency_score(published)
    community = min(1.0, len(result.get("matched_probes") or result.get("matched_queries") or []) / 4.0)
    false_positive_penalty, tev_false_positive, generic_protein_purification, stable_off_target, rna_qpcr_off_target = _false_positive_penalty(
        profile=profile,
        text=text,
        organism_match=organism_match,
        expression_match=expression_match,
        tissue_match=tissue_match,
        method_match=method_match,
        protoplast_mismatch=protoplast_mismatch,
    )

    return {
        "organism_match": round(organism_match, 2),
        "method_match": round(method_match, 2),
        "readout_match": round(readout_match, 2),
        "expression_match": round(expression_match, 2),
        "tissue_match": round(tissue_match, 2),
        "combined_match": round(combined_match, 2),
        "profile_coverage": round(profile_coverage, 2),
        "required_concept_match": round(required_concept_match, 2),
        "title_context_match": round(title_context, 2),
        "completeness": round(min(1.0, completeness), 2),
        "recency": recency,
        "community": round(community, 2),
        "false_positive_penalty": round(false_positive_penalty, 2),
        "tev_false_positive": 1.0 if tev_false_positive else 0.0,
        "generic_protein_purification": 1.0 if generic_protein_purification else 0.0,
        "stable_off_target": 1.0 if stable_off_target else 0.0,
        "rna_qpcr_off_target": 1.0 if rna_qpcr_off_target else 0.0,
        "protoplast_mismatch": 1.0 if protoplast_mismatch else 0.0,
    }


def _organism_match_score(value: Any, text: str) -> float:
    if _emptyish(value) or _is_generic(str(value)):
        return 0.0
    # Multi-organism operator values ("rice or potato", "rice, tomato") must match
    # a result that mentions ANY of the organisms — otherwise the whole phrase is
    # treated as one literal and never matches a single-organism title. Split into
    # operands, score each via the per-organism logic below, and take the best.
    op, operands, _ = detect_operator(str(value))
    if op in {"OR", "AND"} and len(operands) >= 2:
        return max(_organism_match_score(operand, text) for operand in operands)
    raw = str(value).strip().lower()
    plant_context = _has_any_term(text, _PLANT_CONTEXT_TERMS)

    if raw == "nicotiana benthamiana":
        if _has_any_term(text, ["nicotiana benthamiana", "n benthamiana", "n. benthamiana"]):
            return 1.0
        if _has_any_term(text, ["tobacco"]):
            if _has_any_term(text, ["tobacco etch virus", "tev protease"]):
                return 0.0
            return 0.85 if plant_context else 0.4
        return 0.0
    if raw == "tobacco":
        if _has_any_term(text, ["tobacco etch virus", "tev protease"]):
            return 0.0
        return 1.0 if _has_any_term(text, ["tobacco"]) and plant_context else 0.0
    if raw == "arabidopsis thaliana":
        return 1.0 if _has_any_term(text, ["arabidopsis thaliana", "arabidopsis"]) else 0.0
    if raw == "oryza sativa":
        return 1.0 if _has_any_term(text, ["oryza sativa", "rice"]) else 0.0
    if raw == "rice":
        return 1.0 if _has_any_term(text, ["rice", "oryza sativa"]) else 0.0
    if raw == "maize":
        return 1.0 if _has_any_term(text, ["maize", "zea mays", "corn"]) else 0.0
    if raw == "e. coli":
        return 1.0 if _has_any_term(text, ["e. coli", "escherichia coli"]) else 0.0
    return 1.0 if _has_any_term(text, [raw]) else 0.0


def _expression_match_score(value: Any, text: str) -> float:
    if _emptyish(value):
        return 0.0
    raw = str(value).strip().lower()
    if raw == "transient expression":
        if _has_any_term(text, [
            "transient expression",
            "transient overexpression",
            "transient transformation",
            "agrobacterium-mediated transient",
            "agrobacterium mediated transient",
            "agrobacterium-mediated expression",
            "agrobacterium mediated expression",
            "agroinfiltration",
            "agro-infiltration",
        ]):
            return 1.0
        if _has_any_term(text, ["transient"]) and _has_any_term(text, ["expression", "transfection", "transformation"]):
            return 0.8
        if _has_any_term(text, ["protoplast", "protoplasts"]) and _has_any_term(text, ["transfection", "transfect", "expression", "fluorescent protein"]):
            return 0.75
        return 0.0
    if raw == "stable transformation":
        if _has_any_term(text, ["stable transformation", "stable transgenic", "transgenic plant", "floral dip"]):
            return 1.0
        delivery_score = _stable_transformation_delivery_score(text)
        if delivery_score:
            return delivery_score
        if _has_any_term(text, ["stable", "transgenic"]) and _has_any_term(text, ["transformation", "expression"]):
            return 0.75
        return 0.0
    if raw == "stable or transient expression":
        return max(
            _expression_match_score("stable transformation", text),
            _expression_match_score("transient expression", text),
        )
    if raw == "tissue-specific expression":
        return 1.0 if _has_any_term(text, ["tissue-specific expression", "tissue specific expression", "cell-specific expression"]) else 0.0
    if raw == "inducible expression":
        return 1.0 if _has_any_term(text, ["inducible expression", "induced expression", "dexamethasone", "estradiol"]) else 0.0
    if raw == "constitutive expression":
        return 1.0 if _has_any_term(text, ["constitutive expression", "35s promoter", "camv 35s", "ubiquitin promoter"]) else 0.0
    return 1.0 if _has_any_term(text, [raw]) else 0.0


def _method_match_score(value: Any, text: str) -> float:
    if _emptyish(value) or _is_generic(str(value)):
        return 0.0
    raw = str(value).strip().lower()
    if raw == "gene overexpression":
        if _has_any_term(text, [
            "gene overexpression",
            "overexpression",
            "overexpress",
            "over-express",
            "transgene expression",
            "transient expression",
            "agrobacterium-mediated expression",
            "agrobacterium mediated expression",
            "agroinfiltration",
            "agro-infiltration",
            "transgenic overexpression",
        ]):
            return 1.0
        if _has_any_term(text, ["transgene expression", "transgenic plant", "transgenic plants", "transgenic line", "transgenic lines", "heterologous gene expression"]):
            return 0.85
        delivery_score = _stable_transformation_delivery_score(text)
        if delivery_score >= 0.75:
            return 0.8
        if _has_any_term(text, ["transfection", "transfect"]) and _has_any_term(text, ["plasmid", "gfp", "fluorescent protein", "reporter"]):
            return 0.6
        return 0.0
    if raw in {"gene modification", "multiplex gene modification"}:
        if _has_any_term(text, ["gene modification", "genome modification", "gene editing", "genome editing", "modified genes"]):
            return 1.0
        if _has_any_term(text, ["crispr", "cas9", "knockdown", "overexpression", "mutagenesis", "mutation"]):
            return 0.8
        return 0.0
    if raw == "genome editing":
        return 1.0 if _has_any_term(text, ["crispr", "cas9", "genome editing", "gene editing", "base editing", "prime editing"]) else 0.0
    if raw == "gene knockdown":
        return 1.0 if _has_any_term(text, ["gene knockdown", "knockdown", "rnai", "sirna", "gene silencing"]) else 0.0
    if raw == "protein purification":
        return 1.0 if _has_any_term(text, ["protein purification", "recombinant protein purification", "affinity purification"]) else 0.0
    if raw == "protein extraction":
        return 1.0 if _has_any_term(text, ["protein extraction", "extract protein", "total protein"]) else 0.0
    if raw == "western blot":
        return 1.0 if _has_any_term(text, ["western blot", "western blotting", "immunoblot"]) else 0.0
    if raw == "protein detection":
        return 1.0 if _has_any_term(text, ["protein detection", "western blot", "immunoblot", "elisa"]) else 0.0
    if raw == "pcr/qpcr":
        return 1.0 if _has_any_term(text, ["pcr", "qpcr", "q-pcr", "rt-qpcr", "real-time pcr"]) else 0.0
    if raw == "transformation":
        return 1.0 if _has_any_term(text, ["transformation", "transfection", "agrobacterium", "floral dip", "electroporation"]) else 0.0
    if raw == "microscopy":
        return 1.0 if _has_any_term(text, ["microscopy", "microscope", "imaging", "confocal", "fluorescence imaging"]) else 0.0
    if raw == "sequencing prep":
        return 1.0 if _has_any_term(text, ["sequencing", "rna-seq", "library prep", "library preparation", "ngs"]) else 0.0
    if raw == "stress tolerance assay":
        if _has_any_term(text, ["drought tolerance", "stress tolerance", "stress assay", "phenotyping", "phenotype"]):
            return 1.0
        if _has_any_term(text, ["drought", "salt", "salinity", "heat stress", "cold stress"]) and _has_any_term(text, ["tolerance", "response", "assay"]):
            return 0.9
        return 0.0
    return 1.0 if _has_any_term(text, [raw]) else 0.0


def _readout_match_score(value: Any, text: str) -> float:
    if _emptyish(value):
        return 0.0
    raw = str(value).strip().lower()
    if raw == "protein level":
        if _has_any_term(text, [
            "western blot",
            "western blotting",
            "immunoblot",
            "protein detection",
            "protein level",
            "protein assay",
            "protein visualization",
            "fluorescent protein visualization",
            "protein concentration",
            "elisa",
            "sds-page",
            "qubit protein",
        ]):
            return 1.0
        if _has_any_term(text, ["gfp", "fluorescence", "fluorescent protein", "reporter"]):
            return 0.9
        if _has_any_term(text, ["protein"]):
            return 0.35
        return 0.0
    if raw == "rna level / qpcr":
        return 1.0 if _has_any_term(text, ["qpcr", "q-pcr", "rt-qpcr", "transcript", "mrna", "rna level"]) else 0.0
    if raw == "phenotype":
        return 1.0 if _has_any_term(text, ["phenotype", "phenotyping", "morphology", "growth", "trait"]) else 0.0
    if raw == "localization":
        return 1.0 if _has_any_term(text, ["localization", "localisation", "gfp", "fluorescence", "confocal", "imaging"]) else 0.0
    if raw == "stress response":
        return 1.0 if _has_any_term(text, ["stress response", "drought", "salt", "salinity", "heat stress", "cold stress"]) else 0.0
    if raw == "sequencing":
        return 1.0 if _has_any_term(text, ["sequencing", "rna-seq", "ngs"]) else 0.0
    if raw == "reporter assay":
        return 1.0 if _has_any_term(text, ["reporter assay", "luciferase", "gus", "reporter"]) else 0.0
    return 1.0 if _has_any_term(text, [raw]) else 0.0


def _tissue_match_score(value: Any, text: str) -> float:
    if _emptyish(value) or _is_generic(str(value)):
        return 0.0
    raw = str(value).strip().lower()
    if raw in {"leaf", "leaf tissue", "leaves"}:
        if _has_any_term(text, ["leaf", "leaves", "leaf tissue"]):
            return 1.0
        if _has_any_term(text, ["mesophyll protoplast", "mesophyll protoplasts"]):
            return 0.25
        return 0.0
    if raw in {"whole plant", "whole-plant", "in planta / whole plant"}:
        return 1.0 if _has_any_term(text, ["whole plant", "whole-plant", "in planta", "plant"]) else 0.0
    if raw in {"protoplast", "protoplasts"}:
        return 1.0 if _has_any_term(text, ["protoplast", "protoplasts"]) else 0.0
    if raw in {"cell culture", "cells", "cell"}:
        return 1.0 if _has_any_term(text, ["cell culture", "cultured cells", "cells"]) else 0.0
    return 1.0 if _has_any_term(text, [raw]) else 0.0


def _stable_transformation_delivery_score(text: str) -> float:
    if _has_any_term(text, _STABLE_TRANSFORMATION_DELIVERY_TERMS):
        return 1.0
    if _has_any_term(text, ["agrobacterium", "biolistic", "biolistics", "particle bombardment", "gene gun"]):
        return 0.9
    if _has_any_term(text, ["immature embryo", "somatic embryogenesis"]) and _has_any_term(text, ["transformation", "transform", "transgenic"]):
        return 0.9
    if _has_any_term(text, ["callus", "regeneration"]) and _has_any_term(text, ["transformation", "transgenic", "transformant"]):
        return 0.75
    return 0.0


def _combined_match_score(
    organism_match: float,
    method_match: float,
    expression_match: float,
    tissue_match: float,
    readout_match: float,
) -> float:
    strong_organism = organism_match >= 0.75
    strong_method = method_match >= 0.75
    strong_expression = expression_match >= 0.75
    strong_tissue = tissue_match >= 0.75
    strong_readout = readout_match >= 0.75

    score = 0.0
    if strong_organism and strong_expression:
        score += 0.35
    if strong_expression and strong_tissue:
        score += 0.3
    if strong_method and strong_expression:
        score += 0.25
    if strong_expression and strong_readout:
        score += 0.2
    if strong_organism and strong_expression and strong_tissue:
        score += 0.55
    if strong_organism and strong_expression and strong_tissue and strong_readout:
        score += 0.45
    return min(1.0, score)


def _title_context_match_score(profile: Dict[str, Any], title: str) -> float:
    if not title:
        return 0.0
    organism = _organism_match_score(profile.get("organism"), title)
    expression = _expression_match_score(profile.get("expression_type"), title)
    tissue = _tissue_match_score(profile.get("tissue_or_cell_type"), title)
    readout = _readout_match_score(profile.get("readout_assay"), title)
    method = _method_match_score(profile.get("experimental_method"), title)

    if organism >= 0.75 and expression >= 0.75 and tissue >= 0.75:
        return 1.0
    if expression >= 0.75 and tissue >= 0.75 and method >= 0.75:
        return 0.9
    if organism >= 0.75 and expression >= 0.75:
        return 0.75
    if tissue >= 0.75 and readout >= 0.75:
        return 0.55
    return 0.0


def _false_positive_penalty(
    profile: Dict[str, Any],
    text: str,
    organism_match: float,
    expression_match: float,
    tissue_match: float,
    method_match: float,
    protoplast_mismatch: bool,
) -> tuple:
    penalty = 1.0
    method = str(profile.get("experimental_method") or "").lower()
    expression = str(profile.get("expression_type") or "").lower()
    readout = str(profile.get("readout_assay") or profile.get("readout") or "").lower()
    tissue = str(profile.get("tissue_or_cell_type") or "").lower()
    plant_context = _has_any_term(text, _PLANT_CONTEXT_TERMS)
    needs_plant_transient = method == "gene overexpression" and expression == "transient expression"
    needs_stable_overexpression = method == "gene overexpression" and expression == "stable transformation"
    needs_rna_qpcr_overexpression = method == "gene overexpression" and _is_rna_qpcr_readout(readout)

    tev_false_positive = needs_plant_transient and _has_any_term(text, _TEV_FALSE_POSITIVE_TERMS)
    if tev_false_positive and not (
        organism_match >= 0.75
        or (plant_context and expression_match >= 0.75)
        or tissue_match >= 0.75
    ):
        penalty *= 0.25

    generic_protein_purification = needs_plant_transient and _has_any_term(text, _GENERIC_PROTEIN_PURIFICATION_TERMS)
    if generic_protein_purification and not (
        organism_match >= 0.75
        or expression_match >= 0.75
        or tissue_match >= 0.75
    ):
        penalty *= 0.35

    if needs_plant_transient and expression_match < 0.4:
        penalty *= 0.7
    if needs_plant_transient and method_match < 0.4 and expression_match < 0.4:
        penalty *= 0.8
    if needs_plant_transient and organism_match < 0.4 and tissue_match < 0.4 and not plant_context:
        penalty *= 0.55
    if needs_plant_transient and organism_match < 0.4 and not _missing_or_generic(profile.get("organism")):
        penalty *= 0.75
    if tissue in {"leaf", "leaf tissue", "leaves"} and protoplast_mismatch:
        penalty *= 0.85

    rna_qpcr_off_target = needs_rna_qpcr_overexpression and _has_any_term(text, _RNA_QPCR_OFF_TARGET_TERMS)
    if rna_qpcr_off_target:
        penalty *= 0.2
    if needs_rna_qpcr_overexpression and not _has_any_term(text, [
        "qpcr",
        "q-pcr",
        "rt-qpcr",
        "real-time pcr",
        "transcript",
        "mrna",
        "rna level",
        "gene expression assay",
    ]):
        penalty *= 0.35
    if needs_rna_qpcr_overexpression and expression == "transient expression" and expression_match < 0.5:
        penalty *= 0.45
    if needs_rna_qpcr_overexpression and method_match < 0.5:
        penalty *= 0.55

    stable_off_target = needs_stable_overexpression and _has_any_term(text, _STABLE_TRANSFORMATION_OFF_TARGET_TERMS)
    delivery_match = max(expression_match, method_match)
    if needs_stable_overexpression and not _missing_or_generic(profile.get("organism")) and organism_match < 0.4:
        # Transformation in the wrong organism should not beat a maize-specific protocol.
        penalty *= 0.3
    if needs_stable_overexpression and delivery_match < 0.4:
        # Maize-only hydroponics/sequencing/selection hits are off-goal.
        penalty *= 0.25
    if stable_off_target and delivery_match < 0.6:
        penalty *= 0.3

    return max(0.05, penalty), tev_false_positive, generic_protein_purification, stable_off_target, rna_qpcr_off_target


def _protoplast_leaf_mismatch(profile: Dict[str, Any], text: str, title: str = "") -> bool:
    tissue = str(profile.get("tissue_or_cell_type") or "").lower()
    if tissue not in {"leaf", "leaf tissue", "leaves"}:
        return False
    if _has_any_term(title, ["protoplast", "protoplasts"]):
        return True
    return _has_any_term(text, ["protoplast", "protoplasts"]) and not _has_any_term(text, ["leaf", "leaves", "leaf tissue"])


def _has_any_term(text: str, terms: List[str]) -> bool:
    return any(_has_term(text, term) for term in terms)


def _has_term(text: str, term: str) -> bool:
    term = (term or "").strip().lower()
    if not term:
        return False
    escaped = re.escape(term).replace(r"\ ", r"[\s\-_]+")
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None


def _recency_score(value: Any) -> float:
    if not value:
        return 0.0
    try:
        ts = int(value)
    except Exception:
        return 0.0
    age_years = max(0.0, (time.time() - ts) / (365.25 * 24 * 60 * 60))
    if age_years <= 2:
        return 1.0
    if age_years <= 5:
        return 0.6
    if age_years <= 10:
        return 0.3
    return 0.1


def _human_field(field: str) -> str:
    return field.replace("_", " ")


def surface_growth_stage(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror intent_specific.growth_stage (e.g. "seedlings") into the top-level
    Sample type / System cells when they're empty, so a stress-assay growth-stage
    answer shows in the profile grid rather than only in intent_specific. Applied
    as a FINAL step after clarification processing — the clarification-answer fixer
    strips the value from non-pending fields, so this restores it deterministically.
    """
    if not isinstance(profile, dict):
        return profile
    intent_specific = profile.get("intent_specific")
    growth_stage = intent_specific.get("growth_stage") if isinstance(intent_specific, dict) else None
    if growth_stage and _emptyish(profile.get("sample_type")) and _emptyish(profile.get("tissue_or_cell_type")):
        profile["sample_type"] = str(growth_stage).strip()
        profile["tissue_or_cell_type"] = str(growth_stage).strip()
    return profile


# Register the per-field matchers with the declarative schema (late-bound to avoid
# a circular import — the schema declares field→matcher wiring, the matcher logic
# lives here alongside the biology term lists it depends on).
profile_schema.register_matcher("organism_match", _organism_match_score)
profile_schema.register_matcher("method_match", _method_match_score)
profile_schema.register_matcher("readout_match", _readout_match_score)
profile_schema.register_matcher("expression_match", _expression_match_score)
profile_schema.register_matcher("tissue_match", _tissue_match_score)
