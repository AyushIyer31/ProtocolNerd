from __future__ import annotations

from typing import Any, Dict, Optional


INTENT_FAMILIES = {
    "gene_nucleic_acid_manipulation",
    "gene_expression_analysis",
    "protein_work",
    "microscopy_imaging",
    "sequencing_omics",
    "phenotyping_physiology",
    "organism_transformation",
    "general_protocol_search",
    "unknown",
}


INTENT_FAMILY_LABELS = {
    "gene_nucleic_acid_manipulation": "gene / nucleic acid manipulation",
    "gene_expression_analysis": "gene expression analysis",
    "protein_work": "protein work",
    "microscopy_imaging": "microscopy / imaging",
    "sequencing_omics": "sequencing / omics",
    "phenotyping_physiology": "phenotyping / physiology",
    "organism_transformation": "organism transformation",
    "general_protocol_search": "general protocol search",
    "unknown": "unknown",
}


SUB_INTENTS = {
    "gene_modification",
    "multiplex_gene_modification",
    "gene_overexpression",
    "gene_knockdown",
    "genome_editing",
    "cloning",
    "mutagenesis",
    "pcr_qpcr",
    "protein_purification",
    "protein_extraction",
    "western_blot",
    "protein_detection",
    "microscopy",
    "sequencing_prep",
    "stress_tolerance_assay",
    "phenotyping_assay",
    "growth_assay",
    "gas_exchange",
    "transformation",
    "general_protocol_search",
    "chitchat",
    "unknown",
}


SUB_INTENT_LABELS = {
    "gene_modification": "gene modification",
    "multiplex_gene_modification": "multiplex gene modification",
    "gene_overexpression": "gene overexpression",
    "gene_knockdown": "gene knockdown",
    "genome_editing": "genome editing",
    "cloning": "cloning",
    "mutagenesis": "mutagenesis",
    "pcr_qpcr": "PCR/qPCR",
    "protein_purification": "protein purification",
    "protein_extraction": "protein extraction",
    "western_blot": "western blot",
    "protein_detection": "protein detection",
    "microscopy": "microscopy",
    "sequencing_prep": "sequencing prep",
    "stress_tolerance_assay": "stress tolerance assay",
    "phenotyping_assay": "phenotyping assay",
    "growth_assay": "growth assay",
    "gas_exchange": "gas exchange",
    "transformation": "transformation",
    "general_protocol_search": "general protocol search",
    "chitchat": "chitchat",
    "unknown": "unknown",
}


SUB_INTENT_TO_FAMILY = {
    "gene_modification": "gene_nucleic_acid_manipulation",
    "multiplex_gene_modification": "gene_nucleic_acid_manipulation",
    "gene_overexpression": "gene_nucleic_acid_manipulation",
    "gene_knockdown": "gene_nucleic_acid_manipulation",
    "genome_editing": "gene_nucleic_acid_manipulation",
    "cloning": "gene_nucleic_acid_manipulation",
    "mutagenesis": "gene_nucleic_acid_manipulation",
    "pcr_qpcr": "gene_expression_analysis",
    "protein_purification": "protein_work",
    "protein_extraction": "protein_work",
    "western_blot": "protein_work",
    "protein_detection": "protein_work",
    "microscopy": "microscopy_imaging",
    "sequencing_prep": "sequencing_omics",
    "stress_tolerance_assay": "phenotyping_physiology",
    "phenotyping_assay": "phenotyping_physiology",
    "growth_assay": "phenotyping_physiology",
    "gas_exchange": "phenotyping_physiology",
    "transformation": "organism_transformation",
    "general_protocol_search": "general_protocol_search",
    "unknown": "unknown",
}


FAMILY_ALIASES = {
    "gene manipulation": "gene_nucleic_acid_manipulation",
    "nucleic acid manipulation": "gene_nucleic_acid_manipulation",
    "gene_nucleic_acid_manipulation": "gene_nucleic_acid_manipulation",
    "gene expression analysis": "gene_expression_analysis",
    "gene_expression_analysis": "gene_expression_analysis",
    "protein": "protein_work",
    "protein work": "protein_work",
    "protein_work": "protein_work",
    "microscopy": "microscopy_imaging",
    "imaging": "microscopy_imaging",
    "microscopy_imaging": "microscopy_imaging",
    "sequencing": "sequencing_omics",
    "omics": "sequencing_omics",
    "sequencing_omics": "sequencing_omics",
    "phenotyping": "phenotyping_physiology",
    "physiology": "phenotyping_physiology",
    "phenotyping_physiology": "phenotyping_physiology",
    "transformation": "organism_transformation",
    "organism transformation": "organism_transformation",
    "organism_transformation": "organism_transformation",
    "general": "general_protocol_search",
    "general protocol search": "general_protocol_search",
    "general_protocol_search": "general_protocol_search",
    "unknown": "unknown",
}


SUB_INTENT_ALIASES = {
    "gene modification": "gene_modification",
    "gene_modification": "gene_modification",
    "modified genes": "gene_modification",
    "modify genes": "gene_modification",
    "gene editing": "genome_editing",
    "gene_editing": "genome_editing",
    "genome editing": "genome_editing",
    "genome_editing": "genome_editing",
    "crispr": "genome_editing",
    "cas9": "genome_editing",
    "multiplex gene modification": "multiplex_gene_modification",
    "multiplex_gene_modification": "multiplex_gene_modification",
    "multiple gene modification": "multiplex_gene_modification",
    "gene overexpression": "gene_overexpression",
    "gene_overexpression": "gene_overexpression",
    "overexpression": "gene_overexpression",
    "overexpress": "gene_overexpression",
    "transgene expression": "gene_overexpression",
    "gene knockdown": "gene_knockdown",
    "gene_knockdown": "gene_knockdown",
    "knockdown": "gene_knockdown",
    "rnai": "gene_knockdown",
    "silencing": "gene_knockdown",
    "cloning": "cloning",
    "mutagenesis": "mutagenesis",
    "mutation": "mutagenesis",
    "pcr": "pcr_qpcr",
    "qpcr": "pcr_qpcr",
    "pcr_qpcr": "pcr_qpcr",
    "rt-qpcr": "pcr_qpcr",
    "protein purification": "protein_purification",
    "protein_purification": "protein_purification",
    "purify protein": "protein_purification",
    "protein extraction": "protein_extraction",
    "protein_extraction": "protein_extraction",
    "western blot": "western_blot",
    "western_blot": "western_blot",
    "immunoblot": "western_blot",
    "protein detection": "protein_detection",
    "protein_detection": "protein_detection",
    "microscopy": "microscopy",
    "imaging": "microscopy",
    "sequencing prep": "sequencing_prep",
    "sequencing_prep": "sequencing_prep",
    "library prep": "sequencing_prep",
    "stress tolerance": "stress_tolerance_assay",
    "stress tolerance assay": "stress_tolerance_assay",
    "stress_tolerance_assay": "stress_tolerance_assay",
    "drought tolerance": "stress_tolerance_assay",
    "phenotyping": "phenotyping_assay",
    "phenotyping assay": "phenotyping_assay",
    "phenotyping_assay": "phenotyping_assay",
    "growth assay": "growth_assay",
    "growth_assay": "growth_assay",
    "gas exchange": "gas_exchange",
    "gas_exchange": "gas_exchange",
    "transformation": "transformation",
    "plant transformation": "transformation",
    "general": "general_protocol_search",
    "general protocol search": "general_protocol_search",
    "general_protocol_search": "general_protocol_search",
    "chitchat": "chitchat",
    "unknown": "unknown",
}


def _normalize_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", " ").replace("_", " ")


def normalize_intent_family(value: Any) -> str:
    key = _normalize_key(value)
    compact = key.replace(" ", "_")
    if compact in INTENT_FAMILIES:
        return compact
    if key in FAMILY_ALIASES:
        return FAMILY_ALIASES[key]
    for phrase, family in FAMILY_ALIASES.items():
        if phrase not in {"unknown", "general"} and phrase in key:
            return family
    return "unknown"


def normalize_sub_intent(value: Any) -> str:
    key = _normalize_key(value)
    compact = key.replace(" ", "_")
    if compact in SUB_INTENTS:
        return compact
    if key in SUB_INTENT_ALIASES:
        return SUB_INTENT_ALIASES[key]
    for phrase, sub_intent in SUB_INTENT_ALIASES.items():
        if phrase not in {"unknown", "general"} and phrase in key:
            return sub_intent
    return "unknown"


def family_for_sub_intent(sub_intent: Any) -> str:
    normalized = normalize_sub_intent(sub_intent)
    return SUB_INTENT_TO_FAMILY.get(normalized, "unknown")


def controlled_intent_payload(
    sub_intent: Any,
    *,
    intent_family: Optional[Any] = None,
    confidence: Any = 0.7,
    alternatives: Optional[list] = None,
) -> Dict[str, Any]:
    normalized_sub = normalize_sub_intent(sub_intent)
    normalized_family = normalize_intent_family(intent_family)
    if normalized_family == "unknown":
        normalized_family = family_for_sub_intent(normalized_sub)
    return {
        "intent": normalized_sub,
        "label": SUB_INTENT_LABELS.get(normalized_sub, normalized_sub.replace("_", " ")),
        "intent_family": normalized_family,
        "intent_family_label": INTENT_FAMILY_LABELS.get(normalized_family, normalized_family.replace("_", " ")),
        "sub_intent": normalized_sub,
        "sub_intent_label": SUB_INTENT_LABELS.get(normalized_sub, normalized_sub.replace("_", " ")),
        "confidence": confidence,
        "alternatives": alternatives or [],
    }
