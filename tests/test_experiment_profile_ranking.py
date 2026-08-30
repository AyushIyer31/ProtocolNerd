import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "protocolnerd-backend"))

from experiment_profile import (
    apply_profile_ranking,
    build_experiment_profile,
    detect_experiment_intent,
    generate_candidate_search_queries,
    normalize_experiment_goal,
    next_biology_clarification,
    next_clarification,
    organism_aware_system_clarification,
    profile_source_query_for_request,
    should_respond_as_chitchat,
    validate_biology_profile,
)
from main import _clarification_from_plan


class ExperimentProfileRankingTest(unittest.TestCase):
    def test_nicotiana_transient_leaf_protein_profile_prioritizes_combined_match(self):
        profile = {
            "organism": "Nicotiana benthamiana",
            "tissue_or_cell_type": "leaf tissue",
            "gene_or_construct": None,
            "experimental_method": "gene overexpression",
            "expression_type": "transient expression",
            "readout_assay": "protein level",
            "timeline": None,
            "required_equipment": [],
            "protocol_difficulty": None,
            "constraints": [],
        }
        results = [
            {
                "title": "Leaf Protein Extraction for Immunoblot (Soybean, Cowpea, Tobacco)",
                "description": "Extract total protein from tobacco leaves for immunoblot analysis.",
                "score": 7.0,
            },
            {
                "title": "Cleavage of the Fusion Protein (TEV Protease)",
                "description": "Remove affinity tags using Tobacco Etch Virus TEV protease.",
                "score": 8.0,
            },
            {
                "title": "Isolation and Transfection of Nicotiana benthamiana Mesophyll Protoplasts for Fluorescent Protein Visualization",
                "description": "Transfect Nicotiana benthamiana protoplasts for fluorescent protein visualization.",
                "score": 6.0,
            },
            {
                "title": "Agrobacterium-Mediated Transient Expression in Nicotiana benthamiana Leaves",
                "description": "Transient expression in Nicotiana benthamiana leaves using Agrobacterium infiltration.",
                "score": 5.0,
            },
            {
                "title": "pA-Hia5 Protein Expression and Purification",
                "description": "Purify recombinant pA-Hia5 from E. coli.",
                "score": 9.0,
            },
            {
                "title": "Co-immunoprecipitation in Agrobacterium-Mediated Transient Expression System in Nicotiana benthamiana",
                "description": "Assess protein-protein interactions after transient expression in Nicotiana benthamiana leaves.",
                "score": 4.0,
            },
            {
                "title": "Transient GFP reporter assay in Nicotiana benthamiana leaves",
                "description": "Agroinfiltration assay for fluorescent protein readout in leaves.",
                "score": 4.0,
            },
        ]

        ranked = apply_profile_ranking(profile, results, top_k=5)
        titles = [result["title"] for result in ranked]

        self.assertIn(
            "Agrobacterium-Mediated Transient Expression in Nicotiana benthamiana Leaves",
            titles[:3],
        )
        self.assertNotIn("Cleavage of the Fusion Protein (TEV Protease)", titles)
        self.assertNotIn("pA-Hia5 Protein Expression and Purification", titles)
        self.assertLess(
            titles.index("Agrobacterium-Mediated Transient Expression in Nicotiana benthamiana Leaves"),
            titles.index("Leaf Protein Extraction for Immunoblot (Soybean, Cowpea, Tobacco)"),
        )

    def test_maize_stable_transformation_keeps_overexpression_goal(self):
        source_query = "I want to overexpress a gene in maize using stable transformation"
        intent = {"intent": "transformation", "label": "transformation", "confidence": 0.9}
        profile = {
            "organism": "maize",
            "tissue_or_cell_type": "tissue",
            "gene_or_construct": None,
            "experimental_method": "plant transformation",
            "expression_type": "stable transformation",
            "readout_assay": None,
            "timeline": None,
            "required_equipment": [],
            "protocol_difficulty": None,
            "constraints": [],
        }

        normalized_intent, normalized_profile = normalize_experiment_goal(source_query, intent, profile)

        self.assertEqual(normalized_intent["intent"], "gene_overexpression")
        self.assertEqual(normalized_profile["experimental_method"], "gene overexpression")
        self.assertEqual(normalized_profile["expression_type"], "stable transformation")
        self.assertEqual(normalized_profile["organism"], "maize")

    def test_maize_stable_overexpression_queries_preserve_required_concepts(self):
        profile = {
            "organism": "maize",
            "tissue_or_cell_type": "tissue",
            "gene_or_construct": None,
            "experimental_method": "gene overexpression",
            "expression_type": "stable transformation",
            "readout_assay": None,
            "timeline": None,
            "required_equipment": [],
            "protocol_difficulty": None,
            "constraints": [],
        }

        queries = generate_candidate_search_queries(profile, max_queries=5)

        self.assertIn("maize stable transformation gene overexpression", queries)
        self.assertIn("Zea mays transgenic overexpression protocol", queries)
        self.assertIn("maize Agrobacterium transformation overexpression", queries)
        self.assertIn("maize immature embryo transformation transgene expression", queries)
        self.assertIn("maize biolistic transformation gene overexpression", queries)

    def test_maize_stable_transformation_ranks_required_concept_overlap(self):
        profile = {
            "organism": "maize",
            "tissue_or_cell_type": "tissue",
            "gene_or_construct": None,
            "experimental_method": "gene overexpression",
            "expression_type": "stable transformation",
            "readout_assay": None,
            "timeline": None,
            "required_equipment": [],
            "protocol_difficulty": None,
            "constraints": [],
        }
        results = [
            {
                "title": "Tomato Agrobacterium-mediated transformation protocol",
                "description": "Stable plant transformation and regeneration of tomato transgenic plants.",
                "score": 9.0,
            },
            {
                "title": "Maize hydroponics and root ion uptake kinetics",
                "description": "Hydroponics protocol for maize growth and uptake phenotyping.",
                "score": 10.0,
            },
            {
                "title": "Maize ATLAS selection workflow",
                "description": "Selection and genotyping assay for Zea mays panels.",
                "score": 8.0,
            },
            {
                "title": "BSA-seq in Maize",
                "description": "Bulk segregant sequencing analysis in maize.",
                "score": 8.0,
            },
            {
                "title": "SAM-Seq Zea Mays",
                "description": "Sequencing sample preparation for Zea mays shoot apical meristem.",
                "score": 7.0,
            },
            {
                "title": "Maize immature embryo Agrobacterium transformation for transgenic overexpression",
                "description": "Stable transformation of Zea mays immature embryos for transgene expression and plant regeneration.",
                "score": 4.0,
            },
            {
                "title": "Zea mays biolistic transformation for gene overexpression",
                "description": "Particle bombardment protocol for stable transgenic maize lines.",
                "score": 4.0,
            },
            {
                "title": "Maize callus regeneration after stable transformation",
                "description": "Regeneration and transformant selection for maize transgenic plants.",
                "score": 4.0,
            },
            {
                "title": "Maize Agrobacterium transformation vector delivery",
                "description": "Agrobacterium delivery into Zea mays immature embryos for stable plant transformation.",
                "score": 4.0,
            },
            {
                "title": "Zea mays transgenic plant generation protocol",
                "description": "Generate stable transgenic maize lines from transformed immature embryos.",
                "score": 4.0,
            },
        ]

        ranked = apply_profile_ranking(profile, results, top_k=5)
        titles = [result["title"] for result in ranked]

        self.assertIn(
            "Maize immature embryo Agrobacterium transformation for transgenic overexpression",
            titles[:3],
        )
        self.assertNotIn("Maize hydroponics and root ion uptake kinetics", titles)
        self.assertNotIn("Maize ATLAS selection workflow", titles)
        self.assertNotIn("BSA-seq in Maize", titles)
        self.assertNotIn("SAM-Seq Zea Mays", titles)
        if "Tomato Agrobacterium-mediated transformation protocol" in titles:
            self.assertLess(
                titles.index("Maize immature embryo Agrobacterium transformation for transgenic overexpression"),
                titles.index("Tomato Agrobacterium-mediated transformation protocol"),
            )

    def test_active_experiment_context_blocks_chitchat_response(self):
        intent = {"intent": "chitchat", "label": "chitchat", "confidence": 0.9}

        self.assertFalse(should_respond_as_chitchat(True, intent, "respond_chitchat"))
        self.assertTrue(should_respond_as_chitchat(False, intent, "respond_chitchat"))

    def test_arabidopsis_phenotype_answer_completes_profile_for_query_selection(self):
        previous_profile = {
            "organism": "Arabidopsis thaliana",
            "tissue_or_cell_type": "leaf",
            "gene_or_construct": None,
            "experimental_method": "gene overexpression",
            "expression_type": "transient expression",
            "readout_assay": None,
            "timeline": None,
            "required_equipment": [],
            "protocol_difficulty": None,
            "constraints": [],
        }
        source_query = "\n".join([
            "how to test what happens when overexpressing a gene in a plant",
            "phenotype",
        ])

        intent = detect_experiment_intent(source_query)
        profile = build_experiment_profile(source_query, previous_profile=previous_profile)
        clarification = next_clarification(profile, intent)
        queries = generate_candidate_search_queries(profile, fallback_query=source_query, max_queries=5)

        self.assertIsNone(clarification)
        self.assertEqual(profile["readout_assay"], "phenotype")
        self.assertTrue(any("Arabidopsis" in query or "arabidopsis" in query for query in queries))
        self.assertTrue(any("phenotype" in query.lower() or "phenotyping" in query.lower() for query in queries))

    def test_confirmed_search_does_not_parse_selected_query_into_profile(self):
        previous_profile = {
            "organism": "tomato",
            "tissue_or_cell_type": "cell culture",
            "gene_or_construct": None,
            "experimental_method": "gene overexpression",
            "expression_type": "stable transformation",
            "readout_assay": "phenotype",
            "timeline": None,
            "required_equipment": [],
            "protocol_difficulty": None,
            "constraints": [],
        }
        selected_query = "tomato stable transformation gene overexpression"
        conversation_query = "how to test what happens when overexpressing a gene in a plant"

        source_query = profile_source_query_for_request(
            query=selected_query,
            conversation_query=conversation_query,
            search_confirmed=True,
            experiment_profile=previous_profile,
        )
        parsed_profile = build_experiment_profile(source_query, previous_profile=previous_profile)

        self.assertNotIn(selected_query, source_query)
        self.assertIsNone(parsed_profile["gene_or_construct"])

    def test_method_words_are_not_extracted_as_gene_constructs(self):
        profile = build_experiment_profile("tomato stable transformation gene overexpression")

        self.assertIsNone(profile["gene_or_construct"])

    def test_stable_transformation_asks_for_starting_material(self):
        profile = {
            "organism": "tomato",
            "tissue_or_cell_type": None,
            "gene_or_construct": None,
            "experimental_method": "gene overexpression",
            "expression_type": "stable transformation",
            "readout_assay": "phenotype",
            "timeline": None,
            "required_equipment": [],
            "protocol_difficulty": None,
            "constraints": [],
        }
        intent = {"intent": "gene_overexpression", "label": "gene overexpression"}

        clarification = next_clarification(profile, intent)

        self.assertEqual(clarification["field"], "tissue_or_cell_type")
        self.assertIn("immature embryos", clarification["options"])
        self.assertIn("callus / tissue culture", clarification["options"])

    def test_overexpression_source_overrides_unknown_llm_intent(self):
        source_query = "how to test what happens when overexpressing a gene in a plant"
        intent = {"intent": "unknown", "label": "unknown", "confidence": 0.7}
        profile = {
            "organism": "plant",
            "tissue_or_cell_type": None,
            "gene_or_construct": None,
            "experimental_method": None,
            "expression_type": None,
            "readout_assay": None,
            "timeline": None,
            "required_equipment": [],
            "protocol_difficulty": None,
            "constraints": [],
        }

        normalized_intent, normalized_profile = normalize_experiment_goal(source_query, intent, profile)

        self.assertEqual(normalized_intent["intent"], "gene_overexpression")
        self.assertEqual(normalized_profile["experimental_method"], "gene overexpression")

    def test_empty_profile_does_not_generate_junk_candidate_queries(self):
        profile = {
            "organism": "plant",
            "tissue_or_cell_type": None,
            "gene_or_construct": None,
            "experimental_method": None,
            "expression_type": None,
            "readout_assay": None,
            "timeline": None,
            "required_equipment": [],
            "protocol_difficulty": None,
            "constraints": [],
        }

        queries = generate_candidate_search_queries(
            profile,
            fallback_query="What plant species or experimental system are you working with?",
        )

        self.assertEqual(queries, [])

    def test_common_crop_species_answers_are_recognized(self):
        previous_profile = {
            "organism": "plant",
            "tissue_or_cell_type": None,
            "gene_or_construct": None,
            "experimental_method": "gene overexpression",
            "expression_type": None,
            "readout_assay": None,
            "timeline": None,
            "required_equipment": [],
            "protocol_difficulty": None,
            "constraints": [],
        }
        source_query = "\n".join([
            "how to test what happens when overexpressing a gene in a plant",
            "beans",
        ])

        profile = build_experiment_profile(source_query, previous_profile=previous_profile)
        clarification = next_clarification(profile, {"intent": "gene_overexpression"})

        self.assertEqual(profile["organism"], "bean")
        self.assertEqual(clarification["field"], "expression_type")

    def test_pumpkin_species_answer_is_recognized(self):
        previous_profile = {
            "organism": "plant",
            "tissue_or_cell_type": None,
            "gene_or_construct": None,
            "experimental_method": "gene overexpression",
            "expression_type": None,
            "readout_assay": None,
            "timeline": None,
            "required_equipment": [],
            "protocol_difficulty": None,
            "constraints": [],
        }
        source_query = "\n".join([
            "how to test what happens when overexpressing a gene in a plant",
            "pumpkin",
        ])

        profile = build_experiment_profile(source_query, previous_profile=previous_profile)

        self.assertEqual(profile["organism"], "pumpkin")

    def test_unlisted_short_species_answer_is_kept_as_organism(self):
        previous_profile = {
            "organism": "plant",
            "tissue_or_cell_type": None,
            "gene_or_construct": None,
            "experimental_method": "gene overexpression",
            "expression_type": None,
            "readout_assay": None,
            "timeline": None,
            "required_equipment": [],
            "protocol_difficulty": None,
            "constraints": [],
        }
        source_query = "\n".join([
            "how to test what happens when overexpressing a gene in a plant",
            "lentil",
        ])

        profile = build_experiment_profile(source_query, previous_profile=previous_profile)

        self.assertEqual(profile["organism"], "lentil")

    def test_method_answer_is_not_misread_as_organism(self):
        previous_profile = {
            "organism": "plant",
            "tissue_or_cell_type": None,
            "gene_or_construct": None,
            "experimental_method": "gene overexpression",
            "expression_type": None,
            "readout_assay": None,
            "timeline": None,
            "required_equipment": [],
            "protocol_difficulty": None,
            "constraints": [],
        }
        source_query = "\n".join([
            "how to test what happens when overexpressing a gene in a plant",
            "stable transformation",
        ])

        profile = build_experiment_profile(source_query, previous_profile=previous_profile)

        self.assertEqual(profile["organism"], "plant")
        self.assertEqual(profile["expression_type"], "stable transformation")

    def test_ambiguous_modified_genes_maps_to_gene_modification(self):
        source_query = "Find protocols that can allow in-planta tests in which genes are modified."

        intent = detect_experiment_intent(source_query)
        profile = build_experiment_profile(source_query)
        clarification = next_clarification(profile, intent)
        queries = generate_candidate_search_queries(profile, fallback_query=source_query, max_queries=5)

        self.assertEqual(intent["intent"], "gene_modification")
        self.assertEqual(intent["intent_family"], "gene_nucleic_acid_manipulation")
        self.assertEqual(intent["label"], "gene modification")
        self.assertEqual(profile["intent_family"], "gene_nucleic_acid_manipulation")
        self.assertEqual(profile["sub_intent"], "gene_modification")
        self.assertEqual(profile["experimental_method"], "gene modification")
        self.assertEqual(profile["tissue_or_cell_type"], "in planta / whole plant")
        self.assertIsNone(profile["modification_type"])
        self.assertEqual(clarification["field"], "modification_type")
        self.assertEqual(clarification["question"], "What kind of gene modification do you mean?")
        self.assertIn("CRISPR / genome editing", clarification["options"])
        self.assertEqual(queries, [])

    def test_gene_modification_asks_type_before_overexpression_questions(self):
        profile = {
            "organism": "Nicotiana benthamiana",
            "tissue_or_cell_type": "in planta / whole plant",
            "gene_or_construct": None,
            "modification_type": None,
            "experimental_method": "gene modification",
            "delivery_method": None,
            "expression_type": None,
            "readout_assay": None,
            "timeline": None,
            "required_equipment": [],
            "protocol_difficulty": None,
            "constraints": [],
        }
        intent = {"intent": "gene_modification", "label": "gene modification"}

        clarification = next_clarification(profile, intent)

        self.assertEqual(clarification["field"], "modification_type")
        self.assertNotIn("overexpression", clarification["question"].lower())

    def test_gene_modification_queries_include_organism_type_and_in_planta(self):
        profile = {
            "organism": "Nicotiana benthamiana",
            "tissue_or_cell_type": "in planta / whole plant",
            "gene_or_construct": None,
            "modification_type": "CRISPR / genome editing",
            "experimental_method": "gene modification",
            "delivery_method": "Agrobacterium-mediated delivery",
            "expression_type": None,
            "readout_assay": "genotyping / sequencing",
            "timeline": None,
            "required_equipment": [],
            "protocol_difficulty": None,
            "constraints": [],
        }

        queries = generate_candidate_search_queries(profile, max_queries=5)
        combined = " ".join(queries).lower()

        self.assertTrue(any("nicotiana benthamiana" in query.lower() for query in queries))
        self.assertIn("crispr", combined)
        self.assertIn("in planta", combined)

    def test_multiplex_gene_modification_keeps_ambiguous_modification_type(self):
        source_query = "Find protocols that allow more than one transcription factor to be modified at the same time for mice."

        intent = detect_experiment_intent(source_query)
        profile = build_experiment_profile(source_query)
        intent, profile = validate_biology_profile(source_query, intent, profile)
        clarification = next_biology_clarification(profile, intent)
        queries = generate_candidate_search_queries(profile, fallback_query=source_query, max_queries=5)

        self.assertEqual(intent["intent_family"], "gene_nucleic_acid_manipulation")
        self.assertEqual(intent["sub_intent"], "multiplex_gene_modification")
        self.assertEqual(profile["organism"], "mouse")
        self.assertEqual(profile["target"], "transcription factors")
        self.assertTrue(profile["intent_specific"].get("multiplex"))
        self.assertIsNone(profile["modification_type"])
        self.assertEqual(clarification["field"], "modification_type")
        self.assertEqual(queries, [])

    def test_mouse_multiplex_crispr_uses_animal_system_options(self):
        source_query = "Find protocols that allow more than one transcription factor to be modified at the same time for mice."
        first_profile = build_experiment_profile(source_query)
        answer_source = "\n".join([source_query, "CRISPR / genome editing"])

        intent = detect_experiment_intent(answer_source)
        profile = build_experiment_profile(answer_source, previous_profile=first_profile)
        intent, profile = validate_biology_profile(answer_source, intent, profile)
        clarification = next_biology_clarification(profile, intent)

        self.assertEqual(intent["sub_intent"], "multiplex_gene_modification")
        self.assertEqual(profile["sub_intent"], "multiplex_gene_modification")
        self.assertEqual(profile["organism"], "mouse")
        self.assertEqual(profile["target"], "transcription factors")
        self.assertEqual(profile["modification_type"], "CRISPR / genome editing")
        self.assertEqual(clarification["field"], "tissue_or_cell_type")
        self.assertEqual(clarification["question"], "What mouse system should be modified or tested?")
        for option in ["whole animal / in vivo", "embryo", "primary cells", "cell line", "specific tissue or organ"]:
            self.assertIn(option, clarification["options"])
        self.assertNotIn("leaf tissue", clarification["options"])
        self.assertNotIn("in planta / whole plant", clarification["options"])
        self.assertNotIn("protoplasts", clarification["options"])

    def test_plant_system_clarification_options_remain_plant_specific(self):
        source_query = "Find protocols that can allow in-planta tests in which genes are modified."
        first_profile = build_experiment_profile(source_query)
        crispr_source = "\n".join([source_query, "CRISPR / genome editing"])
        crispr_profile = build_experiment_profile(crispr_source, previous_profile=first_profile)

        rice_source = "\n".join([source_query, "rice"])
        rice_intent = detect_experiment_intent(rice_source)
        rice_profile = build_experiment_profile(rice_source, previous_profile=crispr_profile)
        rice_intent, rice_profile = validate_biology_profile(rice_source, rice_intent, rice_profile)
        clarification_profile = dict(rice_profile)
        clarification_profile["tissue_or_cell_type"] = None
        clarification = organism_aware_system_clarification(clarification_profile)

        self.assertEqual(rice_profile["organism"], "rice")
        self.assertEqual(rice_profile["modification_type"], "CRISPR / genome editing")
        self.assertEqual(clarification["field"], "tissue_or_cell_type")
        self.assertEqual(clarification["question"], "Where should the modification be tested?")
        for option in ["in planta / whole plant", "leaf tissue", "callus / tissue culture", "immature embryos", "protoplasts"]:
            self.assertIn(option, clarification["options"])
        self.assertNotIn("whole animal / in vivo", clarification["options"])
        self.assertNotIn("embryo", clarification["options"])

    def test_drought_tolerance_maps_to_stress_phenotyping(self):
        source_query = "Find protocols that test for drought tolerance in Rice."

        intent = detect_experiment_intent(source_query)
        profile = build_experiment_profile(source_query)
        intent, profile = validate_biology_profile(source_query, intent, profile)
        clarification = next_biology_clarification(profile, intent)
        queries = generate_candidate_search_queries(profile, fallback_query=source_query, max_queries=5)

        self.assertEqual(intent["intent_family"], "phenotyping_physiology")
        self.assertEqual(intent["sub_intent"], "stress_tolerance_assay")
        self.assertEqual(profile["organism"], "rice")
        self.assertEqual(profile["intent_specific"].get("stress_type"), "drought")
        self.assertIn("drought", profile["readout_assay"])
        self.assertEqual(clarification["field"], "growth_stage")
        self.assertIn("seedlings", clarification["options"])
        self.assertIn("adult plants", clarification["options"])
        self.assertIn("leaf tissue", clarification["options"])
        self.assertNotIn("whole animal / in vivo", clarification["options"])
        self.assertTrue(any("rice drought tolerance" in query.lower() for query in queries))

    def test_ambiguous_gene_modification_is_not_overexpression(self):
        source_query = "Find protocols for in-planta experiments where genes are modified."

        intent = detect_experiment_intent(source_query)
        profile = build_experiment_profile(source_query)
        intent, profile = validate_biology_profile(source_query, intent, profile)

        self.assertEqual(intent["sub_intent"], "gene_modification")
        self.assertNotEqual(intent["sub_intent"], "gene_overexpression")
        self.assertIsNone(profile["modification_type"])

    def test_gene_modification_type_persists_from_intent_specific_across_turns(self):
        original_query = "Find protocols that can allow in-planta tests in which genes are modified."
        first_profile = build_experiment_profile(original_query)

        mutation_source = "\n".join([original_query, "mutation / mutagenesis"])
        mutation_intent = detect_experiment_intent(mutation_source)
        mutation_profile = build_experiment_profile(mutation_source, previous_profile=first_profile)
        mutation_intent, mutation_profile = validate_biology_profile(mutation_source, mutation_intent, mutation_profile)
        mutation_clarification = next_biology_clarification(mutation_profile, mutation_intent)

        self.assertEqual(mutation_profile["modification_type"], "mutation / mutagenesis")
        self.assertEqual(mutation_profile["intent_specific"]["modification_type"], "mutation / mutagenesis")
        self.assertEqual(mutation_clarification["field"], "organism")

        corrupted_profile = dict(mutation_profile)
        corrupted_profile["modification_type"] = None
        rice_source = "\n".join([original_query, "rice"])
        rice_intent = detect_experiment_intent(rice_source)
        rice_profile = build_experiment_profile(rice_source, previous_profile=corrupted_profile)
        rice_intent, rice_profile = validate_biology_profile(rice_source, rice_intent, rice_profile)
        rice_clarification = next_biology_clarification(rice_profile, rice_intent)

        self.assertEqual(rice_profile["organism"], "rice")
        self.assertEqual(rice_profile["tissue_or_cell_type"], "in planta / whole plant")
        self.assertEqual(rice_profile["modification_type"], "mutation / mutagenesis")
        self.assertEqual(rice_profile["intent_specific"]["modification_type"], "mutation / mutagenesis")
        self.assertEqual(rice_clarification["field"], "delivery_method")

        delivery_source = "\n".join([original_query, "stable transformation"])
        delivery_intent = detect_experiment_intent(delivery_source)
        delivery_profile = build_experiment_profile(delivery_source, previous_profile=rice_profile)
        delivery_intent, delivery_profile = validate_biology_profile(delivery_source, delivery_intent, delivery_profile)
        delivery_clarification = next_biology_clarification(delivery_profile, delivery_intent)

        self.assertEqual(delivery_profile["modification_type"], "mutation / mutagenesis")
        self.assertEqual(delivery_profile["delivery_method"], "stable transformation")
        self.assertEqual(delivery_profile["intent_specific"]["delivery_method"], "stable transformation")
        self.assertEqual(delivery_clarification["field"], "readout_assay")

    def test_crispr_gene_modification_waits_for_organism_before_queries(self):
        original_query = "Find protocols that can allow in-planta tests in which genes are modified."
        first_profile = build_experiment_profile(original_query)
        crispr_source = "\n".join([original_query, "CRISPR / genome editing"])

        intent = detect_experiment_intent(crispr_source)
        profile = build_experiment_profile(crispr_source, previous_profile=first_profile)
        intent, profile = validate_biology_profile(crispr_source, intent, profile)
        clarification = next_biology_clarification(profile, intent)
        queries = generate_candidate_search_queries(profile, fallback_query=crispr_source, max_queries=5)

        self.assertEqual(profile["sub_intent"], "genome_editing")
        self.assertEqual(profile["modification_type"], "CRISPR / genome editing")
        self.assertEqual(profile["tissue_or_cell_type"], "in planta / whole plant")
        self.assertIsNone(profile["organism"])
        self.assertEqual(clarification["field"], "organism")
        self.assertEqual(queries, [])

        rice_source = "\n".join([original_query, "rice"])
        rice_intent = detect_experiment_intent(rice_source)
        rice_profile = build_experiment_profile(rice_source, previous_profile=profile)
        rice_intent, rice_profile = validate_biology_profile(rice_source, rice_intent, rice_profile)
        rice_clarification = next_biology_clarification(rice_profile, rice_intent)

        self.assertEqual(rice_profile["organism"], "rice")
        self.assertEqual(rice_profile["modification_type"], "CRISPR / genome editing")
        self.assertEqual(rice_clarification["field"], "delivery_method")

    def test_overexpression_modification_type_preserves_specific_intent_after_organism_answer(self):
        original_query = "Find protocols that can allow in-planta tests in which genes are modified."
        first_profile = build_experiment_profile(original_query)

        overexpression_source = "\n".join([original_query, "overexpression"])
        overexpression_intent = detect_experiment_intent(overexpression_source)
        overexpression_profile = build_experiment_profile(overexpression_source, previous_profile=first_profile)
        overexpression_intent, overexpression_profile = validate_biology_profile(
            overexpression_source,
            overexpression_intent,
            overexpression_profile,
        )

        rice_source = "\n".join([original_query, "rice"])
        rice_intent = detect_experiment_intent(rice_source)
        rice_profile = build_experiment_profile(rice_source, previous_profile=overexpression_profile)
        rice_intent, rice_profile = validate_biology_profile(rice_source, rice_intent, rice_profile)

        self.assertEqual(rice_intent["intent"], "gene_overexpression")
        self.assertEqual(rice_intent["sub_intent"], "gene_overexpression")
        self.assertEqual(rice_profile["sub_intent"], "gene_overexpression")
        self.assertEqual(rice_profile["experimental_method"], "gene overexpression")
        self.assertEqual(rice_profile["method"], "gene overexpression")
        self.assertEqual(rice_profile["modification_type"], "overexpression")
        self.assertEqual(rice_profile["organism"], "rice")

    def test_stale_llm_modification_clarification_is_rejected_after_transient_overexpression(self):
        original_query = "Find protocols that can allow in-planta tests in which genes are modified."
        profile = build_experiment_profile(original_query)

        for answer in ["overexpression", "rice", "transient expression"]:
            source = "\n".join([original_query, answer])
            intent = detect_experiment_intent(source)
            profile = build_experiment_profile(source, previous_profile=profile)
            intent, profile = validate_biology_profile(source, intent, profile)

        clarification = next_clarification(profile, intent)
        stale_llm_plan = {
            "clarifying_question": {
                "field": "modification_type",
                "question": "What type of gene modification are you interested in?",
                "options": ["overexpression", "knockdown", "deletion", "insertion"],
            }
        }

        self.assertEqual(profile["sub_intent"], "gene_overexpression")
        self.assertEqual(profile["modification_type"], "overexpression")
        self.assertEqual(profile["organism"], "rice")
        self.assertEqual(profile["expression_type"], "transient expression")
        self.assertEqual(profile["delivery_method"], "transient expression")
        self.assertEqual(profile["tissue_or_cell_type"], "in planta / whole plant")
        self.assertEqual(clarification["field"], "readout_assay")
        self.assertIsNone(_clarification_from_plan(stale_llm_plan, profile))

    def test_rice_transient_overexpression_qpcr_queries_preserve_required_concepts(self):
        profile = {
            "intent_family": "gene_nucleic_acid_manipulation",
            "sub_intent": "gene_overexpression",
            "organism": "rice",
            "tissue_or_cell_type": "in planta / whole plant",
            "sample_type": "in planta / whole plant",
            "target": None,
            "gene_or_construct": None,
            "modification_type": "overexpression",
            "method": "gene overexpression",
            "experimental_method": "gene overexpression",
            "delivery_method": "transient expression",
            "expression_type": "transient expression",
            "readout": "RNA level / qPCR",
            "readout_assay": "RNA level / qPCR",
            "timeline": None,
            "equipment": [],
            "required_equipment": [],
            "difficulty": None,
            "protocol_difficulty": None,
            "constraints": [],
            "intent_specific": {"modification_type": "overexpression", "delivery_method": "transient expression"},
        }

        queries = generate_candidate_search_queries(profile, max_queries=5)

        self.assertGreaterEqual(len(queries), 3)
        for query in queries:
            text = query.lower()
            self.assertTrue("rice" in text or "oryza sativa" in text)
            self.assertIn("transient", text)
            self.assertTrue("overexpression" in text or "transgene expression" in text)
            self.assertTrue("qpcr" in text or "rt-qpcr" in text or "rna level" in text or "gene expression" in text)

    def test_rice_transient_overexpression_qpcr_ranking_penalizes_off_target_results(self):
        profile = {
            "organism": "rice",
            "tissue_or_cell_type": "in planta / whole plant",
            "modification_type": "overexpression",
            "experimental_method": "gene overexpression",
            "delivery_method": "transient expression",
            "expression_type": "transient expression",
            "readout_assay": "RNA level / qPCR",
            "required_equipment": [],
            "constraints": [],
        }
        results = [
            {
                "title": "Rice protein expression and purification",
                "description": "Recombinant protein expression and purification from rice samples.",
                "score": 9.0,
            },
            {
                "title": "Rice ribosome footprint library preparation",
                "description": "Ribosome footprinting and sequencing library prep for rice.",
                "score": 8.5,
            },
            {
                "title": "Rice metabolomics sample preparation",
                "description": "LC-MS metabolomics workflow for Oryza sativa tissues.",
                "score": 8.0,
            },
            {
                "title": "Potato transformation protocol",
                "description": "Stable potato transformation and regeneration workflow.",
                "score": 7.5,
            },
            {
                "title": "Rice transient overexpression qPCR protocol",
                "description": "In planta transient overexpression in Oryza sativa followed by RT-qPCR gene expression assay.",
                "score": 4.0,
            },
        ]

        ranked = apply_profile_ranking(profile, results, top_k=5)
        titles = [result["title"] for result in ranked]

        self.assertEqual(titles[0], "Rice transient overexpression qPCR protocol")
        self.assertLess(titles.index("Rice transient overexpression qPCR protocol"), titles.index("Rice protein expression and purification"))
        self.assertLess(titles.index("Rice transient overexpression qPCR protocol"), titles.index("Rice ribosome footprint library preparation"))
        self.assertLess(titles.index("Rice transient overexpression qPCR protocol"), titles.index("Rice metabolomics sample preparation"))
        self.assertLess(titles.index("Rice transient overexpression qPCR protocol"), titles.index("Potato transformation protocol"))


if __name__ == "__main__":
    unittest.main()
