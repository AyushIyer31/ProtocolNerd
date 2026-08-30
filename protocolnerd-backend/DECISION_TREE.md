# Decision Tree: Rule-Based Intent Classification & Clarification

## 1. Intent Classification (Priority Order)

The rules are evaluated top-to-bottom. **First match wins.**

```
USER QUERY
│
├─ contains (drought|salt|heat|cold|stress) AND (tolerance|resistance|phenotype|test)?
│  YES ──► stress_tolerance_assay
│          family: phenotyping_physiology
│
├─ contains (more than one|multiple|simultaneously) AND (gene|target) AND (modify|edit|knock)?
│  YES ──► multiplex_gene_modification
│          family: gene_nucleic_acid_manipulation
│
├─ contains (overexpress|overexpression|forced expression|ectopic expression)?
│  YES ──► gene_overexpression
│          family: gene_nucleic_acid_manipulation
│
├─ contains (knockdown|silencing|siRNA|shRNA|RNAi|VIGS)?
│  YES ──► gene_knockdown
│          family: gene_nucleic_acid_manipulation
│
├─ contains (CRISPR|Cas9|Cas12|base editing|prime editing|gRNA)?
│  YES ──► genome_editing
│          family: gene_nucleic_acid_manipulation
│
├─ contains (gene modification|modify genes|modified genes|genome editing)?
│  YES ──► gene_modification          ◄── KEY RULE: "modified genes" ≠ overexpression
│          family: gene_nucleic_acid_manipulation
│
├─ LLM returned a valid sub_intent?
│  YES ──► normalize to controlled value
│
├─ profile has experimental_method set?
│  YES ──► derive sub_intent from method
│
└─ NONE matched
   ──► unknown
```

## 2. Overexpression Guard Rule

```
"modified genes" in query?
│
├─ YES ──► Does query also contain "overexpress" or "overexpression"?
│          │
│          ├─ YES ──► gene_overexpression  (explicit)
│          │
│          └─ NO  ──► gene_modification    (ambiguous — ask user)
│
└─ NO  ──► normal classification continues
```

## 3. Modification Type Upgrade

When a user answers the modification_type clarification, the sub_intent upgrades:

```
modification_type answer
│
├─ "CRISPR / genome editing"    ──► sub_intent = genome_editing
├─ "overexpression"             ──► sub_intent = gene_overexpression
├─ "knockdown / silencing"      ──► sub_intent = gene_knockdown
├─ "mutation / mutagenesis"     ──► sub_intent = mutagenesis
├─ "stable transformation"      ──► sub_intent = transformation
└─ "not sure"                   ──► sub_intent stays gene_modification
```

## 4. Organism System Classification

Used to filter organism and tissue clarification options.

```
Profile contains organism / tissue_or_cell_type / sample_type text
│
├─ plant terms (arabidopsis, rice, in planta, leaf, callus...)    ──► "plant"
├─ insect terms (drosophila, mosquito, silkworm...)               ──► "insect"
├─ fish terms (zebrafish, medaka...)                              ──► "fish"
├─ worm terms (c. elegans, nematode...)                           ──► "worm"
├─ amphibian terms (xenopus, frog, axolotl...)                    ──► "amphibian"
├─ mammalian in vivo (mouse, rat, in vivo...)                     ──► "mammalian_in_vivo"
├─ mammalian cells (human, HeLa, HEK293...)                      ──► "mammalian_cells"
├─ bacteria (e. coli, agrobacterium...)                           ──► "bacteria"
├─ yeast (saccharomyces, pichia...)                               ──► "yeast"
├─ fungi (aspergillus, neurospora...)                             ──► "fungi"
├─ algae (chlamydomonas, chlorella...)                            ──► "algae"
├─ cell-free (in vitro transcription, lysate...)                  ──► "cell_free"
└─ nothing matched                                                ──► "unknown"
```

## 5. Clarification Priority (per sub-intent)

### gene_modification / genome_editing

```
┌─────────────────────────────────────────────┐
│  1. modification_type missing?              │
│     → "What kind of gene modification?"     │
│     → [CRISPR, overexpression, knockdown,   │
│        mutagenesis, stable transformation]  │
│                                             │
│  2. organism missing?                       │
│     → context-aware organism options        │
│     (plant context → plant species only)    │
│                                             │
│  3. tissue_or_cell_type missing?            │
│     → context-aware system options          │
│     (plant → in planta/leaf/callus/...)     │
│                                             │
│  4. delivery_method missing?                │
│     → [stable, transient, Agrobacterium,    │
│        biolistic, not sure]                 │
│                                             │
│  5. readout_assay missing?                  │
│     → [phenotype, genotyping, qPCR,         │
│        protein level]                       │
└─────────────────────────────────────────────┘
```

### multiplex_gene_modification

```
┌─────────────────────────────────────────────┐
│  1. modification_type missing?              │
│  2. organism missing?                       │
│  3. target missing?                         │
│     → "What gene targets?"                  │
│     → [transcription factors, two named     │
│        genes, gene family, not sure]        │
│  4. tissue_or_cell_type missing?            │
│  5. delivery_method missing?                │
│  6. readout_assay missing?                  │
└─────────────────────────────────────────────┘
```

### stress_tolerance_assay

```
┌─────────────────────────────────────────────┐
│  1. organism missing?                       │
│     → context-aware (defaults to plant      │
│        options for stress queries)          │
│                                             │
│  2. growth_stage missing?                   │
│     → "What growth stage?"                  │
│     → [seedlings, adult plants, leaf, ...]  │
│                                             │
│  3. treatment_condition missing?            │
│     → "What stress treatment?"              │
│     → [drought stress, PEG/osmotic,         │
│        withholding water, not sure]         │
│                                             │
│  4. readout_assay missing?                  │
│     → "How should tolerance be measured?"   │
│     → [survival rate, growth/biomass,       │
│        water loss, phenotype score]         │
└─────────────────────────────────────────────┘
```

### protein_purification

```
┌─────────────────────────────────────────────┐
│  1. expression_host missing?                │
│     → [E. coli, yeast, mammalian, plant]    │
│                                             │
│  2. target missing?                         │
│     → [His-tagged, GST-tagged, native, ...] │
│                                             │
│  3. purification_method missing?            │
│     → [His-tag affinity, GST, native, ...]  │
└─────────────────────────────────────────────┘
```

## 6. Search Gate (can_generate_search_queries)

Search queries are **blocked** until critical fields are filled.

```
sub_intent                        required fields
─────────────────────────────────────────────────────────────────
gene_modification / genome_editing  modification_type + organism + tissue
multiplex_gene_modification         modification_type + organism + tissue + target
stress_tolerance_assay              organism + stress_type + (growth_stage OR tissue OR readout)
protein_purification                target OR protein_source OR expression_host OR organism
gene_overexpression                 organism + expression_type + tissue
general / other                     method OR target
unknown / chitchat                  ALWAYS BLOCKED
```

## 7. Full Request Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER SENDS QUERY                          │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  1. LLM (Ollama) attempts structured JSON extraction             │
│     → intent, experiment_profile, clarifying_question            │
│     → BEST EFFORT ONLY — may fail, may hallucinate              │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. Rule-based validation (validate_biology_profile)             │
│     → Regex intent classification (priority chain from §1)       │
│     → Overexpression guard (§2)                                  │
│     → Extract organism, target, tissue, stress, multiplex        │
│     → Override bad LLM output with controlled values             │
│     → Assign intent_family + sub_intent                          │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  3. Is this chitchat?                                            │
│     └─ YES → return chitchat reply, no search                    │
└──────────────────────┬───────────────────────────────────────────┘
                       │ NO
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  4. Clarification check (next_biology_clarification)             │
│     → Walk the priority list for current sub_intent (§5)         │
│     → First missing required field → return clarification Q      │
│     └─ HAS MISSING FIELD → return question + options to user     │
└──────────────────────┬───────────────────────────────────────────┘
                       │ ALL REQUIRED FILLED
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  5. Search gate check (can_generate_search_queries) (§6)         │
│     └─ NOT READY → return clarification for next missing field   │
└──────────────────────┬───────────────────────────────────────────┘
                       │ READY
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  6. Generate candidate search queries                            │
│     → LLM suggests queries (filtered for concept preservation)   │
│     → Rule-based templates generate backup queries               │
│     → Merge + dedup → max 5 candidates                           │
│     → Return to user: "Select, edit, or search all"              │
└──────────────────────┬───────────────────────────────────────────┘
                       │ USER CONFIRMS
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  7. Execute search                                               │
│     → Live mode: concept expansion → protocols.io API            │
│     → Local mode: TF-IDF over cached index                       │
│     → Profile-aware ranking (boost required concept matches)     │
│     → False-positive penalties (wrong organism, TEV, etc.)       │
│     → Return top-K ranked results + explanations                 │
└──────────────────────────────────────────────────────────────────┘
```

## 8. Example Traces

### "Find protocols for in-planta tests in which genes are modified"

```
Step 1: LLM → may return gene_overexpression (WRONG)
Step 2: Regex → "genes are modified" matches, no "overexpress" → gene_modification ✓
         "in planta" → tissue = "in planta / whole plant"
Step 3: Not chitchat
Step 4: modification_type = null → ASK "What kind of gene modification?"
Step 5: (blocked)
Step 6: (blocked)
```

### "More than one transcription factor modified at the same time for mice"

```
Step 1: LLM → may return gene_modification
Step 2: Regex → "more than one" + "transcription factor" + "modified" → multiplex ✓
         "mice" → organism = mouse
Step 3: Not chitchat
Step 4: modification_type = null → ASK "What kind of gene modification?"
Step 5: (blocked — needs modification_type + organism + tissue + target)
Step 6: (blocked)
```

### "Find protocols that test for drought tolerance in Rice"

```
Step 1: LLM → may return unknown or phenotyping
Step 2: Regex → "drought" + "tolerance" → stress_tolerance_assay ✓
         "rice" → organism = rice
         stress_type = drought, readout = "drought tolerance / phenotype"
Step 3: Not chitchat
Step 4: organism ✓, stress_type ✓ → growth_stage missing → ASK "What growth stage?"
        OR if readout is enough → proceed to queries
Step 5: Ready (organism + stress_type + readout all present)
Step 6: Generate: "rice drought tolerance phenotyping", "Oryza sativa drought stress assay", ...
```
