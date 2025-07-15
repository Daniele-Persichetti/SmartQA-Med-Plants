# 📊 Data Format & Schema Documentation

## 📋 Table of Contents
- [Knowledge Graph Schema](#knowledge-graph-schema)
- [Plant Data Import Format](#plant-data-import-format)
- [Entity Types](#entity-types)
- [Relationship Mappings](#relationship-mappings)
- [Query Intent System](#query-intent-system)
- [Response Formatters](#response-formatters)

## 🗃️ Knowledge Graph Schema

### Node Types & Properties

```cypher
// Plant nodes - Core entities with comprehensive metadata
CREATE (p:Plant {
    id: STRING,              // Unique identifier (e.g., "P001")
    name: STRING,            // Common name (e.g., "Ginger")
    scientific_name: STRING, // Latin binomial (e.g., "Zingiber officinale")
    morphology: STRING,      // Physical description
    distribution_text: STRING // Geographic distribution info
})

// Taxonomic classification
CREATE (f:Family {
    name: STRING             // Family name (e.g., "Zingiberaceae")
})

// Active chemical compounds
CREATE (c:Compound {
    name: STRING             // Compound name (e.g., "gingerol")
})

// Therapeutic effects and uses
CREATE (e:TherapeuticEffect {
    name: STRING             // Effect description (e.g., "anti-inflammatory")
})

// Geographic regions
CREATE (r:Region {
    name: STRING             // Region name (e.g., "Southeast Asia")
})

// Traditional preparation methods
CREATE (m:PreparationMethod {
    name: STRING             // Method name (e.g., "infusion")
})

// Safety information
CREATE (s:SideEffect {
    name: STRING             // Side effect description
})

// Alternative common names
CREATE (cn:CommonName {
    name: STRING             // Alternative name
})
```

### Relationship Schema

```cypher
// Primary plant relationships
(Plant)-[:BELONGS_TO_FAMILY]->(Family)
(Plant)-[:CONTAINS]->(Compound)
(Plant)-[:PRODUCES_EFFECT]->(TherapeuticEffect)
(Plant)-[:GROWS_IN]->(Region)
(Plant)-[:PREPARED_BY]->(PreparationMethod)
(Plant)-[:MAY_CAUSE]->(SideEffect)
(Plant)-[:ALSO_KNOWN_AS]->(CommonName)

// Secondary cross-reference relationships
(Compound)-[:CONTRIBUTES_TO {evidence_count: INT}]->(TherapeuticEffect)
(Family)-[:ASSOCIATED_WITH_EFFECT {plant_count: INT}]->(TherapeuticEffect)
(Region)-[:MAY_SOURCE_COMPOUND {plant_count: INT}]->(Compound)
```

### Database Constraints

```cypher
// Unique constraints for data integrity
CREATE CONSTRAINT plant_name_unique FOR (p:Plant) REQUIRE p.name IS UNIQUE;
CREATE CONSTRAINT plant_sci_name_unique FOR (p:Plant) REQUIRE p.scientific_name IS UNIQUE;
CREATE CONSTRAINT family_name_unique FOR (f:Family) REQUIRE f.name IS UNIQUE;
CREATE CONSTRAINT compound_name_unique FOR (c:Compound) REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT effect_name_unique FOR (e:TherapeuticEffect) REQUIRE e.name IS UNIQUE;
CREATE CONSTRAINT method_name_unique FOR (m:PreparationMethod) REQUIRE m.name IS UNIQUE;
CREATE CONSTRAINT side_effect_name_unique FOR (s:SideEffect) REQUIRE s.name IS UNIQUE;
CREATE CONSTRAINT region_name_unique FOR (r:Region) REQUIRE r.name IS UNIQUE;
CREATE CONSTRAINT common_name_unique FOR (cn:CommonName) REQUIRE cn.name IS UNIQUE;
```

## 📄 Plant Data Import Format

### Structured Text Format (M.Plants.txt)

The system imports plant data from a specifically structured text file format:

```
===PLANT===
id: [unique_identifier]
name: [common_plant_name]
scientific_name: *[latin_binomial_name]*
family: [taxonomic_family]
common_names: [comma_separated_alternative_names]

===characteristics===
morphology: [physical_description]
distribution: [geographic_distribution_information]

===medicinal===
compounds: [comma_separated_active_compounds]
effects: [comma_separated_therapeutic_effects]

===usage===
methods: [comma_separated_preparation_methods]
side_effects: [safety_information_and_contraindications]

===END===
```

### Complete Example Entry

```
===PLANT===
id: P001
name: Ginger
scientific_name: *Zingiber officinale*
family: Zingiberaceae
common_names: ginger root, luyang dilaw, jengibre, fresh ginger

===characteristics===
morphology: Perennial herb with thick, fleshy, aromatic rhizomes that grow horizontally underground. Stems can reach 1 meter in height with narrow lanceolate leaves.
distribution: Native to Southeast Asia, now widely cultivated throughout tropical and subtropical regions including India, China, Jamaica, and Hawaii.

===medicinal===
compounds: gingerol, shogaol, zingerone, paradol, zingiberene
effects: anti-inflammatory, antiemetic, digestive aid, pain relief, immune-boosting, circulation enhancement

===usage===
methods: infusion, decoction, powder, tincture, fresh juice, capsules
side_effects: Mild heartburn or digestive upset in some individuals. Avoid high doses during pregnancy. May interact with blood-thinning medications.

===END===
```

### Data Processing Pipeline

1. **File Parsing**: Extract plant sections using `===PLANT===` delimiters
2. **Section Processing**: Parse characteristics, medicinal, and usage sections
3. **Entity Extraction**: Extract and normalize compounds, effects, methods
4. **Validation**: Ensure required fields (name, scientific_name) are present
5. **Relationship Creation**: Generate cross-references between related entities
6. **Quality Control**: Clean and validate extracted entities

## 🎯 Entity Types & Recognition

### Plant Entities
- **Base Plants**: Core set of 156+ well-known medicinal plants
- **Recognition**: Exact match, fuzzy matching, scientific name matching
- **Synonyms**: Common names, regional names, alternative spellings
- **Examples**: `ginger`, `Zingiber officinale`, `ginger root`

### Condition Entities  
- **Base Conditions**: 200+ health conditions and symptoms
- **Recognition**: Fuzzy matching with condition synonyms
- **Canonical Mapping**: Maps variations to standard terms
- **Examples**: `inflammation` ← `swelling`, `inflammatory`, `inflamed`

### Compound Entities
- **Base Compounds**: 300+ active chemical compounds
- **Recognition**: Exact matching with chemical name variations
- **Plant Linking**: Bidirectional compound-plant relationships
- **Examples**: `curcumin`, `gingerol`, `hypericin`

### Region Entities
- **Geographic Regions**: Continents, countries, traditional medicine regions
- **Recognition**: Broader matching with CONTAINS queries
- **Examples**: `Asia`, `Southeast Asia`, `Amazon`, `Mediterranean`

## 🧠 Query Intent System

### 15 Distinct Intents

| Intent | Description | Example Query | Database Pattern |
|--------|-------------|---------------|------------------|
| `plant_info` | General plant information | "Tell me about Ginger" | Single plant node with all relationships |
| `condition_plants` | Plants for specific conditions | "What herbs help with anxiety?" | Effect → Plants relationship |
| `multi_condition_plants` | Plants for multiple conditions | "Herbs for anxiety and insomnia?" | Intersection of effect-plant relationships |
| `similar_plants` | Plants with similar effects | "What's similar to Valerian?" | Shared effect relationships |
| `compound_effects` | Effects of compounds | "What does curcumin do?" | Compound → Effects relationship |
| `plant_compounds` | Compounds in plants | "What's in Turmeric?" | Plant → Compounds relationship |
| `compound_plants` | Plants with compounds | "Which plants have curcumin?" | Compound ← Plants relationship |
| `safety_info` | Safety information | "Is St. John's Wort safe?" | Plant → SideEffects relationship |
| `region_plants` | Regional plant queries | "Plants from Asia?" | Region ← Plants relationship |
| `region_condition_plants` | Regional plants for conditions | "Asian herbs for arthritis?" | Region ← Plants → Effects |
| `plant_preparation` | Plant preparation methods | "How to prepare Echinacea?" | Plant → Methods relationship |
| `preparation_for_condition` | Preparation for conditions | "How to prepare anxiety herbs?" | Effects ← Plants → Methods |
| `keyword_search` | General keyword search | "Natural remedies" | Cross-entity search |
| `general_query` | Broad informational queries | "What are adaptogens?" | Educational responses |
| `error` | Error handling | Invalid input | Error messaging |

### Intent Classification Logic

```python
def classify_question_intent(self, question: str) -> str:
    """
    Priority-based intent classification:
    1. Safety Intent (High Priority if Plant mentioned)
    2. Preparation Intent  
    3. Similarity Intent
    4. Multi-Constraint Intents (Region + Condition, Multiple Conditions)
    5. Compound-focused Intents
    6. Plant-focused Intents
    7. Condition-focused Intents
    8. Regional Queries
    9. Fallback Intents
    """
```

### Query Template Mapping

Each intent maps to a parameterized Cypher query template:

```cypher
-- plant_info template
MATCH (p:Plant)
WHERE toLower(p.name) = $norm_entity_name
   OR toLower(p.scientific_name) = $norm_entity_name
   OR $norm_entity_name IN [cn_name IN [(p)-[:ALSO_KNOWN_AS]->(cn:CommonName) | toLower(cn.name)] | cn_name]
WITH p LIMIT 1
OPTIONAL MATCH (p)-[:BELONGS_TO_FAMILY]->(f:Family)
OPTIONAL MATCH (p)-[:CONTAINS]->(c:Compound)
OPTIONAL MATCH (p)-[:PRODUCES_EFFECT]->(e:TherapeuticEffect)
OPTIONAL MATCH (p)-[:PREPARED_BY]->(m:PreparationMethod)
OPTIONAL MATCH (p)-[:MAY_CAUSE]->(s:SideEffect)
OPTIONAL MATCH (p)-[:ALSO_KNOWN_AS]->(cn:CommonName)
RETURN p.name as name, p.scientific_name as scientific_name,
       p.morphology as morphology, p.distribution_text as distribution,
       f.name as family, collect(DISTINCT c.name) as compounds,
       collect(DISTINCT e.name) as effects,
       collect(DISTINCT m.name) as preparations,
       collect(DISTINCT s.name) as side_effects,
       collect(DISTINCT cn.name) as common_names
```

## 📝 Response Formatters

### 12 Specialized Formatters

| Formatter | Intent(s) | Purpose | Output Structure |
|-----------|-----------|---------|------------------|
| `format_plant_info` | `plant_info` | Comprehensive plant profiles | Introduction → Description → Effects → Compounds → Preparations → Safety |
| `format_condition_plants` | `condition_plants` | Plants for conditions | Introduction → Primary Plants → Supportive Plants → Guidelines |
| `format_multi_condition_plants` | `multi_condition_plants` | Multi-condition plant matching | Introduction → Relevant Plants → Usage Notes |
| `format_similar_plants` | `similar_plants` | Plant alternatives | Introduction → Similar Plants → Differences → Guidelines |
| `format_safety_info` | `safety_info` | Safety profiles | Introduction → Side Effects → Contraindications → Guidelines |
| `format_compound_effects` | `compound_effects` | Compound information | Introduction → Effects → Sources → Context |
| `format_compound_plants` | `compound_plants` | Plant sources of compounds | Introduction → Plant List → Key Points → Usage |
| `format_plant_compounds` | `plant_compounds` | Compounds in plants | Introduction → Compound List → Context |
| `format_preparation_methods` | `plant_preparation`, `preparation_for_condition` | Preparation information | Introduction → Methods → Key Factors → Selection |
| `format_region_plants` | `region_plants` | Regional plant information | Introduction → Plant List → Context |
| `format_region_condition_plants` | `region_condition_plants` | Regional plants for conditions | Introduction → Relevant Plants → Context |
| `generate_general_explanation` | `keyword_search`, `general_query`, `error` | Fallback responses | Acknowledgment → Suggestions → Guidance |

### Response Structure Template

All formatters follow a consistent structure:

```markdown
**[Entity Name]** ([Scientific Name]) [Introduction]

**[Section Header]:**
• [Formatted list item]
• [Formatted list item]

**[Another Section]:**
[Descriptive text with embedded information]

**Important Note:** [Mandatory safety disclaimer]
```

### Safety Integration

Every response includes comprehensive safety information:

```python
def enhance_response_with_cautions(self, base_response: str) -> str:
    caution = """
    **Important Note:** Information provided is based on available data and is for 
    educational purposes only. It is not intended as medical advice. Always consult 
    with a qualified healthcare professional before using any medicinal plant, 
    especially if you have underlying health conditions, are pregnant or nursing, 
    are taking other medications, or considering combining treatments. Individual 
    responses can vary, and proper identification, dosage, and preparation are 
    crucial for safety and effectiveness.
    """
    return base_response.strip() + caution
```

## 🔧 Data Quality & Validation

### Import Validation Rules

1. **Required Fields**: `name` and `scientific_name` must be present
2. **Format Validation**: Scientific names should be in italic markers
3. **Entity Extraction**: Compounds and effects must be comma-separated
4. **Relationship Validation**: Cross-references must be valid
5. **Duplicate Detection**: Prevent duplicate plant entries

### Data Cleaning Pipeline

```python
def clean_name(name):
    """Standardize entity names."""
    if not name: return None
    name = name.strip().lower()
    name = re.sub(r'\s+', ' ', name)  # Consolidate whitespace
    name = name.rstrip('.,;')         # Remove trailing punctuation
    return name if len(name) > 1 else None
```

### Quality Metrics

- **Entity Coverage**: 156+ plants, 200+ conditions, 300+ compounds
- **Relationship Density**: Average 8-12 relationships per plant
- **Data Completeness**: 95%+ of plants have effects and compounds
- **Validation Success**: 98%+ of entries pass validation rules
