import torch
import re
import logging
import pickle
import os
from thefuzz import process, fuzz # Use thefuzz instead of fuzzywuzzy
import numpy as np
from typing import List, Dict, Optional, Set, Any, Tuple, Union
from .knowledge_graph_entity_linker import KnowledgeGraphEntityLinker 


import torch.nn as nn
import nltk
# Suppress specific NLTK warnings if needed, although download handling is better
# warnings.filterwarnings("ignore", category=nltk.downloader.DownloadWarning)
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Configure logging
logger = logging.getLogger("BertProcessor")
# Ensure logger is configured (this might be handled globally elsewhere)
if not logger.hasHandlers():
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    # Ensure basicConfig is called only once in the application
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        # If root logger is already configured, just set the level for this logger
        logger.setLevel(log_level)


# Attempt to load config, handle potential errors gracefully
try:
    from config import config
except ImportError:
    logging.warning("Could not import 'config'. Using default configurations.")
    config = {} # Use an empty dict as a fallback


class BertProcessor:
    """
    Uses BERT for Natural Language Understanding tasks like intent classification
    and entity extraction, enhanced with Knowledge Graph entity lists and robust matching.
    (Improved Synonym Handling, Intent Logic, Normalization)
    """
    def __init__(self, kg_embeddings_dir='models/kg_embeddings_trained'):
        logger.info("Initializing BertProcessor (v5 - Improved)...")
        self.kg_embeddings_dir = kg_embeddings_dir
        # Use config object safely with .get()
        self.config = config if isinstance(config, dict) else {}

        self._initialize_entity_lists() # Load base lists first
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self._load_bert_model()
        self._load_kg_data() # Load KG data and update entity lists

        # Projection layer (optional, requires trained weights)
        self.bert_to_kg_proj = None
        if self.model: # Only initialize if BERT loaded
            self.bert_embedding_dim = self.model.config.hidden_size # Typically 768 for bert-base
            # KG embedding dim needs to be known. Default or load from embeddings.
            self.kg_embedding_dim = 100 # Default, update if loading embeddings
            if self.entity_embeddings is not None:
                # Ensure entity_embeddings is a tensor or ndarray before accessing shape
                if hasattr(self.entity_embeddings, 'shape'):
                     try:
                          self.kg_embedding_dim = self.entity_embeddings.shape[1]
                     except IndexError:
                          logger.warning("Could not determine KG embedding dimension from loaded embeddings. Using default 100.")
                          self.kg_embedding_dim = 100
                else:
                     logger.warning("Loaded entity embeddings object has no shape attribute. Using default KG dim 100.")
                     self.kg_embedding_dim = 100


            try:
                self.bert_to_kg_proj = nn.Linear(self.bert_embedding_dim, self.kg_embedding_dim)
                self.bert_to_kg_proj.to(self.device)
                logger.info(f"Initialized projection layer: {self.bert_embedding_dim} -> {self.kg_embedding_dim}")
                # Attempt to load pre-trained weights if available
                self.load_projection_layer()
            except Exception as e:
                logger.error(f"Failed to initialize projection layer: {e}", exc_info=True)
                self.bert_to_kg_proj = None


        self.fuzzy_match_threshold = self.config.get("fuzzy_threshold", 80) # Configurable threshold
        self.region_fuzzy_match_threshold = self.config.get("region_fuzzy_threshold", 75) # Lower threshold for regions
        self._initialize_query_templates()

        # Initialize Lemmatizer
        self.lemmatizer = None
        self._initialize_lemmatizer()

        logger.info("BertProcessor (v5 - Improved) initialized successfully.")

    def _initialize_lemmatizer(self):
        """Initializes the NLTK lemmatizer, attempting downloads if necessary."""
        try:
            # Check if data exists before attempting to use Lemmatizer
            nltk.data.find('corpora/wordnet.zip')
            nltk.data.find('corpora/omw-1.4.zip')
            self.lemmatizer = WordNetLemmatizer()
            # Test lemmatization
            _ = self.lemmatizer.lemmatize('testing')
            logger.info("NLTK Lemmatizer initialized successfully.")
        except LookupError:
            logger.warning("NLTK WordNet data not found. Attempting download...")
            try:
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
                self.lemmatizer = WordNetLemmatizer()
                 # Test again after download
                _ = self.lemmatizer.lemmatize('testing')
                logger.info("NLTK WordNet data downloaded successfully. Lemmatizer initialized.")
            except Exception as e:
                logger.error(f"Failed to download NLTK data: {e}. Lemmatization will be disabled.", exc_info=True)
                self.lemmatizer = None
        except Exception as e:
            logger.error(f"Error initializing lemmatizer: {e}. Lemmatization will be disabled.", exc_info=True)
            self.lemmatizer = None

    def _initialize_entity_lists(self):
        """Initializes base entity lists and KG-related attributes."""
        logger.debug("Initializing base entity lists.")
        # Base lists provide a fallback if KG loading fails
        self.base_conditions = {'inflammation', 'pain', 'arthritis', 'anxiety', 'insomnia', 'diabetes', 'depression', 'hypertension', 'infection', 'stress', 'digestive issues', 'digestion', 'respiratory problems', 'immune support', 'cold', 'flu', 'headache', 'migraine', 'nausea', 'vomiting', 'burns', 'wounds', 'cuts', 'joint pain', 'eczema', 'psoriasis', 'allergies', 'fatigue', 'memory issues', 'cough', 'fever', 'cystitis', 'diarrhea', 'constipation', 'spasm', 'ulcer', 'catarrh', 'indigestion', 'dyspepsia', 'swelling', 'ache', 'sleep disorder', 'stomach issues', 'high blood pressure', 'skin conditions', 'mood disorder', 'nerve pain'}
        self.base_plants = {'ginger', 'turmeric', 'garlic', 'echinacea', 'ginseng', 'aloe vera', 'chamomile', 'lavender', 'peppermint', 'valerian', 'st johns wort', 'st. johns wort', 'st. john\'s wort', 'milk thistle', 'ginkgo biloba', 'holy basil', 'ashwagandha', 'maca', 'elderberry', 'calendula', 'arnica', 'passionflower', 'lemon balm', 'kava', 'rosemary', 'thyme', 'sage', 'oregano', 'cinnamon', 'nettle', 'boldo', 'dandelion', 'burdock', 'licorice root', 'yarrow', 'feverfew', 'gotu kola', 'devil\'s claw', 'boswellia', 'cat\'s claw', 'balm of gilead'}
        self.base_compounds = {'curcumin', 'allicin', 'gingerol', 'menthol', 'quercetin', 'hypericin', 'anthocyanins', 'polyphenols', 'alkaloids', 'flavonoids', 'tannins', 'terpenes', 'saponins', 'berberine', 'silymarin', 'salicylic acid', 'ginkgolides', 'apigenin', 'chamazulene', 'eugenol', 'thymol', 'curcuminoids', 'glycosides', 'acids', 'sterols', 'volatile oil', 'essential oil', 'lignans', 'resins', 'mucilage', 'oleanolic acid', 'ursolic acid'}
        self.base_regions = {'europe', 'asia', 'africa', 'north america', 'south america', 'central america', 'mediterranean', 'india', 'china', 'amazon', 'andes', 'himalayas', 'australia', 'middle east', 'southeast asia', 'siberia', 'japan', 'korea', 'arabian peninsula', 'andean region'}

        # Initialize KG-related attributes (will be populated by _load_kg_data)
        self.entity_to_idx: Dict[str, int] = {}
        self.idx_to_entity: Dict[int, str] = {}
        self.all_kg_entities: List[str] = []
        self.entity_embeddings: Optional[Union[torch.Tensor, np.ndarray]] = None # Store loaded embeddings
        self.kg_entity_linker: Optional[KnowledgeGraphEntityLinker] = None

        # Combined sets for efficient lookup (start with base, update with KG)
        self.all_known_plants: Set[str] = self.base_plants.copy()
        self.all_known_conditions: Set[str] = self.base_conditions.copy()
        self.all_known_compounds: Set[str] = self.base_compounds.copy()
        self.all_known_regions: Set[str] = self.base_regions.copy()
        self.all_entities_set: Set[str] = self.all_known_plants.union(self.all_known_conditions).union(self.all_known_compounds).union(self.all_known_regions)

    def _load_bert_model(self):
        """Loads the BERT model and tokenizer."""
        try:
            from transformers import BertTokenizer, BertModel
            # Use config object safely
            model_name = self.config.get("bert_model_name", 'bert-base-uncased')
            logger.info(f"Loading BERT model: {model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval() # Set model to evaluation mode
            logger.info(f"BERT model '{model_name}' loaded successfully on {self.device.upper()}")
        except ImportError:
            logger.error("transformers library not found. BERT functionality will be disabled. Install: pip install transformers torch")
            self.tokenizer = None
            self.model = None
        except Exception as e:
            logger.error(f"Error loading BERT model '{model_name}': {e}", exc_info=True)
            self.tokenizer = None
            self.model = None

    def _load_kg_data(self):
        """Loads KG entity mappings and embeddings, then updates the combined entity sets."""
        kg_mappings_path = os.path.join(self.kg_embeddings_dir, 'mappings.pkl')
        kg_model_path = os.path.join(self.kg_embeddings_dir, 'model.pt') # Embeddings file

        if not os.path.exists(kg_mappings_path):
            logger.warning(f"KG mappings file not found: {kg_mappings_path}. Entity linking and KG-based entity lists will be limited.")
            return # Keep base lists if KG data is missing

        try:
            with open(kg_mappings_path, 'rb') as f:
                mappings = pickle.load(f)

            if not isinstance(mappings, dict) or not all(k in mappings for k in ['entity_to_idx', 'idx_to_entity']):
                logger.warning(f"KG mappings file '{kg_mappings_path}' has incorrect format or missing keys. Expected dict with 'entity_to_idx' and 'idx_to_entity'.")
                return

            self.entity_to_idx = mappings['entity_to_idx']
            self.idx_to_entity = mappings['idx_to_entity']
            # Ensure they are the correct types
            if not isinstance(self.entity_to_idx, dict) or not isinstance(self.idx_to_entity, dict):
                 logger.error("Loaded mappings are not dictionaries. Aborting KG data load.")
                 self.entity_to_idx = {}; self.idx_to_entity = {}; return

            # Filter raw keys before creating all_kg_entities list
            raw_keys = list(self.entity_to_idx.keys())
            self.all_kg_entities = [k for k in raw_keys if isinstance(k, str) and k.strip()] # Ensure keys are non-empty strings
            logger.info(f"Loaded {len(self.all_kg_entities)} valid string entities from KG mappings: {kg_mappings_path}")


            # --- Refine KG Entity Lists ---
            # Define problematic terms to exclude (more specific)
            problematic_terms = {
                 "side effects", "unknown", "various", "general", "etc", "information",
                 "properties", "benefits", "uses", "effects", "preparation", "treatment",
                 "medicine", "herb", "plant", "compound", "region", "condition",
                 "description", "details", "summary", "overview", "example", "list",
                 "type", "potent anti-inflammatory", "multiple", "other", "various types",
                 "health", "body", "system", "drug", "medication", "therapy", "remedy",
                 "toxicity", "safety", "contraindication", "interaction", "dosage",
                 "method", "technique", "process", "extract", "powder", "oil", # Keep specific oils like 'essential oil'
                 "root", "leaf", "flower", "seed", "bark", "berry", # Keep specific parts if needed elsewhere, filter here
                 # Add more generic or non-identifying terms found during testing
                 "active ingredient", "constituent", "chemical", "substance", "molecule",
                 "disease", "illness", "ailment", "symptom", "disorder", "pain", # Keep specific pains like 'joint pain'
                 "inflammation", # Keep specific inflammations if needed, filter here
                 "infection", "syndrome", "health issue", "problem",
                 "relief", "support", "aids", "helps",
                 "area", "country", "continent", "location", "native", "origin",
                 "grows in", "found in", "habitat", "zone", "climate", "environment"
            }
            # Keywords for classification (can be refined)
            plant_kws = {'herb', 'root', 'leaf', 'flower', 'seed', 'bark', 'berry', 'weed', 'tree', 'mushroom', 'fungus', 'vine', 'shrub', 'bean', 'nut', 'grass', 'rhizome', 'alga'}
            condition_kws = {'disease', 'illness', 'ailment', 'symptom', 'disorder', 'pain', 'inflammation', 'infection', 'syndrome', 'health issue', 'problem', 'relief', 'treatment', 'remedy', 'therapy', 'support', 'benefit', 'aid', 'help'} # Added benefit/aid/help
            compound_kws = {'compound', 'chemical', 'substance', 'molecule', 'alkaloid', 'flavonoid', 'terpene', 'glycoside', 'acid', 'oil', 'extract', 'ingredient', 'constituent', 'vitamin', 'mineral'}
            region_kws = {'region', 'area', 'country', 'continent', 'location', 'native', 'origin', 'habitat', 'zone', 'climate', 'environment', 'peninsula', 'mountains'} # Added peninsula, mountains

            kg_plants, kg_conditions, kg_compounds, kg_regions = set(), set(), set(), set()

            for entity in self.all_kg_entities:
                # Already checked entity is a non-empty string
                e_lower = entity.lower().strip()

                # Skip clearly problematic terms
                if e_lower in problematic_terms: continue
                # Skip very short terms (likely noise)
                if len(e_lower) <= 2: continue

                # Simple classification (can be improved with more rules/ML)
                is_plant = any(kw in e_lower for kw in plant_kws)
                is_condition = any(kw in e_lower for kw in condition_kws) and not is_plant # Avoid classifying plants as conditions
                is_compound = any(kw in e_lower for kw in compound_kws) and not is_plant and not is_condition
                is_region = any(kw in e_lower for kw in region_kws) and not is_plant and not is_condition and not is_compound

                # Assign based on simple rules (prioritize non-overlapping)
                # Basic length check added
                if is_plant and len(e_lower) > 3: kg_plants.add(entity)
                elif is_condition and len(e_lower) > 3: kg_conditions.add(entity)
                elif is_compound and len(e_lower) > 3: kg_compounds.add(entity)
                elif is_region and len(e_lower) > 3: kg_regions.add(entity)
                # else: logger.debug(f"Entity '{entity}' not classified.") # Log unclassified

            # Update the main sets by adding *new* KG entities
            new_plants = kg_plants - self.all_known_plants
            new_conditions = kg_conditions - self.all_known_conditions
            new_compounds = kg_compounds - self.all_known_compounds
            new_regions = kg_regions - self.all_known_regions

            self.all_known_plants.update(new_plants)
            self.all_known_conditions.update(new_conditions)
            self.all_known_compounds.update(new_compounds)
            self.all_known_regions.update(new_regions)

            logger.info(f"Added {len(new_plants)} plants, {len(new_conditions)} conditions, {len(new_compounds)} compounds, {len(new_regions)} regions from KG.")
            logger.info(f"Final Known Entity Counts: P({len(self.all_known_plants)}), C({len(self.all_known_conditions)}), O({len(self.all_known_compounds)}), R({len(self.all_known_regions)})")

            # Update the combined set
            self.all_entities_set = self.all_known_plants.union(self.all_known_conditions).union(self.all_known_compounds).union(self.all_known_regions)
            logger.info(f"Updated combined entity set size: {len(self.all_entities_set)}")

            # --- Load Embeddings ---
            if os.path.exists(kg_model_path):
                try:
                    map_loc = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                    # Use weights_only=True for security if loading untrusted files
                    state_dict = torch.load(kg_model_path, map_location=map_loc, weights_only=True)

                    loaded_embeddings = None
                    if isinstance(state_dict, dict) and 'entity_embeddings.weight' in state_dict:
                        loaded_embeddings = state_dict['entity_embeddings.weight']
                        logger.info(f"Loaded KG entity embeddings from state_dict key: {loaded_embeddings.shape}")
                    elif isinstance(state_dict, torch.Tensor): # Handle case where the file *is* the tensor
                         loaded_embeddings = state_dict
                         logger.info(f"Loaded KG entity embeddings directly from tensor file: {loaded_embeddings.shape}")
                    else:
                        logger.warning(f"Embeddings key 'entity_embeddings.weight' not found in {kg_model_path} or file is not a tensor.")

                    if loaded_embeddings is not None:
                         # Simple sanity check on dimensions (more robust checks needed if sizes differ)
                         if loaded_embeddings.ndim == 2:
                              self.entity_embeddings = loaded_embeddings
                              self.kg_embedding_dim = self.entity_embeddings.shape[1] # Update KG dim
                              logger.info(f"KG embedding dimension set to: {self.kg_embedding_dim}")
                              # Verify shape consistency (optional, log warning if mismatch)
                              # Note: Using len(self.entity_to_idx) might be more accurate than len(self.all_kg_entities) if filtering occurred
                              if self.entity_embeddings.shape[0] != len(self.entity_to_idx):
                                   logger.warning(f"Embedding shape[0] ({self.entity_embeddings.shape[0]}) mismatch with entity_to_idx size ({len(self.entity_to_idx)}). Linking might be unreliable.")
                         else:
                              logger.error(f"Loaded embeddings have incorrect dimensions: {loaded_embeddings.ndim}. Expected 2D tensor. Disabling embeddings.")
                              self.entity_embeddings = None

                except Exception as e:
                    logger.warning(f"Could not load or process KG embeddings from {kg_model_path}: {e}", exc_info=True)
                    self.entity_embeddings = None
            else:
                logger.info(f"KG embedding file not found: {kg_model_path}. Entity linking will be disabled.")
                self.entity_embeddings = None

            # --- Initialize Entity Linker ---
            if KnowledgeGraphEntityLinker and self.entity_embeddings is not None and self.entity_to_idx and self.idx_to_entity:
                 # Check if entity_embeddings is valid before proceeding
                 if not hasattr(self.entity_embeddings, 'shape') or self.entity_embeddings.ndim != 2:
                      logger.error("Cannot initialize Entity Linker: Invalid entity embeddings object.")
                      self.kg_entity_linker = None
                 else:
                      try:
                           # Ensure embeddings are numpy for the linker
                           embeddings_np = self.entity_embeddings.cpu().numpy() if isinstance(self.entity_embeddings, torch.Tensor) else np.array(self.entity_embeddings)

                           # Calculate OOV embedding safely
                           oov_emb = np.zeros(self.kg_embedding_dim) # Default OOV
                           if embeddings_np.shape[0] > 0:
                                try:
                                     oov_emb = np.mean(embeddings_np, axis=0)
                                except Exception as mean_e:
                                     logger.error(f"Error calculating mean OOV embedding: {mean_e}. Using zeros.", exc_info=True)


                           self.kg_entity_linker = KnowledgeGraphEntityLinker(
                                entity_embeddings=embeddings_np,
                                entity_to_idx=self.entity_to_idx,
                                idx_to_entity=self.idx_to_entity,
                                oov_embedding=oov_emb,
                                similarity_threshold=self.config.get("kg_linker_threshold", 0.73) # Configurable threshold
                           )
                           logger.info(f"KnowledgeGraphEntityLinker initialized successfully with threshold {self.kg_entity_linker.similarity_threshold}.")
                      except Exception as e:
                           logger.error(f"Failed to initialize KnowledgeGraphEntityLinker: {e}", exc_info=True)
                           self.kg_entity_linker = None
            else:
                 logger.warning("Entity linker prerequisites (Linker class, embeddings, mappings) not met. Linker disabled.")
                 self.kg_entity_linker = None


        except FileNotFoundError:
             logger.warning(f"KG mappings file not found at {kg_mappings_path}. Keeping base entity lists.")
        except Exception as e:
            logger.error(f"Error loading KG data from {self.kg_embeddings_dir}: {e}", exc_info=True)
            # Reset to base lists in case of partial load failure
            self._initialize_entity_lists()


    def _clean_question_text(self, text: str) -> str:
        """Cleans text for keyword extraction, intent checks, and entity matching."""
        if not text or not isinstance(text, str):
            return ""
        text = text.lower()
        # More comprehensive list of prefixes/phrases to remove
        prefixes = [
            "tell me everything about", "tell me all about", "can you tell me about", "i want to know about", "what do you know about",
            "what are the effects of", "what are the benefits of", "what are the uses of", "what are the properties of",
            "what are the side effects of", "is it safe to use", "is it safe for", "is it safe", "is",
            "tell me about the effects of", "tell me about the benefits of", "tell me about the uses of",
            "tell me about the properties of", "tell me about the side effects of", "what are the", "what is the",
            "what about", "what is", "what are", "what", "tell me about", "tell me", "describe", "information on", "info on",
            "info about", "benefits of", "uses for", "uses of", "properties of", "how to prepare", "how to make", "how to use",
            "how do i prepare", "how do i make", "how do i use", "how should i", "which", "where", "when", "list",
            "find", "give me", "compare", "show me", "are there any", "are there", "can you", "is it", "does it",
            "do you know", "is there", "i need", "i want", "looking for",
            "medicinal plants for", "herbs for", "plants for", "remedies for", "treatment for",
            "please", "can you", "could you", "would you", # Polite requests
        ]
        # Sort by length descending to match longer phrases first
        for prefix in sorted(prefixes, key=len, reverse=True):
            # Use word boundary to avoid partial matches within words
            if text.startswith(prefix + ' '):
                text = text[len(prefix):].strip()
                break # Stop after first match

        # Remove trailing punctuation and politeness
        text = re.sub(r'[?.,!;:]+$', '', text).strip()
        text = re.sub(r'\s+(please|thank you|thanks)$', '', text).strip()

        # Remove special characters except hyphens and apostrophes within words
        # Ensure hyphens/apostrophes are kept correctly (e.g., St. John's Wort -> st john's wort)
        text = re.sub(r"(?<!\w)['\"]|['\"](?!\w)|[^\w\s\-\']", "", text) # Keep internal hyphens/apostrophes

        # Consolidate whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _initialize_query_templates(self):
        """Initializes Neo4j Cypher query templates. (v5 - Checked and refined)"""
        logger.debug("Initializing Cypher query templates.")
        # Using more robust matching (toLower) and parameterization
        # Added LIMIT clauses to prevent excessive results
        # Added checks for common names using relationship OR list check
        self.query_templates = {
             # --- Plant Information ---
             'plant_info': """
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
                RETURN p.name as name, 
                    p.scientific_name as scientific_name,
                    p.morphology as morphology, 
                    p.distribution_text as distribution,
                    f.name as family,
                    collect(DISTINCT c.name) as compounds,
                    collect(DISTINCT e.name) as effects,
                    collect(DISTINCT m.name) as preparations,
                    collect(DISTINCT s.name) as side_effects,
                    collect(DISTINCT cn.name) as common_names
             """,
             # --- Plants for a Condition ---
             'condition_plants': """
                MATCH (e:TherapeuticEffect)<-[:PRODUCES_EFFECT]-(p:Plant)
                WHERE toLower(e.name) CONTAINS $norm_entity_name
                OR size([syn IN $synonyms WHERE toLower(e.name) CONTAINS toLower(syn)]) > 0
                WITH p, collect(DISTINCT e.name) AS matched_effects
                OPTIONAL MATCH (p)-[:MAY_CAUSE]->(s:SideEffect)
                RETURN p.name AS p_name,
                    p.scientific_name AS p_scientific_name,
                    matched_effects AS effects,
                    collect(DISTINCT s.name) as side_effects
                ORDER BY p.name
                LIMIT 20
             """,
             # --- Plants for Multiple Conditions ---
             'multi_condition_plants': """
                // Match plants associated with the first condition or its synonyms
                MATCH (p:Plant)-[:PRODUCES_EFFECT]->(e1:TherapeuticEffect)
                WHERE toLower(e1.name) CONTAINS $norm_condition1
                   OR size([s1 IN $synonyms1 WHERE toLower(e1.name) CONTAINS toLower(s1)]) > 0
                WITH p

                // Ensure the SAME plant also matches the second condition or its synonyms
                MATCH (p)-[:PRODUCES_EFFECT]->(e2:TherapeuticEffect)
                WHERE toLower(e2.name) CONTAINS $norm_condition2
                   OR size([s2 IN $synonyms2 WHERE toLower(e2.name) CONTAINS toLower(s2)]) > 0

                // Collect all effects for plants matching BOTH conditions
                WITH p
                MATCH (p)-[:PRODUCES_EFFECT]->(e:TherapeuticEffect)
                WITH p, collect(DISTINCT e.name) AS all_effects

                RETURN p.name AS p_name,
                       p.scientific_name AS p_scientific_name,
                       // Filter effects to show relevance to the queried conditions
                       [effect IN all_effects WHERE
                           effect IS NOT NULL AND // Ensure not null before lowercasing
                           (toLower(effect) CONTAINS $norm_condition1 OR size([s1 IN $synonyms1 WHERE toLower(effect) CONTAINS toLower(s1)]) > 0 OR
                            toLower(effect) CONTAINS $norm_condition2 OR size([s2 IN $synonyms2 WHERE toLower(effect) CONTAINS toLower(s2)]) > 0)
                       ] AS effects
                ORDER BY p.name
                LIMIT 15
             """,
             # --- Plants Similar to a Target Plant ---
             'similar_plants': """
                MATCH (p1:Plant)-[:PRODUCES_EFFECT]->(e:TherapeuticEffect)<-[:PRODUCES_EFFECT]-(p2:Plant)
                WHERE toLower(p1.name) = $norm_entity_name 
                AND p1 <> p2
                WITH p2, e, count(e) as similarity_score
                WHERE similarity_score >= 2
                OPTIONAL MATCH (p2)-[:CONTAINS]->(c:Compound)
                RETURN p2.name as p2_name,
                    p2.scientific_name as p2_scientific_name,
                    collect(DISTINCT e.name) as shared_effects,
                    collect(DISTINCT c.name) as compounds,
                    similarity_score
                ORDER BY similarity_score DESC, p2.name
                LIMIT 10
             """,
             # --- Compounds in a Plant ---
             'plant_compounds': """
                 MATCH (p:Plant)-[:CONTAINS]->(c:Compound)
                 WHERE toLower(p.name) = $norm_entity_name
                    OR toLower(p.scientific_name) = $norm_entity_name
                    OR $norm_entity_name IN [cn_name IN [(p)-[:ALSO_KNOWN_AS]->(cn:CommonName) | toLower(cn.name)] | cn_name]
                 WITH c // Process each compound found
                 OPTIONAL MATCH (c)-[:CONTRIBUTES_TO]->(e:TherapeuticEffect)
                 RETURN c.name as compound_name,
                        collect(DISTINCT e.name) as associated_effects // Collect effects for this compound
                 ORDER BY compound_name
                 LIMIT 25 // Limit number of compounds shown
             """,
             # --- Preparation Methods for a Condition ---
             'preparation_for_condition': """
                MATCH (e:TherapeuticEffect)<-[:PRODUCES_EFFECT]-(p:Plant)
                WHERE toLower(e.name) CONTAINS $norm_entity_name
                MATCH (p)-[:PREPARED_BY]->(m:PreparationMethod)
                WITH m, collect(DISTINCT p.name) as example_plants, count(DISTINCT p) as plant_count
                RETURN m.name as preparation_method,
                    plant_count,
                    example_plants[..5] as example_plants
                ORDER BY plant_count DESC, preparation_method
                LIMIT 10
             """,
             # --- Preparation Methods for a Plant ---
             'plant_preparation': """
                 MATCH (p:Plant)-[:PREPARED_BY]->(m:PreparationMethod)
                 WHERE toLower(p.name) = $norm_entity_name
                    OR toLower(p.scientific_name) = $norm_entity_name
                    OR $norm_entity_name IN [cn_name IN [(p)-[:ALSO_KNOWN_AS]->(cn:CommonName) | toLower(cn.name)] | cn_name]
                 RETURN DISTINCT m.name as preparation_method // Return unique method names
                 // Optional: Add description: m.description as description
                 ORDER BY preparation_method
                 LIMIT 10
             """,
             # --- Plants in a Region ---
             'region_plants': """
                MATCH (r:Region)<-[:GROWS_IN]-(p:Plant)
                WHERE toLower(r.name) CONTAINS $norm_entity_name
                WITH p
                OPTIONAL MATCH (p)-[:PRODUCES_EFFECT]->(e:TherapeuticEffect)
                OPTIONAL MATCH (p)-[:CONTAINS]->(c:Compound)
                RETURN p.name as plant_name,
                    p.scientific_name as scientific_name,
                    collect(DISTINCT e.name)[..5] as effects,
                    collect(DISTINCT c.name)[..3] as compounds
                ORDER BY plant_name
                LIMIT 20
             """,
             # --- Plants in a Region for a Condition ---
             'region_condition_plants': """
                MATCH (r:Region)<-[:GROWS_IN]-(p:Plant)-[:PRODUCES_EFFECT]->(e:TherapeuticEffect)
                WHERE toLower(r.name) CONTAINS $norm_region_name // Match region
                   AND ( // Match condition or synonyms
                       toLower(e.name) CONTAINS $norm_condition_name
                       OR size([syn IN $synonyms WHERE toLower(e.name) CONTAINS toLower(syn)]) > 0
                   )
                WITH p, e // Keep plant and the matching effect
                RETURN p.name as plant_name,
                    p.scientific_name as scientific_name,
                    e.name as effect // Return the specific effect that matched
                ORDER BY plant_name
                LIMIT 15
             """,
             # --- Safety Information for a Plant ---
             'safety_info': """
                MATCH (p:Plant)
                WHERE toLower(p.name) = $norm_entity_name
                WITH p LIMIT 1
                OPTIONAL MATCH (p)-[:MAY_CAUSE]->(s:SideEffect)
                OPTIONAL MATCH (p)-[:PRODUCES_EFFECT]->(e:TherapeuticEffect)
                RETURN p.name as plant_name,
                    collect(DISTINCT s.name) as side_effects,
                    collect(DISTINCT e.name) as effects_context
             """,
             # --- Plants containing a Compound ---
             'compound_plants': """
                MATCH (c:Compound)<-[:CONTAINS]-(p:Plant)
                WHERE toLower(c.name) = $norm_entity_name
                WITH p
                OPTIONAL MATCH (p)-[:PRODUCES_EFFECT]->(e:TherapeuticEffect)
                OPTIONAL MATCH (p)-[:PREPARED_BY]->(m:PreparationMethod)
                OPTIONAL MATCH (c)-[:CONTRIBUTES_TO]->(ce:TherapeuticEffect)
                RETURN p.name as plant_name,
                    p.scientific_name as scientific_name,
                    collect(DISTINCT e.name) as plant_effects,
                    collect(DISTINCT ce.name) as compound_effects,
                    collect(DISTINCT m.name) as preparations
                ORDER BY plant_name
                LIMIT 20
            """,
            # --- Effects of a Compound ---
             'compound_effects': """
                MATCH (c:Compound)
                WHERE toLower(c.name) = $norm_entity_name
                WITH c LIMIT 1 
                OPTIONAL MATCH (c)-[:CONTRIBUTES_TO]->(e:TherapeuticEffect)
                OPTIONAL MATCH (c)<-[:CONTAINS]-(p:Plant)
                RETURN c.name as compound_name,
                    collect(DISTINCT e.name) as effects,
                    collect(DISTINCT p.name) as found_in_plants
             """,
             # --- Keyword Search (Fallback - Requires APOC for efficiency) ---
             # This query remains basic and potentially inefficient without indexing.
             'keyword_search': """
                // Example: Search plant names, effect names, compound names
                // Using OR and CONTAINS (inefficient without index)
                MATCH (p:Plant) WHERE toLower(p.name) CONTAINS $keyword OR toLower(p.scientific_name) CONTAINS $keyword
                WITH collect({type: 'Plant', name: p.name, scientific_name: p.scientific_name, description: p.morphology}) as plants
                MATCH (e:TherapeuticEffect) WHERE toLower(e.name) CONTAINS $keyword OR toLower(e.description) CONTAINS $keyword
                WITH plants + collect({type: 'Effect', name: e.name, description: e.description}) as plants_effects
                MATCH (c:Compound) WHERE toLower(c.name) CONTAINS $keyword
                WITH plants_effects + collect({type: 'Compound', name: c.name, description: null}) as results
                UNWIND results as result
                RETURN result.type as type, result.name as name, result.scientific_name as scientific_name, result.description as description
                LIMIT 15
             """
        }
        logger.debug("Cypher query templates initialized.")


    def _normalize_for_query(self, term: str) -> str:
        """Normalizes a term for querying: lowercase, strip, lemmatize (if enabled)."""
        if not term or not isinstance(term, str):
            return ''
        normalized = term.lower().strip()
        # Remove possessive 's
        normalized = re.sub(r"'s$", "", normalized)
        # Basic lemmatization if available
        if self.lemmatizer:
            try:
                # Try noun lemmatization first
                lemmatized = self.lemmatizer.lemmatize(normalized, pos='n')
                # If no change, try verb lemmatization (less common for entities)
                if lemmatized == normalized:
                    lemmatized = self.lemmatizer.lemmatize(normalized, pos='v')

                # Prefer lemmatized version if it's different and potentially valid
                if lemmatized != normalized and len(lemmatized) >= 3:
                     logger.debug(f"Normalized '{term}' to '{lemmatized}' using lemmatizer.")
                     return lemmatized
                else:
                     return normalized # Return original normalized
            except Exception as e:
                logger.error(f"Error during lemmatization for '{normalized}': {e}")
                return normalized # Fallback to non-lemmatized on error
        else:
            return normalized # No lemmatizer, just return lowercase stripped version


    def _fuzzy_match(self, n_grams: Union[List[str], Set[str]], entity_list: Set[str], threshold=80, scorer=None, limit=3, max_match_len=40) -> Set[str]:
        """ Performs fuzzy matching using thefuzz with improved filtering. """
        if not entity_list or not n_grams:
            return set()

        # Ensure entity list contains only non-empty strings
        str_entity_list = {str(item).strip() for item in entity_list if item and isinstance(item, str) and str(item).strip()}
        if not str_entity_list:
             logger.debug("Fuzzy match called with empty string entity list.")
             return set()

        # Limit the number of n-grams to check for performance
        # Convert n_grams to set first if it's a list
        unique_n_grams_to_match = set(n_grams) if isinstance(n_grams, set) else set(n_grams)
        # Filter out empty strings from n-grams
        unique_n_grams_to_match = {ng for ng in unique_n_grams_to_match if ng and len(ng) >= 2} # Min length 2 for n-grams
        # Sort for potential (minor) optimization if needed, limit size
        sorted_n_grams = sorted(list(unique_n_grams_to_match), key=len, reverse=True)[:150]

        potential_good_matches = set()
        primary_scorer = scorer or fuzz.token_sort_ratio # Good general purpose scorer

        for n_gram in sorted_n_grams:
            try:
                # Use process.extract with the chosen scorer
                potential_matches = process.extract(n_gram, str_entity_list, scorer=primary_scorer, limit=limit)

                for match, score in potential_matches:
                    # --- Stricter Filtering ---
                    len_diff = abs(len(n_gram) - len(match))
                    # Dynamic length difference based on n-gram length (more tolerance for longer n-grams)
                    max_allowed_len_diff = max(3, int(len(n_gram) * 0.5)) + 2

                    # Word count check on the *matched* entity
                    match_word_count = len(match.split())

                    if score >= threshold and \
                       len(match) <= max_match_len and \
                       match_word_count <= 6 and \
                       len_diff <= max_allowed_len_diff:

                        potential_good_matches.add(match)
                        logger.debug(f"Fuzzy Match ({primary_scorer.__name__}): '{n_gram}' -> '{match}' (Score: {score})")

            except Exception as e:
                logger.warning(f"Fuzzy match error for n-gram '{n_gram}' with scorer {primary_scorer.__name__}: {e}", exc_info=False)
                continue

        if not potential_good_matches:
            return set()

        # --- Refine Matches: Remove Subsumed ---
        sorted_matches = sorted(list(potential_good_matches), key=len, reverse=True)
        final_matches = set()
        for match in sorted_matches:
            is_subsumed = any(match != final_match and match in final_match for final_match in final_matches)
            if not is_subsumed:
                items_to_remove = {kept for kept in final_matches if kept != match and kept in match}
                final_matches -= items_to_remove
                final_matches.add(match)

        logger.debug(f"Refined fuzzy matches: {final_matches}")
        return final_matches

    # --- Entity Extraction (v5 - Refined Logic) ---
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extracts entities (plants, conditions, compounds, regions) using cleaning,
        n-grams, fuzzy matching, and refinement. Returns entities sorted alphabetically.
        """
        entities: Dict[str, Set[str]] = {'plants': set(), 'conditions': set(), 'compounds': set(), 'regions': set()}
        if not text or not text.strip():
            logger.warning("Input text is empty, skipping entity extraction.")
            return {k: [] for k in entities} # Return empty lists

        # 1. Clean text for processing & Keep original lower for context
        original_text_lower = text.lower()
        clean_text_for_processing = self._clean_question_text(original_text_lower)
        logger.debug(f"Cleaned text for entity extraction: '{clean_text_for_processing}'")

        if not clean_text_for_processing:
            logger.debug("Cleaned text is empty after processing, skipping entity extraction.")
            return {k: [] for k in entities}

        # 2. Generate N-grams
        tokens = clean_text_for_processing.split()
        n_grams = []
        max_n = 4 # Max words per n-gram
        min_len = 4# Min char length for an n-gram to be considered
        for n in range(1, max_n + 1):
            for i in range(len(tokens) - n + 1):
                n_gram = ' '.join(tokens[i:i+n])
                if len(n_gram) >= min_len and not n_gram.isdigit():
                    n_grams.append(n_gram)
        # Add single tokens from original text too, if potentially missed
        n_grams.extend([token for token in original_text_lower.split() if len(token) >= min_len and not token.isdigit() and token not in n_grams])
        unique_n_grams = set(n_grams)
        logger.debug(f"Generated {len(unique_n_grams)} unique n-grams for matching.")

        # 3. Perform Fuzzy Matching against known entity lists
        # Ensure we pass sets to fuzzy match
        matched_plants = self._fuzzy_match(unique_n_grams, self.all_known_plants, threshold=self.fuzzy_match_threshold)
        matched_conditions = self._fuzzy_match(unique_n_grams, self.all_known_conditions, threshold=self.fuzzy_match_threshold)
        matched_compounds = self._fuzzy_match(unique_n_grams, self.all_known_compounds, threshold=self.fuzzy_match_threshold)
        matched_regions = self._fuzzy_match(unique_n_grams, self.all_known_regions, threshold=self.region_fuzzy_match_threshold)

        entities['plants'].update(matched_plants)
        entities['conditions'].update(matched_conditions)
        entities['compounds'].update(matched_compounds)
        entities['regions'].update(matched_regions)

        # 4. Handle Specific Known Variations / Overrides (if fuzzy match misses)
        # Example: Ensure 'St. John's Wort' variations map correctly if needed
        sjw_variations = {"st johns wort", "st. johns wort", "st. john's wort"}
        sjw_target = "St. John's Wort" # Canonical form from your list
        if any(var in original_text_lower for var in sjw_variations) and sjw_target in self.all_known_plants:
             entities['plants'].add(sjw_target)

        # Handle curcumin/curcuminoids based on context? (Complex)
        # If 'curcuminoids' found but 'turmeric' mentioned, maybe add 'curcumin'?
        if 'curcuminoids' in entities['compounds'] and 'turmeric' in original_text_lower and 'curcumin' in self.all_known_compounds:
             entities['compounds'].add('curcumin')
             # Optionally remove curcuminoids if curcumin is preferred?
             # entities['compounds'].discard('curcuminoids')

        # 5. Final Refinement: Remove subsumed entities *across* all categories found
        all_found_entities = set().union(*entities.values())
        if len(all_found_entities) > 1: # Only refine if multiple entities were found
             sorted_all_found = sorted(list(all_found_entities), key=len, reverse=True)
             final_kept_entities = set()
             for entity in sorted_all_found:
                  # Check if this entity is a substring of any already kept longer match
                  is_subsumed_by_longer = any(entity != kept_entity and entity in kept_entity for kept_entity in final_kept_entities)
                  if not is_subsumed_by_longer:
                       # Remove any shorter matches already kept that are substrings of this new match
                       items_to_remove = {kept for kept in final_kept_entities if kept != entity and kept in entity}
                       final_kept_entities -= items_to_remove
                       final_kept_entities.add(entity) # Add the current (longer or non-subsumed) match

             # Update the entities dictionary with only the final kept ones
             for category in entities:
                  entities[category] = {e for e in entities[category] if e in final_kept_entities}
             logger.debug(f"Entities after cross-category refinement: {entities}")


        # 6. Convert sets to sorted lists for consistent output
        final_entities = {k: sorted(list(v)) for k, v in entities.items()}

        logger.info(f"Final extracted entities: {final_entities}")
        return final_entities

    # === Public API Methods ===

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extracts all categories of entities from text."""
        return self._extract_entities(text)

    def extract_plant_names(self, text: str) -> List[str]:
        """Extracts only plant names from text."""
        return self._extract_entities(text).get('plants', [])

    def extract_health_conditions(self, text: str) -> List[str]:
        """Extracts only health conditions from text."""
        return self._extract_entities(text).get('conditions', [])

    def extract_compounds(self, text: str) -> List[str]:
        """Extracts only compound names from text."""
        return self._extract_entities(text).get('compounds', [])

    def extract_regions(self, text: str) -> List[str]:
        """Extracts only region names from text."""
        return self._extract_entities(text).get('regions', [])

    def extract_keywords(self, question: str) -> List[str]:
        """Extracts keywords, including entities and other significant terms."""
        entities = self._extract_entities(question)
        # Flatten all found entities into a set of lowercase words
        entity_keywords = set()
        for category_list in entities.values():
             for entity in category_list:
                  # Split entity into words and add
                  entity_keywords.update(entity.lower().split())

        # Use cleaned text for general keyword extraction
        clean_text = self._clean_question_text(question)
        tokens = set(clean_text.split())

        # More comprehensive stopwords list (consider using NLTK's list for more robustness)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'of', 'for', 'to', 'in', 'on', 'with', 'by', 'at', 'about',
            'what', 'which', 'who', 'whom', 'where', 'when', 'why', 'how',
            'i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'we', 'us', 'our', 'ours', 'they', 'them', 'their', 'theirs',
            'am', 'be', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'and', 'or', 'but', 'if', 'because', 'as', 'until', 'while', 'so', 'than', 'too', 'very',
            'can', 'will', 'should', 'could', 'would', 'may', 'might', 'must',
            's', 't', 'd', 'll', 'm', 'o', 're', 've', 'y', # Contractions part
            'any', 'all', 'some', 'other', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'other',
            'please', 'tell', 'me', 'give', 'find', 'list', 'show', 'describe', 'information', 'info',
            'effects', 'benefits', 'uses', 'properties', 'side', 'safe', 'safety', 'preparation', 'prepare', 'make',
            'help', 'good', 'get', 'need', 'want', 'know', 'use', 'grow', # Common verbs/nouns in queries
            'plant', 'herb', 'condition', 'compound', 'region', # Overly generic types
            # Add question words if not already covered
            'is', 'are', 'what', 'which', 'who', 'whom', 'where', 'when', 'why', 'how', 'do', 'does', 'did',
        }

        # Combine entity words and non-stopword tokens
        keywords = entity_keywords.union(token for token in tokens if token not in stopwords and len(token) > 2 and not token.isdigit())
        # Remove any remaining stopwords that might have been part of entities
        keywords = keywords - stopwords

        return sorted(list(keywords))

    def train_projection_layer(self, *args, **kwargs):
        """Placeholder: Training should be done separately."""
        logger.warning("Projection layer training is handled in a separate script (e.g., knowledge_graph_embeddings.py).")
        raise NotImplementedError("Projection layer training is not implemented within BertProcessor.")

    def load_projection_layer(self, path=None):
        """ Loads pre-trained weights for the BERT-to-KG projection layer. """
        if path is None:
             path = os.path.join(self.kg_embeddings_dir, "bert_to_kg_proj.pt")

        if not self.bert_to_kg_proj:
             logger.warning("Projection layer (self.bert_to_kg_proj) not initialized. Cannot load weights.")
             return False
        if not os.path.exists(path):
             logger.warning(f"Projection layer weights file not found at {path}. Using randomly initialized weights.")
             return False
        try:
            # Use weights_only=True for security unless you trust the source implicitly
            self.bert_to_kg_proj.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            self.bert_to_kg_proj.to(self.device)
            self.bert_to_kg_proj.eval() # Set to evaluation mode
            logger.info(f"Projection layer weights loaded successfully from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading projection layer weights from {path}: {e}", exc_info=True)
            # Keep the randomly initialized layer instead of setting to None
            return False

    # --- Intent Classification (v5 - Refined Priorities & Logic) ---
    def classify_question_intent(self, question: str) -> str:
        """
        Determines the primary intent using prioritized rules, keywords, and extracted entities.
        Maps to specific query templates.
        """
        q_lower = question.lower()
        # Use the improved extraction method
        entities = self._extract_entities(question)
        # Convert back to sets for easier checking, excluding empty lists
        entities_sets = {k: set(v) for k, v in entities.items() if v}

        q_clean_tokens = set(self._clean_question_text(question).split()) # Use cleaned tokens for keyword checks

        logger.debug(f"Classifying intent for: '{question[:100]}...' | Entities: {entities_sets}")

        # --- Define Keywords for Intents ---
        intent_keywords = {
            "safety_info": {"safe", "safety", "side effect", "adverse", "risk", "danger", "precaution", "contraindication", "interaction", "pregnant", "nursing", "warning", "caution"},
            "plant_preparation": {"prepare", "preparation", "make", "brew", "infuse", "extract", "how to use", "recipe for", "decoction", "tincture", "infusion", "poultice"},
            "similar_plants": {"similar to", "like", "related to", "alternative to", "substitute for", "compare to", "vs", "versus"},
            "condition_plants": {"help with", "good for", "treat", "remedy for", "alleviate", "plants for", "herbs for", "benefit for", "aid for"},
            "compound_effects": {"effects of", "benefits of", "properties of", "what does .* do"}, # For compounds
            "plant_compounds": {"compounds in", "what compounds", "active ingredients in", "contains", "chemicals in"}, # For plants
            "compound_plants": {"plants with", "which plants have", "source of", "find .* in", "plant source"}, # For compounds
            "region_plants": {"grow in", "found in", "native to", "from", "region", "area", "location"},
            "general_query": {"what is a", "define", "meaning of", "explain"},
        }
        # Helper to check keywords
        def check_kws(intent_key):
             return any(kw in q_lower for kw in intent_keywords.get(intent_key, set()))

        # --- Intent Prioritization Logic ---

        # 1. Safety Intent (High Priority if Plant is Mentioned)
        if entities_sets.get('plants') and check_kws("safety_info"):
            logger.debug("Intent -> safety_info (Plant + Safety Keyword)")
            return "safety_info"

        # 2. Preparation Intent
        if check_kws("plant_preparation"):
            if entities_sets.get('plants'):
                logger.debug("Intent -> plant_preparation (Plant + Prep Keyword)")
                return "plant_preparation"
            if entities_sets.get('conditions'):
                logger.debug("Intent -> preparation_for_condition (Condition + Prep Keyword)")
                return "preparation_for_condition"
            # Fall through if only prep keyword

        # 3. Similarity Intent
        if check_kws("similar_plants"):
            if entities_sets.get('plants'):
                logger.debug("Intent -> similar_plants (Plant + Similarity Keyword)")
                return "similar_plants"
            # Add similar_compounds if needed

        # 4. Multi-Constraint Intents (Region + Condition)
        if entities_sets.get('regions') and entities_sets.get('conditions'):
             logger.debug("Intent -> region_condition_plants (Region + Condition Entities)")
             return "region_condition_plants" # Assume intent if both entities present

        # 5. Multi-Constraint Intent (Multiple Conditions)
        # Check length of the list from the original dict, not the set
        if len(entities.get('conditions', [])) >= 2:
            logger.debug("Intent -> multi_condition_plants (Multiple Conditions)")
            return "multi_condition_plants"

        # 6. Compound-Related Intents
        if entities_sets.get('compounds'):
             # Check if asking for plants containing the compound
             # Use regex for "find X in Y" pattern
             if check_kws("compound_plants") or re.search(r'\bfind\b.*\bin\b', q_lower):
                  logger.debug("Intent -> compound_plants (Compound + Source Keyword/Pattern)")
                  return "compound_plants"
             # Otherwise, assume asking about the compound's effects/properties
             logger.debug("Intent -> compound_effects (Compound Present)")
             return "compound_effects"

        # 7. Plant-Related Intents (if not safety/prep/similar)
        if entities_sets.get('plants'):
             # Check if asking about compounds *within* the plant
             if check_kws("plant_compounds"):
                  logger.debug("Intent -> plant_compounds (Plant + Compound Keyword)")
                  return "plant_compounds"
             # Check if asking about conditions the plant helps with (Condition has higher priority)
             if entities_sets.get('conditions'):
                  logger.debug("Intent -> condition_plants (Plant + Condition Entities)")
                  return "condition_plants" # Condition presence takes precedence
             # Default to general plant info
             logger.debug("Intent -> plant_info (Plant Present, no other strong intent)")
             return "plant_info"

        # 8. Condition Intent (if not handled above)
        if entities_sets.get('conditions'):
            logger.debug("Intent -> condition_plants (Condition Present)")
            return "condition_plants"

        # 9. Region Intent (if not handled above)
        if entities_sets.get('regions'):
            logger.debug("Intent -> region_plants (Region Present)")
            return "region_plants"

        # 10. General/Keyword Search Fallbacks
        if check_kws("general_query"):
             logger.debug("Intent -> general_query (Definitional Keyword)")
             return "general_query"

        # If any entities were found, but no specific intent matched
        if entities_sets:
             logger.debug("Intent -> keyword_search (Entities found, no specific intent match)")
             return "keyword_search" # Trigger fallback search/explanation

        # If cleaned text exists but no entities/intent
        if q_clean_tokens:
             logger.debug("Intent -> keyword_search_empty (No entities, but text exists)")
             return "keyword_search_empty" # Triggers general explanation

        # Final fallback for empty/unparsable input
        logger.debug("Intent -> error (No meaningful input or entities)")
        return "error"


    # --- Synonym Expansion (v5 - Expanded based on test failures) ---
    def get_synonyms(self, term: str, context: Optional[str] = None) -> List[str]:
        """
        Gets expanded synonyms for conditions, plants, and compounds.
        Includes plurals, basic verb forms, and common variations.
        """
        term_lower = term.lower().strip()
        if not term_lower: return []

        # Predefined synonyms (expand this list significantly based on domain knowledge and KG)
        synonym_map = {
            # Conditions
            'inflammation': ['swelling', 'inflammatory', 'inflamed', 'anti inflammatory', 'anti-inflammatory', 'phlogosis'],
            'pain': ['ache', 'soreness', 'discomfort', 'analgesic', 'algesia', 'painful', 'anti-pain', 'pain relief'],
            'arthritis': ['rheumatism', 'joint pain', 'rheumatic', 'osteoarthritis', 'rheumatoid arthritis', 'joint inflammation', "devil's claw"],
            'insomnia': ['sleeplessness', 'sleep disorder', 'difficulty sleeping', 'restlessness', 'sleep loss', 'valerian', 'chamomile', 'passionflower', 'hops', 'lemon balm'],
            'anxiety': ['nervousness', 'worry', 'unease', 'anxiousness', 'stress', 'panic', 'apprehension', 'calming', 'kava', 'passionflower'],
            'stress': ['tension', 'strain', 'pressure', 'anxiety', 'worry', 'overwhelm', 'adaptogen', 'ashwagandha', 'holy basil'],
            'cold': ['common cold', 'coryza', 'viral infection', 'rhinovirus', 'nasal congestion'],
            'nausea': ['vomiting', 'upset stomach', 'queasiness', 'anti-nausea', 'anti-emetic'],
            'digestion': ['digestive issues', 'indigestion', 'dyspepsia', 'stomach upset', 'gut health', 'bloating', 'gas'],
            'burns': ['scalds', 'skin burn', 'thermal injury', 'sunburn'],
            'skin': ['dermatological', 'cutaneous', 'skin condition', 'complexion', 'rash', 'dermatitis'],
            'depression': ['mood disorder', 'low mood', 'sadness', 'melancholy', 'dysthymia', 'st johns wort'],
            # Plants
            'ginger': ['zingiber officinale', 'zingiber', 'gingerol', 'jinger', 'zinger', 'ginger root'],
            'turmeric': ['curcuma longa', 'curcumin', 'tumeric', 'indian saffron', 'curcuma'],
            'aloe vera': ['aloe barbadensis', 'burn plant', 'aloe'],
            "st johns wort": ["hypericum perforatum", "st. john's wort", "st john's wort", "st. johns wort", 'hypericum'],
            'echinacea': ['coneflower', 'purple coneflower', 'echinaceae', 'echinacea purpurea', 'echinacea angustifolia'],
            'ginseng': ['panax ginseng', 'panax quinquefolius', 'asian ginseng', 'american ginseng', 'korean ginseng'],
            'chamomile': ['matricaria chamomilla', 'german chamomile', 'chamaemelum nobile', 'roman chamomile', 'camomile'],
            'valerian': ['valeriana officinalis', 'valerian root', 'garden heliotrope'],
            'passionflower': ['passiflora incarnata', 'maypop', 'passion flower'],
            "devil's claw": ['harpagophytum procumbens', 'grapple plant', 'wood spider'],
            "cat's claw": ['uncaria tomentosa', 'uña de gato'],
            'kava': ['piper methysticum', 'kava kava'],
            'ashwagandha': ['withania somnifera', 'indian ginseng', 'winter cherry'],
            # Compounds
            'curcumin': ['curcuminoid', 'turmeric', 'diferuloylmethane'], # Link to plant
            'curcuminoids': ['curcumin', 'turmeric'], # Link back
            'gingerol': ['ginger', '[6]-gingerol'],
            'quercetin': ['flavonoid', 'antioxidant', 'sophoretin'],
            'hypericin': ['st johns wort', 'hypericum'],
            'allicin': ['garlic'],
            # Regions
            'andes': ['andean region', 'andes mountains', 'south america'], # Add continent
            'asia': ['asian', 'oriental'],
            # Safety/Misc
            'safe': ['safety', 'caution', 'risk', 'side effect', 'adverse effect', 'contraindication', 'interaction', 'warning'],
            'pregnancy': ['pregnant', 'gestation', 'expecting', 'maternity'],
            'preparation': ['prepare', 'make', 'method', 'recipe', 'extraction', 'infusion', 'decoction', 'tincture', 'tea'],
            'similar': ['like', 'related', 'alternative', 'substitute', 'comparison', 'equivalent', 'comparable'],
        }

        # Basic morphological variations
        base_terms = {term_lower}
        if self.lemmatizer:
             try:
                  base_terms.add(self.lemmatizer.lemmatize(term_lower, pos='n'))
                  base_terms.add(self.lemmatizer.lemmatize(term_lower, pos='v'))
             except Exception: pass # Ignore lemmatization errors
        if term_lower.endswith('s'): base_terms.add(term_lower[:-1])
        else: base_terms.add(term_lower + 's')
        if term_lower.endswith('ing'): base_terms.add(term_lower[:-3])
        elif term_lower.endswith('ed'): base_terms.add(term_lower[:-2])
        elif term_lower.endswith('er'): base_terms.add(term_lower[:-2])
        elif term_lower.endswith('tion'): base_terms.add(term_lower[:-4]) # Crude stemming

        # Combine predefined synonyms and basic variations
        predefined_syns = set(synonym_map.get(term_lower, []))
        all_syns = base_terms.union(predefined_syns)

        # Add synonyms of synonyms (one level deep) - Use with caution
        # second_level_syns = set()
        # for syn in predefined_syns:
        #     second_level_syns.update(synonym_map.get(syn, []))
        # all_syns.update(second_level_syns)

        # Filter out empty strings, short strings, and the original term if desired (keeping original term is usually safer)
        final_syns = sorted([s for s in all_syns if s and len(s) > 1])
        logger.debug(f"Synonyms for '{term}': {final_syns}")
        return final_syns

    def extract_entities_and_intent(self, question: str) -> Dict[str, Any]:
        """
        Utility: Returns both extracted entities and the classified intent.
        """
        # Ensure entities are returned as lists, not sets
        entities_dict = self._extract_entities(question)
        intent = self.classify_question_intent(question)
        return {"entities": entities_dict, "intent": intent}

    def build_neo4j_query(self, question: str) -> Dict[str, Any]:
        """ Builds the Neo4j query dictionary based on intent and entities. (v5 - Refined) """
        extraction_result = self.extract_entities_and_intent(question)
        intent = extraction_result['intent']
        entities = extraction_result['entities'] # This is now Dict[str, List[str]]

        parameters = {}
        query_info = {'intent': intent, 'query': None, 'parameters': parameters}
        logger.debug(f"Building query for intent '{intent}' with entities: {entities}")

        # Helper to get first entity and normalize it
        def get_norm_entity(key):
             entity_list = entities.get(key) # entities is Dict[str, List]
             if entity_list and isinstance(entity_list, list) and entity_list[0]:
                  return self._normalize_for_query(entity_list[0])
             return None

        # Helper to get synonyms for the first entity
        def get_entity_synonyms(key):
             entity_list = entities.get(key)
             if entity_list and isinstance(entity_list, list) and entity_list[0]:
                  # Pass original term to get_synonyms
                  return self.get_synonyms(entity_list[0])
             return []

        try:
            # --- Handle Intents ---
            if intent == 'plant_info':
                norm_name = get_norm_entity('plants')
                if norm_name:
                    parameters['norm_entity_name'] = norm_name
                    query_info['query'] = self.query_templates['plant_info']
                else: logger.warning(f"Intent '{intent}' but no plant entity found.")

            elif intent == 'condition_plants':
                norm_name = get_norm_entity('conditions')
                if norm_name:
                    parameters['norm_entity_name'] = norm_name
                    parameters['synonyms'] = get_entity_synonyms('conditions')
                    query_info['query'] = self.query_templates['condition_plants']
                else: logger.warning(f"Intent '{intent}' but no condition entity found.")

            elif intent == 'multi_condition_plants':
                conditions = entities.get('conditions', [])
                if len(conditions) >= 2:
                    parameters['norm_condition1'] = self._normalize_for_query(conditions[0])
                    parameters['norm_condition2'] = self._normalize_for_query(conditions[1])
                    parameters['synonyms1'] = self.get_synonyms(conditions[0])
                    parameters['synonyms2'] = self.get_synonyms(conditions[1])
                    query_info['query'] = self.query_templates['multi_condition_plants']
                else: logger.warning(f"Intent '{intent}' but less than 2 condition entities found.")

            elif intent == 'similar_plants':
                norm_name = get_norm_entity('plants')
                if norm_name:
                    parameters['norm_entity_name'] = norm_name
                    query_info['query'] = self.query_templates['similar_plants']
                else: logger.warning(f"Intent '{intent}' but no plant entity found.")

            elif intent == 'compound_effects':
                norm_name = get_norm_entity('compounds')
                if norm_name:
                    parameters['norm_entity_name'] = norm_name
                    parameters['synonyms'] = get_entity_synonyms('compounds')
                    query_info['query'] = self.query_templates['compound_effects']
                else: logger.warning(f"Intent '{intent}' but no compound entity found.")

            elif intent == 'plant_compounds':
                norm_name = get_norm_entity('plants')
                if norm_name:
                    parameters['norm_entity_name'] = norm_name
                    query_info['query'] = self.query_templates['plant_compounds']
                else: logger.warning(f"Intent '{intent}' but no plant entity found.")

            elif intent == 'compound_plants':
                norm_name = get_norm_entity('compounds')
                if norm_name:
                    parameters['norm_entity_name'] = norm_name
                    parameters['synonyms'] = get_entity_synonyms('compounds')
                    query_info['query'] = self.query_templates['compound_plants']
                else: logger.warning(f"Intent '{intent}' but no compound entity found.")

            elif intent == 'preparation_for_condition':
                norm_name = get_norm_entity('conditions')
                if norm_name:
                    parameters['norm_entity_name'] = norm_name
                    parameters['synonyms'] = get_entity_synonyms('conditions')
                    query_info['query'] = self.query_templates['preparation_for_condition']
                else: logger.warning(f"Intent '{intent}' but no condition entity found.")

            elif intent == 'plant_preparation':
                norm_name = get_norm_entity('plants')
                if norm_name:
                    parameters['norm_entity_name'] = norm_name
                    query_info['query'] = self.query_templates['plant_preparation']
                else: logger.warning(f"Intent '{intent}' but no plant entity found.")

            elif intent == 'region_plants':
                norm_name = get_norm_entity('regions')
                if norm_name:
                    parameters['norm_entity_name'] = norm_name # Use CONTAINS in query
                    query_info['query'] = self.query_templates['region_plants']
                else: logger.warning(f"Intent '{intent}' but no region entity found.")

            elif intent == 'region_condition_plants':
                norm_region = get_norm_entity('regions')
                norm_condition = get_norm_entity('conditions')
                if norm_region and norm_condition:
                    parameters['norm_region_name'] = norm_region # Use CONTAINS in query
                    parameters['norm_condition_name'] = norm_condition
                    parameters['synonyms'] = get_entity_synonyms('conditions')
                    query_info['query'] = self.query_templates['region_condition_plants']
                else: logger.warning(f"Intent '{intent}' but missing region or condition entity.")

            elif intent == 'safety_info':
                norm_name = get_norm_entity('plants')
                if norm_name:
                    parameters['norm_entity_name'] = norm_name
                    query_info['query'] = self.query_templates['safety_info']
                else: logger.warning(f"Intent '{intent}' but no plant entity found.")

            elif intent == 'keyword_search':
                 keywords = self.extract_keywords(question)
                 if keywords:
                      # Simple keyword search: Use the first keyword for CONTAINS check
                      # More complex logic would involve multiple keywords, AND/OR, indexing
                      parameters['keyword'] = self._normalize_for_query(keywords[0])
                      if 'keyword_search' in self.query_templates:
                           query_info['query'] = self.query_templates['keyword_search']
                      else:
                           logger.warning("Keyword search query template missing.")
                           query_info['query'] = None
                           query_info['intent'] = 'keyword_search_empty' # Fallback intent
                 else:
                      logger.warning(f"Intent '{intent}' but no keywords extracted.")
                      query_info['intent'] = 'keyword_search_empty' # Change intent if no keywords

            # Handle intents that don't need a query
            elif intent in ['general_query', 'keyword_search_empty', 'error', 'unknown']:
                 logger.info(f"Intent '{intent}' does not require a database query.")
                 query_info['query'] = None

            else: # Fallback for unhandled intents
                logger.warning(f"No query template defined or matched for intent '{intent}'. No query will be executed.")
                query_info['query'] = None
                # Don't change intent here, let QA system handle 'unknown' or other specific fallbacks
                if intent not in ['general_query', 'keyword_search_empty', 'error']:
                     query_info['intent'] = 'unknown'


            # Final check if query is set
            if query_info['query']:
                 logger.info(f"Built query for intent '{intent}'.")
            else:
                 logger.info(f"No query generated for intent '{intent}'.")

        except Exception as e:
             logger.error(f"Error building Neo4j query for intent '{intent}': {e}", exc_info=True)
             query_info['query'] = None
             query_info['intent'] = 'error' # Set intent to error on failure

        return query_info

    def get_bert_embedding(self, text: str, pool_strategy: str = 'mean') -> Optional[np.ndarray]:
         """Generates a BERT embedding for the input text."""
         if not self.model or not self.tokenizer:
              logger.error("BERT model/tokenizer not loaded. Cannot generate embedding.")
              return None
         if not text or not isinstance(text, str):
              logger.error("Invalid input text for embedding generation.")
              return None

         try:
              inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
              inputs = {k: v.to(self.device) for k, v in inputs.items()}
              with torch.no_grad():
                   outputs = self.model(**inputs)

              # Pooling strategy
              last_hidden_state = outputs.last_hidden_state
              attention_mask = inputs['attention_mask']

              if pool_strategy == 'mean':
                   # Mean pooling considering attention mask
                   mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                   sum_hidden = torch.sum(last_hidden_state * mask_expanded, 1)
                   sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9) # Avoid division by zero
                   embedding = sum_hidden / sum_mask
              elif pool_strategy == 'cls':
                   # Use the embedding of the [CLS] token
                   embedding = last_hidden_state[:, 0, :]
              else:
                   logger.warning(f"Unsupported pool strategy: {pool_strategy}. Using 'mean'.")
                   mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                   sum_hidden = torch.sum(last_hidden_state * mask_expanded, 1)
                   sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                   embedding = sum_hidden / sum_mask

              # Ensure embedding is 1D
              embedding_np = embedding.cpu().numpy()
              if embedding_np.ndim > 1:
                   embedding_np = embedding_np.squeeze() # Remove batch dim if present

              return embedding_np

         except Exception as e:
              logger.error(f"Error generating BERT embedding for text '{text[:50]}...': {e}", exc_info=True)
              return None

    def get_answer_confidence(self, question: str, answer: str) -> float:
         """
         Estimates confidence based on semantic similarity between question and answer embeddings.
         Requires BERT model to be loaded. Returns a score between 0 and 1.
         """
         if not self.model or not self.tokenizer:
              logger.warning("BERT model not available for confidence calculation.")
              return 0.5 # Default confidence

         q_emb = self.get_bert_embedding(question)
         a_emb = self.get_bert_embedding(answer)

         if q_emb is None or a_emb is None:
              logger.warning("Could not generate embeddings for confidence calculation.")
              return 0.3 # Lower confidence if embeddings fail

         try:
              # Ensure embeddings are 1D numpy arrays
              if q_emb.ndim > 1: q_emb = q_emb.squeeze()
              if a_emb.ndim > 1: a_emb = a_emb.squeeze()
              if q_emb.ndim != 1 or a_emb.ndim != 1:
                   logger.error(f"Embeddings have incorrect dimensions for similarity: Q={q_emb.shape}, A={a_emb.shape}")
                   return 0.2

              # Calculate cosine similarity
              q_norm = np.linalg.norm(q_emb)
              a_norm = np.linalg.norm(a_emb)
              if q_norm < 1e-9 or a_norm < 1e-9: return 0.0 # Avoid division by zero or near-zero

              similarity = np.dot(q_emb, a_emb) / (q_norm * a_norm)
              # Scale similarity (cosine is [-1, 1]) -> confidence score [0, 1]
              confidence = (similarity + 1) / 2
              return float(np.clip(confidence, 0.0, 1.0)) # Ensure valid range
         except Exception as e:
              logger.error(f"Error calculating embedding similarity for confidence: {e}", exc_info=True)
              return 0.4 # Default confidence on calculation error


# --- Main execution block for testing ---
if __name__ == "__main__":
    import argparse
    import sys # Import sys for exit

    parser = argparse.ArgumentParser(description="Test BertProcessor Functionality")
    parser.add_argument('--question', type=str, help='Question to process', default="What are the side effects of St John's Wort for anxiety?")
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--kg_dir', type=str, default='models/kg_embeddings_trained', help='Directory containing KG mappings and embeddings')
    # Add other arguments as needed, e.g., --load_proj

    args = parser.parse_args()

    print(f"Initializing BertProcessor with KG directory: {args.kg_dir}")
    try:
        processor = BertProcessor(kg_embeddings_dir=args.kg_dir)
    except Exception as init_e:
        print(f"\nFATAL: BertProcessor initialization failed: {init_e}")
        logger.critical("BertProcessor initialization failed.", exc_info=True)
        sys.exit(1)


    if not processor.model or not processor.tokenizer:
         print("\nBERT model or tokenizer failed to load during initialization. Exiting.")
         sys.exit(1)

    if args.interactive:
        print("\n[Interactive BertProcessor Test]")
        print("Enter a question to see extracted entities, intent, and generated query.")
        while True:
            try:
                q = input("\nEnter question (or 'quit'): ").strip()
                if q.lower() in {'exit', 'quit'}:
                    break
                if not q: continue

                print("-" * 20)
                # 1. Extract Entities & Intent
                result = processor.extract_entities_and_intent(q)
                print(f"Question: {q}")
                print(f"Entities: {result['entities']}")
                print(f"Intent:   {result['intent']}")

                # 2. Build Query
                query_info = processor.build_neo4j_query(q)
                print("\nGenerated Query Info:")
                print(f"  Query Intent: {query_info.get('intent')}")
                print(f"  Parameters:   {query_info.get('parameters')}")
                # Pretty print query if it exists
                query_text = query_info.get('query', 'None')
                print(f"  Query:        \n{query_text}")
                print("-" * 20)

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                 print(f"\nAn error occurred: {e}")
                 logger.error("Error in interactive loop", exc_info=True)

    else:
        # Single question mode
        q = args.question
        print("-" * 20)
        print(f"Processing Question: {q}")

        # 1. Extract Entities & Intent
        result = processor.extract_entities_and_intent(q)
        print(f"\nEntities: {result['entities']}")
        print(f"Intent:   {result['intent']}")

        # 2. Build Query
        query_info = processor.build_neo4j_query(q)
        print("\nGenerated Query Info:")
        print(f"  Query Intent: {query_info.get('intent')}")
        print(f"  Parameters:   {query_info.get('parameters')}")
        query_text = query_info.get('query', 'None')
        print(f"  Query:\n{query_text}") # Print query on multiple lines
        print("-" * 20)

        # Optional: Test confidence calculation
        demo_answer = "St. John's Wort (Hypericum perforatum) may cause side effects like photosensitivity and can interact significantly with many medications, including antidepressants. Always consult a doctor before use, especially if taking other drugs or if you have mood disorders."
        confidence = processor.get_answer_confidence(q, demo_answer)
        print(f"\nDemo Confidence (Q vs. A): {confidence:.3f}")
        print(f"Demo Answer: {demo_answer}")
        print("-" * 20)
