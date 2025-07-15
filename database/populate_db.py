import os
import re
from neo4j import GraphDatabase
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Neo4jConnection:
    """Handles connection and queries to Neo4j database."""
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        logging.info("Neo4j connection established.")

    def close(self):
        self._driver.close()
        logging.info("Neo4j connection closed.")

    def query(self, query, parameters=None, db=None):
        """Execute a Cypher query."""
        assert self._driver is not None, "Driver not initialized!"
        session = None
        response = None
        try:
            session = self._driver.session(database=db) if db else self._driver.session()
            response = list(session.run(query, parameters))
        except Exception as e:
            logging.error(f"Query failed: {e}\nQuery: {query}\nParameters: {parameters}")
        finally:
            if session is not None:
                session.close()
        return response

# --- Data Parsing Functions ---

def parse_plant_entry(entry_text):
    """Parses a single plant entry string into a dictionary."""
    plant_data = {
        'id': None, 'name': None, 'scientific_name': None, 'family': None,
        'common_names': [], 'morphology': None, 'distribution_text': None,
        'compounds_text': None, 'effects_text': None, 'methods_text': None,
        'side_effects_text': None
    }
    lines = entry_text.strip().split('\n')
    current_section = 'base' # base, characteristics, medicinal, usage

    for line in lines:
        line = line.strip()
        if not line or line == '-':
            continue

        # Detect section headers
        if line.startswith('===') and line.endswith('==='):
            section_name = line.strip('=').lower()
            if section_name in ['characteristics', 'medicinal', 'usage']:
                current_section = section_name
            elif section_name == 'end':
                break # End of plant entry
            continue

        # Parse based on current section
        try:
            key, value = line.split(':', 1)
            key = key.strip().lower().replace('_', '') # Normalize key
            value = value.strip()

            if current_section == 'base':
                if key == 'id': plant_data['id'] = value
                elif key == 'name': plant_data['name'] = value
                elif key == 'scientificname':
                    # Remove markdown italics and leading/trailing spaces
                    plant_data['scientific_name'] = re.sub(r'\*([^*]+)\*', r'\1', value).strip()
                elif key == 'family': plant_data['family'] = value
                elif key == 'commonnames':
                     plant_data['common_names'] = [cn.strip() for cn in value.split(',') if cn.strip()]

            elif current_section == 'characteristics':
                if key == 'morphology': plant_data['morphology'] = value
                elif key == 'distribution': plant_data['distribution_text'] = value

            elif current_section == 'medicinal':
                if key == 'compounds': plant_data['compounds_text'] = value
                elif key == 'effects': plant_data['effects_text'] = value

            elif current_section == 'usage':
                if key == 'methods': plant_data['methods_text'] = value
                elif key == 'sideeffects': plant_data['side_effects_text'] = value

        except ValueError:
             # Handle lines without a colon, maybe part of a previous multi-line value (though unlikely with current format)
             logging.warning(f"Skipping malformed line in section '{current_section}': {line}")


    # Basic validation
    if not plant_data['name'] or not plant_data['scientific_name']:
        logging.warning(f"Skipping entry due to missing name or scientific name. Content snippet: {entry_text[:100]}")
        return None

    return plant_data

def parse_structured_plants_file(file_path):
    """Reads the entire file and parses all plant entries."""
    parsed_plants = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split into plant entries using ===PLANT=== as delimiter
        # Use positive lookbehind to keep the delimiter for context if needed, though we remove it
        plant_entries = re.split(r'(?===PLANT===)', content)

        for entry in plant_entries:
            entry = entry.strip()
            # Remove the ===PLANT=== marker itself and check if entry is substantial
            if entry.startswith('===PLANT==='):
                entry = entry[len('===PLANT==='):].strip()

            if not entry or entry == '===': # Skip empty entries or remnants
                continue

            plant_data = parse_plant_entry(entry)
            if plant_data:
                parsed_plants.append(plant_data)

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except Exception as e:
        logging.error(f"Error parsing file {file_path}: {e}", exc_info=True)

    logging.info(f"Parsed {len(parsed_plants)} plants from {file_path}")
    return parsed_plants


# --- Entity Extraction Functions ---

def clean_name(name):
    """Standardize entity names."""
    if not name: return None
    # Lowercase, remove extra whitespace, handle simple plurals (add more sophisticated logic if needed)
    name = name.strip().lower()
    # Very basic plural removal for common patterns
    if name.endswith('s') and not name.endswith('ss'):
         # check if the singular form makes sense (e.g., avoid turning 'ginseng' into 'ginseng')
         # This is simplistic; a better lemmatizer would be ideal but adds complexity.
         # Let's keep it simple for now.
         pass # Keep plural forms for now to avoid errors like ginseng -> ginsen
    name = re.sub(r'\s+', ' ', name) # Consolidate whitespace
    # Remove trailing punctuation like commas or periods if they are artifacts
    name = name.rstrip('.,;')
    return name if len(name) > 1 else None # Return None if too short

def extract_entities(text, delimiters=r',|;|\band\b|\bor\b'):
    """Generic function to split text into potential entities."""
    if not text: return []
    # Split by delimiters, remove parenthetical content first
    text_no_paren = re.sub(r'\([^)]*\)', '', text).strip()
    # Split using regex for more flexibility
    entities = re.split(delimiters, text_no_paren)
    cleaned_entities = set() # Use set to avoid duplicates initially
    for entity in entities:
        cleaned = clean_name(entity)
        if cleaned:
            cleaned_entities.add(cleaned)
    return sorted(list(cleaned_entities)) # Return sorted list

def extract_regions(text):
    """Extracts geographic regions, handling common patterns."""
    if not text: return []
    # Prioritize ';' then ',' for splitting major regions
    regions = set()
    if ';' in text:
        parts = text.split(';')
    else:
        parts = text.split(',')

    for part in parts:
        # Remove details in parentheses like "(e.g., ...)
        part_cleaned = re.sub(r'\([^)]*\)', '', part).strip()
        part_cleaned = re.sub(r'\bnative to\b', '', part_cleaned, flags=re.IGNORECASE).strip()
        part_cleaned = re.sub(r'\bwidely cultivated in\b', '', part_cleaned, flags=re.IGNORECASE).strip()
        part_cleaned = re.sub(r'\bnaturalized in\b', '', part_cleaned, flags=re.IGNORECASE).strip()
        part_cleaned = part_cleaned.rstrip('.,')

        # Further split by ' and ' if necessary
        sub_regions = re.split(r'\s+and\s+', part_cleaned)
        for sub_region in sub_regions:
            cleaned = clean_name(sub_region)
            # Avoid adding overly generic terms if more specific ones exist
            if cleaned and len(cleaned) > 2 and cleaned not in ['worldwide', 'globally', 'elsewhere']:
                 regions.add(cleaned)

    # Add broad terms if they appear meaningful
    if 'worldwide' in text.lower(): regions.add('worldwide')
    if 'globally' in text.lower(): regions.add('globally') # Treat as synonym for worldwide?
    if 'tropics' in text.lower(): regions.add('tropics')
    if 'asia' in text.lower(): regions.add('asia')
    if 'europe' in text.lower(): regions.add('europe')
    if 'africa' in text.lower(): regions.add('africa')
    if 'americas' in text.lower(): regions.add('americas')
    if 'north america' in text.lower(): regions.add('north america')
    if 'south america' in text.lower(): regions.add('south america')

    # Basic filtering of vague/non-regional terms
    regions = {r for r in regions if not any(stop in r for stop in ['likely', 'parts of', 'regions of'])}

    return sorted(list(regions))


# --- Database Interaction Functions ---

def setup_schema(db_conn):
    """Sets up database constraints for unique names."""
    constraints = [
        "CREATE CONSTRAINT plant_name_unique IF NOT EXISTS FOR (p:Plant) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT plant_sci_name_unique IF NOT EXISTS FOR (p:Plant) REQUIRE p.scientific_name IS UNIQUE",
        "CREATE CONSTRAINT family_name_unique IF NOT EXISTS FOR (f:Family) REQUIRE f.name IS UNIQUE",
        "CREATE CONSTRAINT common_name_unique IF NOT EXISTS FOR (cn:CommonName) REQUIRE cn.name IS UNIQUE",
        "CREATE CONSTRAINT compound_name_unique IF NOT EXISTS FOR (c:Compound) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT effect_name_unique IF NOT EXISTS FOR (e:TherapeuticEffect) REQUIRE e.name IS UNIQUE",
        "CREATE CONSTRAINT method_name_unique IF NOT EXISTS FOR (m:PreparationMethod) REQUIRE m.name IS UNIQUE",
        "CREATE CONSTRAINT side_effect_name_unique IF NOT EXISTS FOR (s:SideEffect) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT region_name_unique IF NOT EXISTS FOR (r:Region) REQUIRE r.name IS UNIQUE"
    ]
    logging.info("Setting up schema constraints...")
    for constraint in constraints:
        try:
            db_conn.query(constraint)
        except Exception as e:
            # It's okay if constraints already exist, but log other errors
            if "already exists" not in str(e):
                 logging.error(f"Failed to create constraint: {constraint}. Error: {e}")
    logging.info("Schema constraints checked/created.")

def clear_database(db_conn):
    """Clears all nodes and relationships from the database."""
    logging.info("Clearing database...")
    try:
        db_conn.query("MATCH (n) DETACH DELETE n")
        logging.info("Database cleared successfully.")
    except Exception as e:
        logging.error(f"Failed to clear database: {e}")

def import_plant_data(db_conn, plant):
    """Imports a single plant's data into Neo4j."""
    logging.debug(f"Importing plant: {plant['name']}")

    # Create or merge the Plant node
    # Use MERGE on scientific_name as it's more stable than common name
    plant_query = """
    MERGE (p:Plant {scientific_name: $scientific_name})
    ON CREATE SET p.name = $name, p.id = $id, p.morphology = $morphology, p.distribution_text = $distribution_text
    ON MATCH SET p.name = $name, p.id = $id, p.morphology = $morphology, p.distribution_text = $distribution_text
    RETURN p
    """
    try:
        db_conn.query(plant_query, {
            "scientific_name": plant['scientific_name'],
            "name": plant['name'],
            "id": plant['id'],
            "morphology": plant['morphology'],
            "distribution_text": plant['distribution_text']
        })
    except Exception as e:
        logging.error(f"Failed to create/merge plant node for {plant['name']}: {e}")
        return # Stop processing this plant if node creation fails

    plant_identifier = plant['scientific_name'] # Use scientific name for matching relationships

    # --- Create related nodes and relationships ---

    # Family
    if plant['family']:
        fam_name = clean_name(plant['family'])
        if fam_name:
            db_conn.query("""
            MATCH (p:Plant {scientific_name: $sci_name})
            MERGE (f:Family {name: $fam_name})
            MERGE (p)-[:BELONGS_TO_FAMILY]->(f)
            """, {"sci_name": plant_identifier, "fam_name": fam_name})

    # Common Names
    for cn in plant['common_names']:
        com_name = clean_name(cn)
        if com_name:
            db_conn.query("""
            MATCH (p:Plant {scientific_name: $sci_name})
            MERGE (cn:CommonName {name: $com_name})
            MERGE (p)-[:ALSO_KNOWN_AS]->(cn)
            """, {"sci_name": plant_identifier, "com_name": com_name})

    # Regions
    regions = extract_regions(plant['distribution_text'])
    for region in regions:
        reg_name = clean_name(region)
        if reg_name:
            db_conn.query("""
            MATCH (p:Plant {scientific_name: $sci_name})
            MERGE (r:Region {name: $reg_name})
            MERGE (p)-[:GROWS_IN]->(r)
            """, {"sci_name": plant_identifier, "reg_name": reg_name})

    # Compounds
    compounds = extract_entities(plant['compounds_text'])
    for comp in compounds:
        comp_name = clean_name(comp)
        if comp_name:
            db_conn.query("""
            MATCH (p:Plant {scientific_name: $sci_name})
            MERGE (c:Compound {name: $comp_name})
            MERGE (p)-[:CONTAINS]->(c)
            """, {"sci_name": plant_identifier, "comp_name": comp_name})

    # Therapeutic Effects
    effects = extract_entities(plant['effects_text'])
    for effect in effects:
        eff_name = clean_name(effect)
         # Filter out vague terms sometimes caught by simple splitting
        if eff_name and eff_name not in ['etc', 'e.g.']:
            db_conn.query("""
            MATCH (p:Plant {scientific_name: $sci_name})
            MERGE (e:TherapeuticEffect {name: $eff_name})
            MERGE (p)-[:PRODUCES_EFFECT]->(e)
            """, {"sci_name": plant_identifier, "eff_name": eff_name})

    # Preparation Methods
    methods = extract_entities(plant['methods_text'])
    for method in methods:
        meth_name = clean_name(method)
        if meth_name:
            db_conn.query("""
            MATCH (p:Plant {scientific_name: $sci_name})
            MERGE (m:PreparationMethod {name: $meth_name})
            MERGE (p)-[:PREPARED_BY]->(m)
            """, {"sci_name": plant_identifier, "meth_name": meth_name})

    # Side Effects
    side_effects = extract_entities(plant['side_effects_text'], delimiters=r'[.;]+') # Split by sentences for side effects
    for side_effect in side_effects:
        # Side effects are often phrases, don't lowercase everything
        se_name = side_effect.strip().rstrip('.,')
        if se_name and len(se_name) > 5: # Filter very short fragments
             # Basic check to avoid importing instructions like "use cautiously" as a side effect node
             if not any(instr in se_name.lower() for instr in ['use cautiously', 'consult doctor', 'avoid during pregnancy', 'generally safe']):
                 db_conn.query("""
                 MATCH (p:Plant {scientific_name: $sci_name})
                 MERGE (s:SideEffect {name: $se_name})
                 MERGE (p)-[:MAY_CAUSE]->(s)
                 """, {"sci_name": plant_identifier, "se_name": se_name})

def create_cross_relationships(db_conn):
    """Creates relationships between non-plant nodes based on shared plants."""
    logging.info("Creating cross-relationships...")

    queries = {
        "Compound_Effects": """
            MATCH (c:Compound)<-[:CONTAINS]-(p:Plant)-[:PRODUCES_EFFECT]->(e:TherapeuticEffect)
            WITH c, e, COUNT(p) AS strength
            WHERE strength > 0 // Ensure connection exists
            MERGE (c)-[r:CONTRIBUTES_TO]->(e)
            ON CREATE SET r.evidence_count = strength
            ON MATCH SET r.evidence_count = strength
        """,
        "Family_Effects": """
            MATCH (f:Family)<-[:BELONGS_TO_FAMILY]-(p:Plant)-[:PRODUCES_EFFECT]->(e:TherapeuticEffect)
            WITH f, e, COUNT(p) AS strength
            WHERE strength > 1 // Only link if multiple family members share effect
            MERGE (f)-[r:ASSOCIATED_WITH_EFFECT]->(e)
            ON CREATE SET r.plant_count = strength
            ON MATCH SET r.plant_count = strength
        """,
        "Region_Compounds": """
            MATCH (r:Region)<-[:GROWS_IN]-(p:Plant)-[:CONTAINS]->(c:Compound)
            WITH r, c, COUNT(p) AS strength
            WHERE strength > 1 // Only link if multiple plants share compound in region
            MERGE (r)-[rel:MAY_SOURCE_COMPOUND]->(c)
            ON CREATE SET rel.plant_count = strength
            ON MATCH SET rel.plant_count = strength
        """,
        # Add more cross-relationship queries as needed
    }

    for name, query in queries.items():
        try:
            logging.info(f"Creating cross-relationship: {name}")
            db_conn.query(query)
        except Exception as e:
            logging.error(f"Failed to create cross-relationship '{name}': {e}")

    logging.info("Cross-relationships creation finished.")


def cleanup_database(db_conn):
    """Performs basic cleanup tasks like removing empty nodes."""
    logging.info("Performing database cleanup...")
    try:
        # Remove nodes with empty or null names (should be prevented by constraints, but good practice)
        db_conn.query("""
        MATCH (n) WHERE n.name IS NULL OR trim(n.name) = ''
        DETACH DELETE n
        """)
        logging.info("Removed nodes with empty/null names.")
    except Exception as e:
        logging.error(f"Error during database cleanup: {e}")


def verify_database_structure(db_conn):
    """Prints counts of nodes and relationships to verify import."""
    logging.info("Verifying database structure...")
    stats = {}
    try:
        # Node counts
        node_labels_query = "CALL db.labels() YIELD label RETURN label"
        labels_result = db_conn.query(node_labels_query)
        stats['Node Counts'] = {}
        for record in labels_result:
            label = record['label']
            count_query = f"MATCH (n:`{label}`) RETURN count(n) AS count"
            count_result = db_conn.query(count_query)
            stats['Node Counts'][label] = count_result[0]['count'] if count_result else 0

        # Relationship counts
        rel_types_query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        rel_types_result = db_conn.query(rel_types_query)
        stats['Relationship Counts'] = {}
        for record in rel_types_result:
            rel_type = record['relationshipType']
            count_query = f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) AS count"
            count_result = db_conn.query(count_query)
            stats['Relationship Counts'][rel_type] = count_result[0]['count'] if count_result else 0

        # Print stats
        print("\n--- Database Verification ---")
        for category, counts in stats.items():
            print(f"\n{category}:")
            if counts:
                for name, count in counts.items():
                    print(f"  - {name}: {count}")
            else:
                print("  (No data)")
        print("--- Verification Complete ---\n")

    except Exception as e:
        logging.error(f"Error during database verification: {e}")


# --- Main Execution ---

def main():
    """Main function to run the database population script."""
    load_dotenv()
    neo4j_uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687") # Use neo4j:// scheme
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password") # Ensure this is set in .env

    if not neo4j_password:
        logging.error("NEO4J_PASSWORD environment variable not set.")
        return

    plants_file = "M.Plants.txt" # Relative path to the data file

    db_conn = None # Initialize db_conn to None
    try:
        db_conn = Neo4jConnection(neo4j_uri, neo4j_user, neo4j_password)

        # Setup schema
        setup_schema(db_conn)

        # Optional: Clear database
        clear_choice = input("Clear existing database before import? (yes/no): ").strip().lower()
        if clear_choice == 'yes':
            confirm_clear = input("ARE YOU SURE you want to delete all data? (yes/no): ").strip().lower()
            if confirm_clear == 'yes':
                clear_database(db_conn)
            else:
                logging.info("Database clear aborted.")
        else:
            logging.info("Skipping database clear.")


        # Parse data
        plants = parse_structured_plants_file(plants_file)
        if not plants:
            logging.error("No plant data parsed. Exiting.")
            return

        # Import data
        logging.info(f"Starting import of {len(plants)} plants...")
        count = 0
        for plant in plants:
            # Check if plant dict is valid before importing
            if plant and isinstance(plant, dict) and 'name' in plant and 'scientific_name' in plant:
                 import_plant_data(db_conn, plant)
                 count += 1
                 if count % 20 == 0:
                     logging.info(f"Imported {count}/{len(plants)} plants...")
            else:
                 logging.warning(f"Skipping invalid plant data structure: {plant}")

        logging.info(f"Finished importing {count} plants.")

        # Create semantic links
        create_cross_relationships(db_conn)

        # Perform cleanup
        cleanup_database(db_conn)

        # Verify
        verify_database_structure(db_conn)

        logging.info("Database population complete.")

    except Exception as e:
        logging.error(f"An error occurred during the process: {e}", exc_info=True)
    finally:
        if db_conn:
            db_conn.close()

if __name__ == "__main__":
    main()
