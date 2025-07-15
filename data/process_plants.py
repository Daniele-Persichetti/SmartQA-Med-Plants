import re

def parse_medicinal_plants_file(input_filename, output_filename):
    """
    Parse a file containing medicinal plant information and convert it to a structured format.
    """
    # Read the entire file content
    with open(input_filename, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split the content by plant entries
    # Plants are typically numbered and have a heading like "## 1. Aloe Vera"
    plant_pattern = r'## (\d+)\. ([^(]+)(?:\(([^)]+)\))?(.*?)(?=## \d+\.|$)'
    plant_matches = re.finditer(plant_pattern, content, re.DOTALL)
    
    output_text = ""
    plant_id = 0
    
    for match in plant_matches:
        plant_id += 1
        plant_number = match.group(1)
        plant_name = match.group(2).strip()
        scientific_name = match.group(3) if match.group(3) else extract_scientific_name(match.group(0))
        plant_content = match.group(0)
        
        # Extract information using patterns
        taxonomy_info = extract_section(plant_content, "**Taxonomic Information:**", "**Physical Characteristics:**")
        family = extract_field(taxonomy_info, "**Family:**")
        common_names = extract_field(taxonomy_info, "**Common Names:**")
        
        physical_info = extract_section(plant_content, "**Physical Characteristics:**", "**Geographic Distribution:**")
        morphology = extract_field(physical_info, "**Morphology:**")
        
        geographic_info = extract_section(plant_content, "**Geographic Distribution:**", "**Medicinal Properties:**")
        distribution = geographic_info.strip()
        
        medicinal_info = extract_section(plant_content, "**Medicinal Properties:**", "**Usage Information:**")
        compounds = extract_field(medicinal_info, "**Active Compounds:**")
        effects = extract_field(medicinal_info, "**Therapeutic Effects:**")
        
        usage_info = extract_section(plant_content, "**Usage Information:**", "---")
        methods = extract_field(usage_info, "**Preparation Methods:**")
        side_effects = extract_field(usage_info, "**Side Effects:**")
        
        # Format the output
        plant_entry = f"""===PLANT===
ID: {plant_id}
NAME: {plant_name}
SCIENTIFIC_NAME: {scientific_name}
FAMILY: {family}
COMMON_NAMES: {common_names}
===CHARACTERISTICS===
MORPHOLOGY: {morphology}
DISTRIBUTION: {distribution}
===MEDICINAL===
COMPOUNDS: {compounds}
EFFECTS: {effects}
===USAGE===
METHODS: {methods}
SIDE_EFFECTS: {side_effects}
===END===

"""
        output_text += plant_entry
    
    # Write the formatted output to a new file
    with open(output_filename, 'w', encoding='utf-8') as file:
        file.write(output_text)
    
    return plant_id  # Return the number of plants processed

def extract_scientific_name(text):
    """Extract scientific name in italics from text"""
    scientific_match = re.search(r'\*([^*]+)\*', text)
    if scientific_match:
        return scientific_match.group(1).strip()
    return ""

def extract_section(text, start_marker, end_marker):
    """Extract a section of text between two markers"""
    pattern = f"{re.escape(start_marker)}(.*?){re.escape(end_marker)}"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_field(text, field_marker):
    """Extract a field from text based on its marker"""
    if field_marker in text:
        # Get content after the field marker
        field_content = text.split(field_marker, 1)[1].strip()
        # If there's another field marker, only take content up to that
        next_field = re.search(r"\*\*[A-Za-z ]+:\*\*", field_content)
        if next_field:
            field_content = field_content[:next_field.start()].strip()
        return field_content
    return ""

def clean_text(text):
    """Clean up the text by removing extra whitespace, etc."""
    # Replace multiple whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Run the parser
if __name__ == "__main__":
    input_file = "M.Plants_List.txt"  # Replace with your input filename
    output_file = "medicinal_plants_structured.txt"
    
    num_plants = parse_medicinal_plants_file(input_file, output_file)
    print(f"Successfully processed {num_plants} medicinal plants.")
