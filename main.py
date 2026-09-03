import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ====================================
# config
# ====================================
DATA_DIR = '/Users/kritikabhat/Documents/InformationRetrievalSystems/dm_spl_monthly_update_feb2026/prescription'
OUTPUT_DIR = 'drug_corpus_json'

NAMESPACE = {'ns': 'urn:hl7-org:v3'}  # FDA SPL XMLs use this HL7 namespace

# LOINC CODES RELIABLE TO IDENTIFY SPL SECTIONS
SECTION_CODES = {
    "34084-4": "side_effects",
    "34073-7": "drug_interactions",
    "34068-7": "dosage_and_administration"
}

# FALLBACK IF NO LOINC CODE IS PRESENT
TITLE_FALLBACKS = {
    "adverse reactions": "side_effects",
    "drug interactions": "drug_interactions",
    "dosage and administration": "dosage_and_administration",
}

# ====================================
# helper functions
# ====================================

def clean_text(text):
    """Clean and normalize text."""
    return ' '.join(text.split())


def extract_section_text(section):
    if section is None:
        return ""

    return clean_text(" ".join(section.itertext()))

# ====================================
# extract drug name
# ====================================
# drug name lives in <manufacturedProduct><name> (or <manufacturedMedicine><name>)
def extract_drug_name(root):
    for xpath in [
        './/ns:manufacturedProduct/ns:name',
        './/ns:manufacturedMedicine/ns:name',
        './/ns:manufacturedProduct/ns:manufacturedMedicine/ns:name',
    ]:
        elem = root.find(xpath,NAMESPACE)
        if elem is not None:
            name = extract_section_text(elem)
            if name and name.strip():
                return name.strip()
            
    # fallback to the document title
    title_elem = root.find('.//ns:title',NAMESPACE)
    if title_elem is not None:
        name = extract_section_text(title_elem)
        if name and name.strip():
            return name.strip()
        
    return "unknown"


# ====================================
# extract section body
# ====================================
def extract_section_body(section):
    # get the body text of a section without the section's own <title>.
    text_elem = section.find('ns:text',NAMESPACE)
    if text_elem is None:
        parts = []
        for child in section:
            tag = child.tag.split('}')[-1]
            if tag in ('title','code'):
                continue
            parts.append(extract_section_text(child))
        return clean_text(" ".join(p for p in parts if p))
    
    return extract_section_text(text_elem)



# ====================================
# xml parsing and data extraction
# ====================================

def parse_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root =  tree.getroot()

        # drug name
        drug_name = extract_drug_name(root)

        extracted = {
            "side_effects": "",
            "drug_interactions": "",
            "dosage_and_administration": ""
        }

        for section in root.findall('.//ns:section', NAMESPACE):
            key = None
            # Try LOINC CODE
            code_elem = section.find('ns:code',NAMESPACE)
            if code_elem is not None:
                code = code_elem.get('code')
                if code in SECTION_CODES:
                    key = SECTION_CODES[code]

            # fall back to title text match
            if key is None:
                title_elem = section.find('ns:title',NAMESPACE)
                if title_elem is not None:
                    title_text = extract_section_text(title_elem).lower()
                    for needle,target in TITLE_FALLBACKS.items():
                        if needle in title_text:
                            key = target
                            break
            
            if key is None:
                continue

            if not extracted[key]:
                extracted[key] = extract_section_body(section)

        return {
            "drug_name": drug_name,
            "side_effects": extracted["side_effects"],
            "drug_interactions": extracted["drug_interactions"],
            "dosage_and_administration": extracted["dosage_and_administration"],
        }

    except ET.ParseError as e:
        print(f"Error parsing {file_path}: {e}")
        return None
    
# ====================================
# main processing loop
# ====================================

documents = []
for root_dir, dirs,files in os.walk(DATA_DIR):
    for file in files:

        # only xml files
        if file.lower().endswith('.xml'):
            xml_path = os.path.join(root_dir, file)
            parsed = parse_xml(xml_path)
            if parsed:

                if (
                    parsed["side_effects"] or 
                    parsed["drug_interactions"] or 
                    parsed["dosage_and_administration"]
                ):
                    documents.append(parsed)

print(f"Extracted {len(documents)} documents with relevant sections.")

with open(OUTPUT_DIR,'w',encoding='utf-8') as f:
    json.dump(documents, f, indent=2, ensure_ascii=False)
print(f"Saved extracted data to {OUTPUT_DIR}")