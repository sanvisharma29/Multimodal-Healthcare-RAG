import sys
sys.path.append("src")
#from langchain_text_splitters import RecursiveCharacterTextSplitter

# These are the section headers we saw in the report
SECTION_HEADERS = [
    "Complete Blood Count",
    "Blood Group",
    "Lipid Profile",
    "Liver Function",
    "Kidney Function",
    "Thyroid",
    "Blood Sugar",
    "Urine",
    "Vitamin",
    "Iron",
    "Electrolytes",
]

def split_by_sections(text):
    """
    Splits the full report text into sections.
    Each section becomes a separate chunk.
    """
    chunks = []
    current_section = "General"
    current_text = ""

    for line in text.split("\n"):
        # Check if this line is a section header
        matched_section = None
        for header in SECTION_HEADERS:
            if header.lower() in line.lower():
                matched_section = header
                break

        if matched_section:
            # Save the previous section before starting new one
            if current_text.strip():
                chunks.append({
                    "section": current_section,
                    "text": current_text.strip()
                })
            # Start new section
            current_section = matched_section
            current_text = line + "\n"
        else:
            current_text += line + "\n"

    # Don't forget the last section
    if current_text.strip():
        chunks.append({
            "section": current_section,
            "text": current_text.strip()
        })

    return chunks


def add_metadata(chunks, patient_name, report_date):
    """
    Adds patient info to every chunk.
    This is what gets stored alongside the text in the vector DB.
    """
    for chunk in chunks:
        chunk["patient_name"] = patient_name
        chunk["report_date"] = report_date
    return chunks


# Test it
if __name__ == "__main__":
    from parser import load_pdf

    text = load_pdf(r"data\sample_reports\sample_report.pdf")
    chunks = split_by_sections(text)
    chunks = add_metadata(chunks, "Lyubochka Svetka", "20-Feb-2023")

    print(f"Total chunks created: {len(chunks)}")
    print()
    for chunk in chunks:
        print(f"Section : {chunk['section']}")
        print(f"Patient : {chunk['patient_name']}")
        print(f"Date    : {chunk['report_date']}")
        print(f"Text    : {chunk['text'][:100]}...")
        print("-" * 40)

def chunk_image_description(description, image_filename):
    """
    Takes the vision model's text description of an image
    and converts it into chunks ready for ChromaDB.
    """
    chunks = [{
        "section": "Medical Image Analysis",
        "text": description,
        "patient_name": "Patient",
        "report_date": "Unknown"
    }]
    return chunks