# parser.py
# Step 2 — Load and read a PDF lab report
# This file handles everything related to extracting text from PDFs

import pdfplumber
import os

import base64
from groq import Groq
from dotenv import load_dotenv
load_dotenv()


def load_pdf(file_path: str) -> str:
    """
    Reads a PDF file and returns all the text as one big string.
    Works page by page and joins them together.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file: {file_path}")

    full_text = []

    with pdfplumber.open(file_path) as pdf:
        print(f"Total pages found: {len(pdf.pages)}")

        for i, page in enumerate(pdf.pages):
            text = page.extract_text()

            if text:
                full_text.append(f"--- PAGE {i + 1} ---\n{text}")
            else:
                full_text.append(f"--- PAGE {i + 1} --- (no text found, may be scanned)")

    return "\n\n".join(full_text)


def load_pdf_by_page(file_path: str) -> list:
    """
    Same as load_pdf but returns a list of pages instead of one big string.
    Each item in the list is the text of one page.
    Useful for debugging — you can print just one page at a time.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file: {file_path}")

    pages = []

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            pages.append({
                "page_number": i + 1,
                "text": text if text else "",
                "has_text": bool(text)
            })

    return pages


# ── Quick test ──────────────────────────────────────────────
# This block only runs when you execute this file directly
# It will NOT run when other files import parser.py

if __name__ == "__main__":

    # Change this to match your actual filename
    PDF_PATH = r"data\sample_reports\sample_report.pdf"

    print("=" * 50)
    print("STEP 2 — Loading PDF")
    print("=" * 50)

    # Test 1: load full text
    print("\n[Test 1] Loading full text...")
    full_text = load_pdf(PDF_PATH)
    print(f"Total characters extracted: {len(full_text)}")
    print("\nFirst 500 characters of the report:")
    print("-" * 40)
    print(full_text[:500])

    # Test 2: load page by page
    print("\n[Test 2] Loading page by page...")
    pages = load_pdf_by_page(PDF_PATH)
    for page in pages:
        status = "✓ has text" if page["has_text"] else "✗ no text (scanned?)"
        print(f"  Page {page['page_number']}: {status}")


import base64

def load_image(file_path):
    """
    Reads an image file and converts it to base64.
    This is what vision models need to see the image.
    """
    with open(file_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    return image_data


def describe_image(file_path):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    image_data = load_image(file_path)

    ext = file_path.split(".")[-1].lower()
    mime_type = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "You are a medical imaging expert. Analyze this medical image in detail. Describe: type of image, body part, any visible abnormalities, overall impression."
                    }
                ]
            }
        ]
    )

    return response.choices[0].message.content

# Test it
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text = load_pdf(sys.argv[1])
        print(text[:500])
    else:
        print("Testing image description...")
        desc = describe_image(r"data\sample_reports\hands.jpg")
        print(desc)