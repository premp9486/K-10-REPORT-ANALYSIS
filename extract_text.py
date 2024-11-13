# scripts/extract_text.py

from PyPDF2 import PdfReader
import os

def extract_text(pdf_path, output_folder):
    pdf_reader = PdfReader(pdf_path)
    os.makedirs(output_folder, exist_ok=True)
    
    for i, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        if page_text:
            with open(os.path.join(output_folder, f"page_{i+1}.txt"), "w", encoding="utf-8") as text_file:
                text_file.write(page_text)
    print("Text extraction completed.")

if __name__ == "__main__":
    pdf_path = './data/RIL-Integrated-Annual-Report-2023-24.pdf'
    output_folder = './data/extracted_text/'
    extract_text(pdf_path, output_folder)
