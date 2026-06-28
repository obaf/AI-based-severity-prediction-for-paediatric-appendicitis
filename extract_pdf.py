import pdfplumber

with pdfplumber.open("Interpretatble and intervenable ultrasonography-based ML models for pediatric appendicitis.pdf") as pdf:
    print("NUM PAGES:", len(pdf.pages))
    with open("paper_text.txt", "w", encoding="utf-8") as f:
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            f.write(f"\n\n===== PAGE {i+1} =====\n\n")
            f.write(txt)
print("done")
