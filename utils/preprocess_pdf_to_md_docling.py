import os
import time
from pathlib import Path
from docling.document_converter import DocumentConverter


print("Working Directory attuale:", os.getcwd())


input_pdf = Path(r"..\data\assessment\inbox\ENISA_1.0.pdf")
output_md = Path(r"..\data\assessment\inbox\ENISA_1.0.docling.md")


print("⏳ Inizializzazione del convertitore...")
# Avvio del timer
start_time = time.perf_counter()

converter = DocumentConverter()

print(f"📄 Lettura ed elaborazione di: {input_pdf.name}")
print("🔄 Conversione in corso (potrebbe richiedere alcuni istanti)...")

# Operazione bloccante
result = converter.convert(str(input_pdf))

print("✍️ Generazione del file Markdown...")
md_text = result.document.export_to_markdown()
output_md.write_text(md_text, encoding="utf-8")

# Fine del timer e calcolo
end_time = time.perf_counter()
execution_time = end_time - start_time

print("-" * 40)
print(f"✅ OK: File salvato in {output_md}")
print(f"⏱️ Tempo di esecuzione: {execution_time:.2f} secondi")