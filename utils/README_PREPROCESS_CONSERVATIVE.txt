PREPROCESS PDF TO MD - VERSIONE CONSERVATIVA

Cosa cambia
-----------
- Nessun fix adattativo di parole tipo:
  Regu lation -> Regulation
  supervi sion -> supervision

- I fix aggressivi dei glifi Word/PDF sono disattivati di default.
- Restano solo cleanup generici:
  Unicode normalization
  ligature standard
  soft-hyphen
  spazi
  header/footer
  marker pagina
  heading Markdown

Uso standard consigliato
------------------------
python .\utils\preprocess_pdf_to_md.py ".\data\assessment\INBOX\file.pdf"

Uso con output esplicito
------------------------
python .\utils\preprocess_pdf_to_md.py ".\data\assessment\INBOX\file.pdf" --out ".\data\assessment\INBOX\file.clean.md"

Uso solo se il PDF viene da Word ed e' pieno di glifi strani
-----------------------------------------------------------
python .\utils\preprocess_pdf_to_md.py ".\data\assessment\INBOX\file.pdf" --out ".\data\assessment\INBOX\file.clean.md" --aggressive-glyph-fix

Nota
----
Per i PDF ufficiali testuali usa sempre la modalità standard, senza --aggressive-glyph-fix.
