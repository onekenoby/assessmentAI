BATCH SEMPLIFICATI V2 - RESET DOCUMENTI DAI DB

Problema risolto
----------------
Il container ingestion_assessment non vedeva la cartella locale utils, quindi Python cercava:

/app/utils/reset_dbs_one.py

ma il file non esisteva dentro il container.

Soluzione applicata
-------------------
I batch montano temporaneamente la cartella locale:

%cd%\utils

dentro il container come:

/app/utils

Comando singolo file
--------------------
docker compose --profile assessment --profile ingestion run --rm -v "%cd%\utils:/app/utils:ro" ingestion_assessment python /app/utils/reset_dbs_one.py %*

Comando reset totale
--------------------
docker compose --profile assessment --profile ingestion run --rm -v "%cd%\utils:/app/utils:ro" ingestion_assessment python /app/utils/reset_dbs.py %*

Dove mettere i batch
--------------------
Nella root del progetto, nella stessa cartella di docker-compose.yml.

Prerequisito
------------
I file devono esistere localmente qui:

utils\reset_dbs_one.py
utils\reset_dbs.py
