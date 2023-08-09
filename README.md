# Maskinlæring og roing

## Oppsett
### Modeller
Implementasjon  av ML-modeller
- Folder: `models`
- Filer: `rnn_models.py`,`lstm_model.py`,`final_testing_of_rnn_model.py`

### Scripts
- Folder: `scripts`
- Filer:
    - `fix_handle_names.py`
    - `inspect_data.py`: Få et overblikk over datasettet
    - `update_median_rowstroke_time.py`: Oppdater medianverdier av rowing stroke

### Utils
Nyttige funksjoner og variabler
- Folder: `utils`
- Filer: `utils.py`, `variables.py`

### Tester
Testing av koden. Kjør `pytest test_rnn_model.py`
- Folder: `tests`
- Filer: `test_rnn_model.py`

### Annet
- Versjonskrav: `requirements.txt`

## Tensorboard
Kjør `tensorboard --logdir logs/fit`

## Datasett
### Kolonner (totalt 38)
- 1 kolonne linjenummer
- 1 kolonne med tid
- 12*3=36 kolonner med markørinfo: de 12 markørene har 3 kolonner hver (x,y,z)

### Rader
Hver rad er en frame (1/240 sek)

### Tid
Roingen starter omtrent 5 sek ut i opptaket.
Alle roforsøk varer 180 sek
Kan være lurt å ignorere de første og siste 10 sek etter at roingen begynner

### Markører
På roeren:
- lsh = left shoulder
- rsh = right shoulder
- lhi = left hip
- rhi = right hip

På romaskinen:
- lse = left seat
- rse = right seat
- rha = right handle (høyre del av håndtak)
- mha = middle handle (midtre del av håndtak)
- lha = left handle  (venstre del av håndtak)
- pfr = posterior front (bakre del av frontdelen på romaskinen)
- mfr = middle front (midtre del av frontdelen på romaskinen)
- afr = anterior front (fremre del av frontdelen på romaskinen)

### Filnavn
- Elite: id 9 og lavere
- Lavere nivå: id 10 og høyere
