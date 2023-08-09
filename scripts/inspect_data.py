# Reading the rowing data using pandas. Another option: import csv
import pandas as pd

"""
INFORMASJON OM DATASETTET

KOLONNER (totalt 38)
- 1 kolonne "trajectory types" (linjenummer)
- 1 kolonne med tid
- 12*3=36 kolonner med markørinfo: de 12 markørene har 3 kolonner hver (x,y,z)

RADER
Hver rad er en frame (1/240 sek)

Roingen starter omtrent 5 sek ut i opptaket "test_data_id7_mocap.csv"
Alle roforsøk varer 180 sek

Kan være lurt å ignorere de første og siste 10 sek etter at roingen begynner

MARKØRNAVN

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

FILNAVN
- Elite: id7, id9
- Lavere nivå: id10, id11
"""

# Load file into a pandas DataFrame
#data = pd.read_csv("src/test_data_id7_mocap.csv")
data = pd.read_csv("src/id7_phml_c4_mocap.csv",sep=";")
#print(data.info())

# Normal python indexing is possible
print(data[:3])
# But using pandas methods is probably better

# Get the first 5 rows (5 is the deafault)
print(data.head())
# Get the first 2 rows
print(data.head(2))
# Get the last 5 rows
print(data.tail())
# Get row 0
#print(data.loc[0])
# Get row 3 and row 5
print(data.loc[[3, 5]])

# Get column by name
print(data["lsh_x"])

# Get single value from the DataFrame
print("first value of lsh_x:",data.at[0,"lsh_x"])

# Get mean and median of a column
print("mean",data["lsh_x"].mean())
print("median",data["lsh_x"].median())

# Convert DataFrame to numpy array
print(data.to_numpy())

# Pandas also has methods for cleaning data, statistics, plotting, etc.
