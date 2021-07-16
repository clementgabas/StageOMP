#--- load packages
from python.objets.wobjet import WDegCarre
from python.objets.lentille import LentilleGravita

import numpy as np
from astropy.io import fits
import pandas as pd

import pickle

# --- Données fields W1-4 exposition et seeing
all_fields_data_file = "./source/W1-4_CFHTLS_esposition_seeing.csv"
all_fields_df = pd.read_csv(all_fields_data_file, header=0, sep=";")
all_fields_df = all_fields_df[["Field_Name",
                               "RA", "DEC", "Seeing", "Exposition"]]

for i in range(len(all_fields_df.index)):
    all_fields_df["RA"].iloc[i] = str(
        all_fields_df["RA"].iloc[i]).replace(",", ".")
    all_fields_df["DEC"].iloc[i] = str(
        all_fields_df["DEC"].iloc[i]).replace(",", ".")

all_fields_df["RA"] = pd.to_numeric(all_fields_df["RA"])
all_fields_df["DEC"] = pd.to_numeric(all_fields_df["DEC"])

list_of_objects = list()
for i in range(len(all_fields_df.index)):
    list_of_objects.append(
        WDegCarre(
            ra_deg=all_fields_df["RA"].iloc[i],
            dec_deg=all_fields_df["DEC"].iloc[i],
            cadran=int(str(all_fields_df["Field_Name"].iloc[i])[1]),
            exposition=all_fields_df["Exposition"].iloc[i],
            seeing=all_fields_df["Seeing"].iloc[i]
        )
    )
    
# --- enregistrement des données fields W
pickle.dump(list_of_objects, open("./source_pickle/fields", "wb"))


# --- Chargement des données lentilles
file_lentilles = "./source/Lentilles.fits"
with fits.open(file_lentilles, memmap=True) as hdulist_Lentilles:
    # hdulist_Lentilles.info()
    data_lentilles = hdulist_Lentilles[1].data
    lenses_aleready_loaded = True

# --- Création de la liste de toutes les lentilles
lentilles_list = list()
for lentille_coord in data_lentilles:
    lentilles_list.append(LentilleGravita(
        ra_deg=lentille_coord[0], dec_deg=lentille_coord[1]))
    
#-- Valeurs de seeing et d'exposition des lentilles
for lentille in lentilles_list:
    seeing_value = lentille.compute_seeing(list_of_objects)
    lentille.set_seeing(seeing_value)
    exposition_value = lentille.compute_exposition(list_of_objects)
    lentille.set_exposition(exposition_value)

# --- enregistrement des données lentilles
pickle.dump(lentilles_list, open("./source_pickle/lentilles_list", "wb"))
