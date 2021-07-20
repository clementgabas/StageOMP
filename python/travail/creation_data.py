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
    cadran = lentille.compute_cadran(list_of_objects)
    lentille.set_cadran(cadran)
    
for i in range(len(lentilles_list)):
    curr_lent = lentilles_list[i]
    curr_lent_ppv = curr_lent.closest_object(
        lentilles_list[0:i]+lentilles_list[i+1:])
    curr_lent.set_ppv(curr_lent_ppv)

# --- enregistrement des données lentilles
pickle.dump(lentilles_list, open("./source_pickle/lentilles_list", "wb"))



# --- Chargement des données lentilles 2
file_lentilles2 = "./source/Lentilles.csv"
data_lentilles2 = pd.read_csv(file_lentilles2, header=0, sep=";", skiprows=[1])
data_lentilles2 = data_lentilles2[['_RAJ2000', '_DEJ2000', 'RA', 'Rad', 'zph', 'Fld']]
data_lentilles2["RA"] = pd.to_numeric(data_lentilles2["RA"])
data_lentilles2["Rad"] = pd.to_numeric(data_lentilles2["Rad"])



lentilles_list2 = list()
for i in range(len(data_lentilles2)):
    lentilles_list2.append(
        LentilleGravita(
            ra_deg=data_lentilles2["_RAJ2000"].iloc[i],
            dec_deg=data_lentilles2["_DEJ2000"].iloc[i],
            field=data_lentilles2["Fld"].iloc[i],
            z_redshift=data_lentilles2["zph"].iloc[i],
            arc_radius=data_lentilles2["Rad"].iloc[i]
        )
    )
    
#-- Valeurs de seeing et d'exposition des lentilles
for lentille in lentilles_list2:
    seeing_value = lentille.compute_seeing(list_of_objects)
    lentille.set_seeing(seeing_value)
    exposition_value = lentille.compute_exposition(list_of_objects)
    lentille.set_exposition(exposition_value)
    
for i in range(len(lentilles_list2)):
    curr_lent = lentilles_list2[i]
    curr_lent_ppv = curr_lent.closest_object(
        lentilles_list2[0:i]+lentilles_list2[i+1:])
    curr_lent.set_ppv(curr_lent_ppv)

# --- enregistrement des données lentilles
pickle.dump(lentilles_list2, open("./source_pickle/lentilles_list2", "wb"))
