from python.objets.wobjet import WDegCarre
from python.objets.lentille import LentilleGravita

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits
import pandas as pd
import scipy.stats as scs
import statsmodels.api as sm

from python.plotfunctions import *

#-- chargement des données

# pour éviter le rechargement des données quand on relance tout le fichier sans faire attention

def does_var_exist(var: str):
    if var in globals():
        print(f"{var} already loaded")
        return True
    else:
        return False

"""
if not does_var_exist("w2_already_loaded"):
    data_w2 = pd.read_csv("./source/W2.csv", header=0, sep=';')
    w2_already_loaded = True
if not does_var_exist("w3_already_loaded"):
    data_w3 = pd.read_csv("./source/W3.csv", header=0, sep=';')
    w3_already_loaded = True
if not does_var_exist("w4_already_loaded"):Z
    data_w4 = pd.read_csv("./source/W4.csv", header=0, sep=';')
    w4_already_loaded = True
"""
if not does_var_exist("lenses_aleready_loaded"): 
    file_lentilles = "./source/Lentilles.fits"
    with fits.open(file_lentilles, memmap=True) as hdulist_Lentilles:
        # hdulist_Lentilles.info()
        data_lentilles = hdulist_Lentilles[1].data
        lenses_aleready_loaded = True

lentilles_list = list()
for lentille_coord in data_lentilles:
    lentilles_list.append(LentilleGravita(ra_deg = lentille_coord[0], dec_deg= lentille_coord[1]))    

# --- Carte des exposition et seeing des champs

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


# df4 = all_fields_df.loc[all_fields_df['Field_Name'].str.contains("W4")]
# df4

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

#-- Valeurs de seeing et d'exposition des lentilles
for lentille in lentilles_list:
    seeing_value = lentille.compute_seeing(list_of_objects)
    lentille.set_seeing(seeing_value)
    exposition_value = lentille.compute_exposition(list_of_objects)
    lentille.set_exposition(exposition_value)


# -- premiers histogrammes empiriques
seeing_list_lentille = np.array([lentille.seeing for lentille in lentilles_list])
seeing_list_obj = np.array([obj.seeing for obj in list_of_objects])

seeing_list_lentille_noNA = seeing_list_lentille[~np.isnan(seeing_list_lentille)]
seeing_list_obj_noNA = seeing_list_obj[~np.isnan(seeing_list_obj)]

def centrer_reduit(array):
    return (array-np.mean(array))/np.std(array)


seeing_list_lentille_r = centrer_reduit(seeing_list_lentille_noNA)
seeing_list_obj_r = centrer_reduit(seeing_list_obj_noNA)


bins = np.linspace(0, 1, 30)

plt.hist(seeing_list_lentille_noNA, bins, alpha=0.5, density=True,
         label=f"Répartition empirique des lentilles (n = {len(seeing_list_lentille_noNA)})")
plt.hist(seeing_list_obj_noNA, bins, alpha=0.5, density=True,
         label=f"Répartition théorique des champs (n = {len(seeing_list_obj_noNA)})")
plt.legend(loc = 'upper right')
plt.title(
    "Densité de répartition des lentilles en fonction du seeing sur les champs W1-4")
plt.xlabel("Seeing")
plt.ylabel("Densité")
plt.show()

bw_dict = {'bw': 0.3}
sns.distplot(seeing_list_lentille_noNA, kde=True, bins=bins, rug=True, kde_kws=bw_dict,
            label=f"empirique (n = {len(seeing_list_lentille_noNA)})")
sns.distplot(seeing_list_obj_noNA, kde=True, bins=bins, rug=True, kde_kws=bw_dict,
            label=f"théorique (n = {len(seeing_list_obj_noNA)})")
plt.legend(loc='upper right')
plt.title("Densité de répartition des lentilles en fonction du seeing sur les champs W1-4")
plt.suptitle(
    f"Non parametric Parzen-Rosenblatt estimator - estimation par noyau non paramétrique - bw = {bw_dict['bw']}")
plt.xlabel("Seeing")
plt.ylabel("Densité")
plt.show()

# -- Normalité des échantillons

fig, (axe0, axe1) = plt.subplots(1, 2)
make_global_title(
    fig,
    title="QQ-plot des répartition empiriques et thoériques des lentilles selon le seeing")
sm.qqplot(seeing_list_lentille_r, line='45', ax=axe0, )
sm.qqplot(seeing_list_obj_r, line='45', ax=axe1)
axe0.title.set_text(f"Lentilles (n = {len(seeing_list_lentille_r)})")
axe1.title.set_text(f"Champs (n = {len(seeing_list_obj_r)})")

plt.show()

# T test de Welch
_, p_value = scs.ttest_ind(seeing_list_lentille_r, seeing_list_obj_r, equal_var=False)
p_value #on accepte H_0


"""
# -- Wilcoxon signed rank test
# Nécessite 2 échantillons de même taille
import random
random.shuffle(seeing_list_obj)
seeing_list_obj_bis = seeing_list_obj[0:len(seeing_list_lentille)]
len(seeing_list_lentille) == len(seeing_list_obj_bis)

# Test
import scipy.stats as scs
scs.wilcoxon(x = seeing_list_lentille, y = seeing_list_obj_bis, alternative = "two-sided")
"""

#--- Copie des plots
'''
fig, axes = plt.subplots(2, 2)

make_global_title(
    fig,
    title="Portion de champs et dispositions des lentilles gravitationelles",
    x_title="Ascension droite (deg)",
    y_title="Déclinaison (deg)")


plot_sub_data(axes, [0, 0], data_lentilles["RAJ2000"],
              data_lentilles["DEJ2000"], (30, 40), (-14, -2), "W1")

sc2 = axes[0, 1].scatter(data_w2["RA"], data_w2["DEC"],
                         marker="+", c=data_w2["G"])
#plt.colorbar(sc2)
plot_sub_data(axes, [0, 1], data_lentilles["RAJ2000"],
              data_lentilles["DEJ2000"], (131, 138), (-6, 0), "W2")

sc3 = axes[1, 0].scatter(data_w3["RA"], data_w3["DEC"],
                         marker="+", c=data_w3["G"])
# plt.colorbar(sc3)
plot_sub_data(axes, [1, 0], data_lentilles["RAJ2000"],
              data_lentilles["DEJ2000"], (205, 225), (50, 60), "W3")

sc4 = axes[1, 1].scatter(data_w4["RA"], data_w4["DEC"],
                         marker="+", c=data_w4["G"])
plt.colorbar(sc4)
plot_sub_data(axes, [1, 1], data_lentilles["RAJ2000"],
              data_lentilles["DEJ2000"], (328, 338), (-2, 5), "W4")


plt.show()
'''






















s = plt.scatter(df4["RA"], df4["DEC"], marker='s', c=df4["Seeing"], s = 1500)
plt.colorbar(s)
plt.scatter(data_lentilles["RAJ2000"], data_lentilles["DEJ2000"], marker = '+', c = 'red')
plt.xlim((329.5, 336))
plt.ylim((-1.5, 4.75))

plt.show()
