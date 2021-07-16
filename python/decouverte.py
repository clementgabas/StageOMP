import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits
import pandas as pd

# --- Prises des données

file_lentilles = "./source/Lentilles.fits"
file_W2 = "./source/pdz_W2_270912.fits"
file_W3 = "./source/pdz_W3_270912.fits"
file_W4 = "./source/pdz_W4_270912.fits"


with fits.open(file_lentilles, memmap = True) as hdulist_Lentilles:
    # hdulist_Lentilles.info()
    data_lentilles = hdulist_Lentilles[1].data

with fits.open(file_W2, memmap=True) as hdulist_w2:
    # hdulist_w2.info()
    data_w2 = hdulist_w2[1].data

with fits.open(file_W3, memmap=True) as hdulist_W3:
    # hdulist_W3.info()
    data_w3 = hdulist_W3[1].data

with fits.open(file_W4, memmap=True) as hdulist_W4:
    # hdulist_W4.info()
    data_w4 = hdulist_W4[1].data


# --- Plot functions 
def plot_sub_data(subplot_axe_object, axe: [int, int], data1, data2, xlim: (int, int), ylim: (int, int), title: str, invert_x: bool = True, invert_y: bool = False):
    subplot_axe_object[axe[0], axe[1]].set_xlim(xlim)
    subplot_axe_object[axe[0], axe[1]].set_ylim(ylim)
    subplot_axe_object[axe[0], axe[1]].scatter(data1, data2, marker = "+", c = 'red')
    subplot_axe_object[axe[0], axe[1]].title.set_text(title)
    if invert_x:
        subplot_axe_object[axe[0], axe[1]].invert_xaxis()
    if invert_y:
        subplot_axe_object[axe[0], axe[1]].invert_yaxis()

def make_global_title(subplot_fig_object, title, x_title, y_title):
    subplot_fig_object.suptitle(title)
    subplot_fig_object.supxlabel(x_title)
    subplot_fig_object.supylabel(y_title)

    

    

# --- Plots des 4 subplots avec positions des lentilles + positions des objets selon leur Z
    
fig, axes = plt.subplots(2, 2)

make_global_title(
    fig,
    title="Portion de champs et dispositions des lentilles gravitationelles",
    x_title="Ascension droite (deg)",
    y_title="Déclinaison (deg)")


plot_sub_data(axes, [0, 0], data_lentilles["RAJ2000"],
              data_lentilles["DEJ2000"], (30, 40), (-14, -2), "W1")

sc2 = axes[0, 1].scatter(data_w2["RA"], data_w2["DEC"],
                         marker="+", c=data_w2["Z_FINAL"])
#plt.colorbar(sc2)
plot_sub_data(axes, [0, 1], data_lentilles["RAJ2000"],
              data_lentilles["DEJ2000"], (132, 138), (-6, 0), "W2")

sc3 = axes[1, 0].scatter(data_w3["RA"], data_w3["DEC"], marker="+", c=data_w3["Z_FINAL"])
# plt.colorbar(sc3)
plot_sub_data(axes, [1, 0], data_lentilles["RAJ2000"],
              data_lentilles["DEJ2000"], (205, 225), (50, 60), "W3")

sc4 = axes[1, 1].scatter(data_w4["RA"], data_w4["DEC"], marker="+", c=data_w4["Z_FINAL"])
plt.colorbar(sc4)
plot_sub_data(axes, [1, 1], data_lentilles["RAJ2000"],
              data_lentilles["DEJ2000"], (328, 338), (-2, 5), "W4")


plt.show()
    
    
# -- fichier en .out
file_w4_out = "./source/photozCFHTLS-W4_270912.out"

header_list = ["Id", "alpha", "delta", "flag", "StarGal", "r2", "final photo-z", "zPDF", "zPDF_l68", "zPDF_u68", "chi2_zPDF", "mod", "ebv", "NbFilt", "zMin", "zl68", "zu68", "chi2_best", "zp_2", "chi2_2", "mods", "chis", "zq", "chiq", "modq", "U", "G", "R", "I", "Y", "Z", "eU", "eG", "eR", "eI", "eY", "eZ", "MU", "MG", "MR", "MI", "MY", "MZ"]

df_w4 = pd.read_csv(file_w4_out, header = None, names = header_list, delim_whitespace = True)
    
df_w4.head()

df_w4_fits__RA = pd.DataFrame(data_w4["RA"], columns = ["RA"])
df_w4_fits__DEC = pd.DataFrame(data_w4["DEC"], columns=["DEC"])
df_w4_fits__ID = pd.DataFrame(data_w4["ID"], columns=["Id"])
ddf = pd.concat([df_w4_fits__ID, df_w4_fits__DEC, df_w4_fits__RA], axis = 1)

df4 = pd.merge(df_w4, ddf)
df4.head()


