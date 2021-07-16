import numpy as np
import pandas as pd
from astropy.io import fits

HEADER_LIST_OUT = ["Id", "alpha", "delta", "flag", "StarGal", "r2", "final photo-z", "zPDF", "zPDF_l68", "zPDF_u68", "chi2_zPDF", "mod", "ebv", "NbFilt", "zMin", "zl68", "zu68",
               "chi2_best", "zp_2", "chi2_2", "mods", "chis", "zq", "chiq", "modq", "U", "G", "R", "I", "Y", "Z", "eU", "eG", "eR", "eI", "eY", "eZ", "MU", "MG", "MR", "MI", "MY", "MZ"]

# --- Prises des données

file_lentilles = "./source/Lentilles.fits"
file_W2 = "./source/pdz_W2_270912.fits"
file_W3 = "./source/pdz_W3_270912.fits"
file_W4 = "./source/pdz_W4_270912.fits"


with fits.open(file_lentilles, memmap=True) as hdulist_Lentilles:
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


# ---- W2
file_w2_in = "./source/photozCFHTLS-W2_270912.out"


df_w2 = pd.read_csv(file_w2_in, header=None,
                    names=HEADER_LIST_OUT, delim_whitespace=True)

df_w2_fits__RA = pd.DataFrame(data_w2["RA"], columns=["RA"])
df_w2_fits__DEC = pd.DataFrame(data_w2["DEC"], columns=["DEC"])
df_w2_fits__ID = pd.DataFrame(data_w2["ID"], columns=["Id"])
ddf2 = pd.concat([df_w2_fits__ID, df_w2_fits__DEC, df_w2_fits__RA], axis=1)

df2 = pd.merge(df_w2, ddf2)

file_w2_out = "./source/W2.csv"
df2.to_csv(file_w2_out, sep=';')


# ---- W3
file_w3_in = "./source/photozCFHTLS-W3_270912.out"


df_w3 = pd.read_csv(file_w3_in, header=None,
                    names=HEADER_LIST_OUT, delim_whitespace=True)

df_w3_fits__RA = pd.DataFrame(data_w3["RA"], columns=["RA"])
df_w3_fits__DEC = pd.DataFrame(data_w3["DEC"], columns=["DEC"])
df_w3_fits__ID = pd.DataFrame(data_w3["ID"], columns=["Id"])
ddf3 = pd.concat([df_w3_fits__ID, df_w3_fits__DEC, df_w3_fits__RA], axis=1)

df3 = pd.merge(df_w3, ddf3)

file_w3_out = "./source/W3.csv"
df3.to_csv(file_w3_out, sep=';')


# ---- W4
file_w4_in = "./source/photozCFHTLS-W4_270912.out"

df_w4 = pd.read_csv(file_w4_in, header=None,
                    names=HEADER_LIST_OUT, delim_whitespace=True)

# df_w4.head()

df_w4_fits__RA = pd.DataFrame(data_w4["RA"], columns=["RA"])
df_w4_fits__DEC = pd.DataFrame(data_w4["DEC"], columns=["DEC"])
df_w4_fits__ID = pd.DataFrame(data_w4["ID"], columns=["Id"])
ddf4 = pd.concat([df_w4_fits__ID, df_w4_fits__DEC, df_w4_fits__RA], axis=1)

df4 = pd.merge(df_w4, ddf4)
# df4.head()

file_w4_out = "./source/W4.csv"
df4.to_csv(file_w4_out, sep=';')

