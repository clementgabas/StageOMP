
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits
from astropy.table import Table
import pandas as pd
import math


file_lentilles = "./source/Lentilles.fits"
with fits.open(file_lentilles, memmap=True) as hdulist_Lentilles:
    hdulist_Lentilles.info()
    lentilles_df = pd.DataFrame(hdulist_Lentilles[1].data)
    
file_lentilles_2 = "./source/Lentilles.csv"
lentilles_df_2 = pd.read_csv(file_lentilles_2, sep=";", header =[0], skiprows = [1])

class Lense:
    
    def __init__(self, RA, DEC, Z=-1, RE=-1, ID="None"):
        self.ra = RA
        self.dec = DEC
        self.z = Z
        self.redshift = self.z
        self.re = RE
        self.ID = ID
        
    def distance(self, other):
        delta_RA = self.ra - other.ra
        DEC_sep = self.dec - other.dec
        avg_DEC = DEC_sep / 2
        RA_sep = delta_RA * math.cos(avg_DEC)
        d2 = DEC_sep*DEC_sep + RA_sep*RA_sep
        return math.sqrt(d2)
        
    def compute_NN(self, list_of_galaxies):
        curent_NN = list_of_galaxies[0]
        curent_NN_dist = self.distance(list_of_galaxies[0])
        for galaxy in list_of_galaxies:
            dist = self.distance(galaxy)
            if dist < curent_NN_dist:
                current_NN_dist = dist
                curent_NN = galaxy
        return (curent_NN, current_NN_dist)
    
    def compute_number_neighbor_in_R(self, R, list_of_galaxies):
        number_of_neighbor = 0
        for galaxy in list_of_galaxies:
            if self.distance(galaxy) < R:
                number_of_neighbor += 1
        return number_of_neighbor 
    
class Galaxy:
    
    def __init__(self, RA, DEC, Z=-1, ID="None"):
        self.ra = RA
        self.dec = DEC
        self.z = Z
        self.redshift = self.z
        self.ID = ID


# W4 ----------------------
file_W4 = "./source/pdz_W4_270912.fits"
dat = Table.read(file_W4, format='fits')
W4_df = dat.to_pandas()
W4_df = W4_df[["ID", "RA", "DEC", "Z_MED_PDZ"]]

lentilles_W4_df = lentilles_df_2[lentilles_df_2['Fld'] == "W4"]

W4_galaxies_list = list()
for index, row in W4_df.iterrows():
    gal = Galaxy(RA=row["RA"], DEC=row["DEC"], Z=row["Z_MED_PDZ"], ID=row["ID"])
    W4_galaxies_list.append(gal)

W4_lenses_list = list()
for index, row in lentilles_W4_df.iterrows():
    gal = Lense(RA=row["_RAJ2000"], DEC=row["_DEJ2000"], Z=row["zph"], RE=row["Rad"]/3600, ID=row["ID"])
    W4_lenses_list.append(gal)

W4_results_dict = {}
for lense in W4_lenses_list:
    W4_results_dict[lense.ID] = {
        "3R": 0,
        "4R": 0
        }
    W4_results_dict[lense.ID]["3R"] = lense.compute_number_neighbor_in_R(R=3*lense.re, list_of_galaxies=W4_galaxies_list)
    W4_results_dict[lense.ID]["4R"] = lense.compute_number_neighbor_in_R(R=4*lense.re, list_of_galaxies=W4_galaxies_list)
    W4_results_dict[lense.ID]["RA"] = lense.ra
    W4_results_dict[lense.ID]["DEC"] = lense.dec
W4_results_df = pd.DataFrame.from_dict(W4_results_dict, orient='index')
W4_results_df['Field'] = "W4"  
    
plt.scatter(x=W4_results_df["RA"], y=W4_results_df["DEC"], marker='+', c='red')
for i, txt in enumerate(W4_results_df["3R"]):
    plt.annotate(txt, (list(W4_results_df["RA"])[i], list(W4_results_df["DEC"])[i]))
for i, txt in enumerate(W4_results_df["4R"]):
    plt.annotate(txt, (list(W4_results_df["RA"])[i]+.05, list(W4_results_df["DEC"])[i]+.05))
plt.show()


# W3 -----------
file_W3 = "./source/pdz_W3_270912.fits"
dat = Table.read(file_W3, format='fits')
W3_df = dat.to_pandas()
W3_df = W3_df[["ID", "RA", "DEC", "Z_MED_PDZ"]]

lentilles_W3_df = lentilles_df_2[lentilles_df_2['Fld'] == "W3"]

W3_galaxies_list = list()
for index, row in W3_df.iterrows():
    gal = Galaxy(RA=row["RA"], DEC=row["DEC"], Z=row["Z_MED_PDZ"], ID=row["ID"])
    W3_galaxies_list.append(gal)

W3_lenses_list = list()
for index, row in lentilles_W3_df.iterrows():
    gal = Lense(RA=row["_RAJ2000"], DEC=row["_DEJ2000"], Z=row["zph"], RE=row["Rad"]/3600, ID=row["ID"])
    W3_lenses_list.append(gal)

W3_results_dict = {}
for lense in W3_lenses_list:
    W3_results_dict[lense.ID] = {
        "3R": 0,
        "4R": 0
        }
    W3_results_dict[lense.ID]["3R"] = lense.compute_number_neighbor_in_R(R=3*lense.re, list_of_galaxies=W3_galaxies_list)
    W3_results_dict[lense.ID]["4R"] = lense.compute_number_neighbor_in_R(R=4*lense.re, list_of_galaxies=W3_galaxies_list)
    W3_results_dict[lense.ID]["RA"] = lense.ra
    W3_results_dict[lense.ID]["DEC"] = lense.dec
W3_results_df = pd.DataFrame.from_dict(W3_results_dict, orient='index')
W3_results_df['Field'] = "W3"

plt.scatter(x=W3_results_df["RA"], y=W3_results_df["DEC"], marker='+', c='red')
for i, txt in enumerate(W3_results_df["3R"]):
    plt.annotate(txt, (list(W3_results_df["RA"])[i], list(W3_results_df["DEC"])[i]))
for i, txt in enumerate(W3_results_df["4R"]):
    plt.annotate(txt, (list(W3_results_df["RA"])[i]+.1, list(W3_results_df["DEC"])[i]+.1))
plt.show()

# --- W2
file_W2 = "./source/pdz_W2_270912.fits"
dat = Table.read(file_W2, format='fits')
W2_df = dat.to_pandas()
W2_df = W2_df[["ID", "RA", "DEC", "Z_MED_PDZ"]]

lentilles_W2_df = lentilles_df_2[lentilles_df_2['Fld'] == "W2"]

W2_galaxies_list = list()
for index, row in W2_df.iterrows():
    gal = Galaxy(RA=row["RA"], DEC=row["DEC"],
                 Z=row["Z_MED_PDZ"], ID=row["ID"])
    W2_galaxies_list.append(gal)

W2_lenses_list = list()
for index, row in lentilles_W2_df.iterrows():
    gal = Lense(RA=row["_RAJ2000"], DEC=row["_DEJ2000"],
                Z=row["zph"], RE=row["Rad"]/3600, ID=row["ID"])
    W2_lenses_list.append(gal)

W2_results_dict = {}
for lense in W2_lenses_list:
    W2_results_dict[lense.ID] = {
        "3R": 0,
        "4R": 0
    }
    W2_results_dict[lense.ID]["3R"] = lense.compute_number_neighbor_in_R(
        R=3*lense.re, list_of_galaxies=W2_galaxies_list)
    W2_results_dict[lense.ID]["4R"] = lense.compute_number_neighbor_in_R(
        R=4*lense.re, list_of_galaxies=W2_galaxies_list)
    W2_results_dict[lense.ID]["RA"] = lense.ra
    W2_results_dict[lense.ID]["DEC"] = lense.dec
W2_results_df = pd.DataFrame.from_dict(W2_results_dict, orient='index')
W2_results_df['Field'] = "W2"

plt.scatter(x=W2_results_df["RA"], y=W2_results_df["DEC"], marker='+', c='red')
for i, txt in enumerate(W2_results_df["3R"]):
    plt.annotate(txt, (list(W2_results_df["RA"])[
                 i], list(W2_results_df["DEC"])[i]))
for i, txt in enumerate(W2_results_df["4R"]):
    plt.annotate(txt, (list(W2_results_df["RA"])[
                 i]+.1, list(W2_results_df["DEC"])[i]+.1))
plt.show()


# --- W1
file_W1_1 = "./source/pdz_W1_270912_part1.fits"
file_W1_2 = "./source/pdz_W1_270912_part2.fits"

dat_1 = Table.read(file_W1_1, format='fits')
W1_1_df = dat_1.to_pandas()
W1_1_df = W1_1_df[["ID", "RA", "DEC", "Z_MED_PDZ"]]

dat_2 = Table.read(file_W1_2, format='fits')
W1_2_df = dat_2.to_pandas()
W1_2_df = W1_2_df[["ID", "RA", "DEC", "Z_MED_PDZ"]]

W1_df = pd.concat([W1_1_df, W1_2_df], axis=0)

lentilles_W1_df = lentilles_df_2[lentilles_df_2['Fld'] == "W1"]


W1_galaxies_list = list()
for index, row in W1_df.iterrows():
    gal = Galaxy(RA=row["RA"], DEC=row["DEC"],
                 Z=row["Z_MED_PDZ"], ID=row["ID"])
    W1_galaxies_list.append(gal)

W1_lenses_list = list()
for index, row in lentilles_W1_df.iterrows():
    gal = Lense(RA=row["_RAJ2000"], DEC=row["_DEJ2000"],
                Z=row["zph"], RE=row["Rad"]/3600, ID=row["ID"])
    W1_lenses_list.append(gal)

W1_results_dict = {}
for lense in W1_lenses_list:
    W1_results_dict[lense.ID] = {
        "3R": 0,
        "4R": 0
    }
    W1_results_dict[lense.ID]["3R"] = lense.compute_number_neighbor_in_R(
        R=3*lense.re, list_of_galaxies=W1_galaxies_list)
    W1_results_dict[lense.ID]["4R"] = lense.compute_number_neighbor_in_R(
        R=4*lense.re, list_of_galaxies=W1_galaxies_list)
    W1_results_dict[lense.ID]["RA"] = lense.ra
    W1_results_dict[lense.ID]["DEC"] = lense.dec
W1_results_df = pd.DataFrame.from_dict(W1_results_dict, orient='index')
W1_results_df['Field'] = "W1"

plt.scatter(x=W1_results_df["RA"], y=W1_results_df["DEC"], marker='+', c='red')
for i, txt in enumerate(W1_results_df["3R"]):
    plt.annotate(txt, (list(W1_results_df["RA"])[
                 i], list(W1_results_df["DEC"])[i]))
for i, txt in enumerate(W1_results_df["4R"]):
    plt.annotate(txt, (list(W1_results_df["RA"])[
                 i]+.1, list(W1_results_df["DEC"])[i]+.1))
plt.show()


W1_results_df["3Rd"] = W1_results_df["3R"]/64072
W1_results_df["4Rd"] = W1_results_df["4R"]/64072
W2_results_df["3Rd"] = W2_results_df["3R"]/72939
W2_results_df["4Rd"] = W2_results_df["4R"]/72939
W3_results_df["3Rd"] = W3_results_df["3R"]/65305
W3_results_df["4Rd"] = W3_results_df["4R"]/65305
W4_results_df["3Rd"] = W4_results_df["3R"]/70618
W4_results_df["4Rd"] = W4_results_df["4R"]/70618

allW_results_df = pd.concat(
    [W1_results_df, W2_results_df, W3_results_df, W4_results_df], axis=0)

allW_results_df.to_csv('./source/ppv_results.csv')

allW_results_df = pd.read_csv('./source/ppv_results.csv', header = 0, index_col=0)
W1_results_df = allW_results_df[allW_results_df["Field"]=="W1"]
W2_results_df = allW_results_df[allW_results_df["Field"]=="W2"]
W3_results_df = allW_results_df[allW_results_df["Field"]=="W3"]
W4_results_df = allW_results_df[allW_results_df["Field"] == "W4"]




# -- global
from scipy import stats
from sklearn import *

_bins = np.linspace(0, max(allW_results_df['4R']), 50)
__bins = np.linspace(-10, max(allW_results_df['4R']), 50000)

plt.hist([allW_results_df['4R'], allW_results_df['3R']],
         bins=_bins, label=['4r', '3r'], density=True)
kde4 = stats.gaussian_kde(allW_results_df['4R'], bw_method=0.03)
#plt.plot(__bins, kde4(__bins), color='blue', label='kde density 4R')
kde3 = stats.gaussian_kde(allW_results_df['3R'], bw_method=0.03)
#plt.plot(__bins, kde3(__bins), color='orange', label='kde density 3R')
plt.legend()
plt.title("Nombre de galaxies voisines pour chaque lentille dans un rayon de 3 et 4 $R_E$ pour les champs W1-4")
plt.xlabel("Nombre de galaxies voisines / densité de galaxies par $deg^2$ dans le champ")
plt.show()


__bins = np.linspace(0, max(allW_results_df['4R']), 50000)
plt.plot(__bins, kde4(__bins), color='green', label='kde density 4R')
plt.plot(__bins, kde3(__bins), color='red', label='kde density 3R')
plt.title("Nombre de galaxies voisines pour chaque lentille dans un rayon de 3 et 4 $R_E$ pour les champs W1-4")
plt.xlabel("Nombre de galaxies voisines / densité de galaxies par $deg^2$ dans le champ")

k_gamma, theta_gamma = 3, 2
# plt.plot(__bins, stats.gamma.pdf(__bins, a=k_gamma, scale=theta_gamma), label='gamma')

# mu_lognorm, sigma_lognorm = 5, 2
# plt.plot(__bins, stats.lognorm.pdf(__bins, s=sigma_lognorm, loc=mu_lognorm), label='lognormale')

df_chi2 = 4
plt.plot(__bins, stats.chi2.pdf(__bins, df=df_chi2), label='chi2')

plt.plot(__bins, stats.norm.pdf(__bins, loc=df_chi2, scale = (2*df_chi2)**1/2), label='gauss')

plt.legend()
plt.show()


fig, (ax1, ax2) = plt.subplots(2, 1)
_bins = np.linspace(0, max(allW_results_df['3R']), 50)
__bins = np.linspace(0, max(allW_results_df['3R']), 50000)

ax1.hist([W1_results_df['3R'], W2_results_df['3R'], W3_results_df['3R'], W4_results_df['3R']],
         bins=_bins, label=['W1', 'W2', 'W3', 'W4'], density=True,alpha=.5)

kde1 = stats.gaussian_kde(W1_results_df['3R'], bw_method=1)
ax1.plot(__bins, kde1(__bins), color='blue')
kde2 = stats.gaussian_kde(W2_results_df['3R'], bw_method=1)
ax1.plot(__bins, kde2(__bins), color='orange')
kde3 = stats.gaussian_kde(W3_results_df['3R'], bw_method=1)
ax1.plot(__bins, kde3(__bins), color='green')
kde4 = stats.gaussian_kde(W4_results_df['3R'], bw_method=1)
ax1.plot(__bins, kde4(__bins), color='red')


ax1.legend()
ax1.set_title("Nombre de galaxies voisines pour chaque lentille dans un rayon de 3 $R_E$ pour les champs W1-4")


__bins = np.linspace(0, max(allW_results_df['4R']), 50000)
ax2.hist([W1_results_df['4R'], W2_results_df['4R'], W3_results_df['4R'], W4_results_df['4R']],
         bins=_bins, label=['W1', 'W2', 'W3', 'W4'], density=True, alpha=.5)

kde1 = stats.gaussian_kde(W1_results_df['4R'])
ax2.plot(__bins, kde1(__bins), color='blue')
kde2 = stats.gaussian_kde(W2_results_df['4R'])
ax2.plot(__bins, kde2(__bins), color='orange')
kde3 = stats.gaussian_kde(W3_results_df['4R'])
ax2.plot(__bins, kde3(__bins), color='green')
kde4 = stats.gaussian_kde(W4_results_df['4R'])
ax2.plot(__bins, kde4(__bins), color='red')


ax2.legend()
ax2.set_title("Nombre de galaxies voisines pour chaque lentille dans un rayon de 4 $R_E$ pour les champs W1-4")
ax2.set_xlabel(
    "Nombre de galaxies voisines dans le champ")
plt.show()




fig, (ax1, ax2) = plt.subplots(2, 1)
_bins = np.linspace(0, max(allW_results_df['3Rd']), 50)
__bins = np.linspace(-0.001, max(allW_results_df['3Rd']), 50000)
ax1.hist([W1_results_df['3Rd'], W2_results_df['3Rd'], W3_results_df['3Rd'], W4_results_df['3Rd']],
         bins=_bins, label=['W1', 'W2', 'W3', 'W4'], density=True, alpha=.5,)

kde1 = stats.gaussian_kde(W1_results_df['3Rd'])
ax1.plot(__bins, kde1(__bins), color='blue')
kde2 = stats.gaussian_kde(W2_results_df['3Rd'])
ax1.plot(__bins, kde2(__bins), color='orange')
kde3 = stats.gaussian_kde(W3_results_df['3Rd'])
ax1.plot(__bins, kde3(__bins), color='green')
kde4 = stats.gaussian_kde(W4_results_df['3Rd'])
ax1.plot(__bins, kde4(__bins), color='red')


ax1.legend()
ax1.set_title(
    "Nombre de galaxies voisines pour chaque lentille dans un rayon de 3 $R_E$ pour les champs W1-4")


__bins = np.linspace(-0.001, max(allW_results_df['4Rd']), 50000)
ax2.hist([W1_results_df['4Rd'], W2_results_df['4Rd'], W3_results_df['4Rd'], W4_results_df['4Rd']],
         bins=_bins, label=['W1', 'W2', 'W3', 'W4'], density=True, alpha=.5)

kde1 = stats.gaussian_kde(W1_results_df['4Rd'])
ax2.plot(__bins, kde1(__bins), color='blue')
kde2 = stats.gaussian_kde(W2_results_df['4Rd'])
ax2.plot(__bins, kde2(__bins), color='orange')
kde3 = stats.gaussian_kde(W3_results_df['4Rd'])
ax2.plot(__bins, kde3(__bins), color='green')
kde4 = stats.gaussian_kde(W4_results_df['4Rd'])
ax2.plot(__bins, kde4(__bins), color='red')


ax2.legend()
ax2.set_title(
    "Nombre de galaxies voisines pour chaque lentille dans un rayon de 4 $R_E$ pour les champs W1-4")
ax2.set_xlabel(
    "Nombre de galaxies voisines / densité de galaxies par $deg^2$ dans le champ")
plt.show()


from scipy import stats

res = stats.probplot(x=allW_results_df["3Rd"], dist=stats.chi2, sparams=(5,), plot=plt)
plt.show()

res = stats.probplot(x=allW_results_df["3Rd"], dist=stats.lognorm, sparams=(
    np.mean(allW_results_df["3Rd"]), np.std(allW_results_df["3Rd"])**2), plot=plt)
plt.show()


res = stats.probplot(x=allW_results_df["3Rd"], dist=stats.gamma, sparams=(1.5,50), plot=plt)
plt.show()

