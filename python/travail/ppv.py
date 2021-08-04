
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

file_W4 = "./source/pdz_W4_270912.fits"
dat = Table.read(file_W4, format='fits')
W4_df = dat.to_pandas()
W4_df = W4_df[["ID", "RA", "DEC", "Z_MED_PDZ"]]

lentilles_W4_df = lentilles_df_2[lentilles_df_2['Fld'] == "W4"]

class Lense:
    
    def __init__(self, RA, DEC, Z=-1, RE=-1):
        self.ra = RA
        self.dec = DEC
        self.z = Z
        self.redshift = self.z
        self.re = RE
        
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
    
    def __init__(self, RA, DEC, Z=-1):
        self.ra = RA
        self.dec = DEC
        self.z = Z
        self.redshift = self.z

W4_galaxies_list = list()
for index, row in W4_df.iterrows():
    gal = Galaxy(RA=row["RA"], DEC=row["DEC"], Z=row["Z_MED_PDZ"])
    W4_galaxies_list.append(gal)

W4_lenses_list = list()
for index, row in lentilles_W4_df.iterrows():
    gal = Lense(RA=row["_RAJ2000"], DEC=row["_DEJ2000"], Z=row["zph"], RE=row["Rad"]/3600)
    W4_lenses_list.append(gal)

R3_list = list()
for lense in W4_lenses_list:
    R3_list.append(lense.compute_number_neighbor_in_R(R=3*lense.re, list_of_galaxies=W4_galaxies_list))
   
R4_list = list()
for lense in W4_lenses_list:
    R4_list.append(lense.compute_number_neighbor_in_R(R=4*lense.re, list_of_galaxies=W4_galaxies_list))
 
print(R3_list, R4_list)    
x = range(10)
plt.plot(x, R3_list)
plt.plot(x, R4_list)
plt.show()