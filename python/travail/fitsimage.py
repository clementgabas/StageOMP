import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
import pandas as pd
import math
import random as rd
import scipy.stats as st

from python.travail.count_in_cells import *

image_file = "./source/w1_gaussian_1amin.fits"
image_file = "./source/w1_mrl.fits"
# image_file = "./source/w1_mrl.fits"
with fits.open(image_file) as hdu_list:
    hdu_list.info()
hdu = fits.open(image_file)[0]
# wcs = WCS(hdu.header)
image_data = fits.getdata(image_file)

W1_loc = [[30, -11.5], [30, -3.5], [39, -3.5], [39, -11.5]]
extent = [39, 30, -11.5, -3.5]


fig1 = plt.figure(figsize=(12, 12))
# fig1.add_subplot(111, projection=wcs)
ax = fig1.gca()
im = ax.imshow(image_data, extent=extent)
cbar_im = fig1.colorbar(im)
cbar_im.set_label("Signatures de weak-lensing")
"""
fig1 = plot_simulations(n_simul=20_000, coord_champ=W1_loc, n_col=2, n_row=2, lentilles_df=lentilles_WD1_df, coord_W=D1_loc, fill_grid=False, write_num=False,
                 title="Surdensité des lentilles dans les champs W1 & D1 \n20 000 simulations selon une loi uniforme $\mathcal{U}_{W1}$",
                 base_fig=fig1, return_fig=True)

plt.show()
"""
x = lentilles_W1_df["_RAJ2000"]
y = lentilles_W1_df["_DEJ2000"]

deltaX = (max(x)-min(x))
deltaY = (max(y)-min(y))
xmin = min(x) - deltaX
xmax = max(x) + deltaX
ymin = min(y) - deltaY
ymax = max(y) + deltaY
print(xmin, xmax, ymin, ymax)
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = st.gaussian_kde(values, bw_method=0.3)
f = np.reshape(kernel(positions).T, xx.shape)

ax = fig1.gca()


cset = ax.contour(xx, yy, f, cmap='coolwarm')
cbar_gauss = fig1.colorbar(cset, location='left')
cbar_gauss.set_label("Densité de lentilles")

ax.set_xlabel('RA (deg)')
ax.set_ylabel('DEC (deg)')
plt.title('Signatures de weak-lensing du champ W1 et estimation de la densité de lentille')


ax.set_xlim(39, 30)
ax.set_ylim(-11.5, -3.5)

plt.show()
