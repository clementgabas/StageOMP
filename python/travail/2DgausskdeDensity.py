import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from astropy.io import fits
from astropy.table import Table
import pandas as pd
import math
import random as rd
import scipy.stats as st

from python.travail.count_in_cells import *

# W1
x = lentilles_W1_df["_RAJ2000"]
y = lentilles_W1_df["_DEJ2000"]
plt.scatter(x, y)
plt.show()

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

fig1 = plt.figure(figsize=(12, 12))
ax = fig1.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])

cset = ax.contour(xx, yy, f, colors='k')
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.title('2D Gaussian Kernel density estimation')
plot_simulations(n_simul=20_000, coord_champ=W1_loc, n_col=2, n_row=2, lentilles_df=lentilles_W1_df, coord_W=D1_loc,
                 title="Surdensité des lentilles dans le champ W1 \n20 000 simulations selon une loi uniforme $\mathcal{U}_{W1}$ \n2D Gaussian Kernel density estimation",
                 base_fig=fig1, save_title="pics_surdensite_locale_W1_kde", save=False)

"""
fig = plt.figure(figsize=(13, 7))
ax = plt.axes(projection='3d')
w = ax.plot_wireframe(xx, yy, f)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('PDF')
ax.set_title('Wireframe plot of Gaussian 2D KDE')
plt.show()
"""

# W2
x = lentilles_W2_df["_RAJ2000"]
y = lentilles_W2_df["_DEJ2000"]
plt.scatter(x, y)
plt.show()

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
kernel = st.gaussian_kde(values, bw_method=0.5)
f = np.reshape(kernel(positions).T, xx.shape)

fig2 = plt.figure(figsize=(12, 12))
ax = fig2.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
cset = ax.contour(xx, yy, f, colors='k')
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.title('2D Gaussian Kernel density estimation')
plot_simulations(n_simul=20_000, coord_champ=W2_loc, n_col=15, n_row=15, lentilles_df=lentilles_W2_df,
                 title="Surdensité des lentilles dans le champ W2 \n20 000 simulations selon une loi uniforme $\mathcal{U}_{W1}$ \n2D Gaussian Kernel density estimation",
                 base_fig=fig2, save_title="pics_surdensite_locale_WD2_kde", save=True)

# W2
x = lentilles_W3_df["_RAJ2000"]
y = lentilles_W3_df["_DEJ2000"]
plt.scatter(x, y)
plt.show()

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
kernel = st.gaussian_kde(values, bw_method=0.5)
f = np.reshape(kernel(positions).T, xx.shape)

fig3 = plt.figure(figsize=(12, 12))
ax = fig3.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
cset = ax.contour(xx, yy, f, colors='k')
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.title('2D Gaussian Kernel density estimation')
plot_simulations(n_simul=20_000, coord_champ=W3_loc, n_col=15, n_row=15, lentilles_df=lentilles_W3_df, coord_W=D3_loc,
                 title="Surdensité des lentilles dans le champ W3 \n20 000 simulations selon une loi uniforme $\mathcal{U}_{W1}$ \n2D Gaussian Kernel density estimation",
                 base_fig=fig3, save_title="pics_surdensite_locale_WD3_kde", save=True)
