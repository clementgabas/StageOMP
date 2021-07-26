import statsmodels.api as sm
import statsmodels.formula.api as sm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from python.objets.wobjet import WDegCarre
from python.objets.lentille import LentilleGravita
lentilles_list = pickle.load(open("./source_pickle/lentilles_list2", "rb"))
obj_list = pickle.load(open("./source_pickle/fields", "rb"))

# -- Creation des data

Y = [lent.rad for lent in lentilles_list]
X = [[lent.z, lent.seeing, lent.exposition] for lent in lentilles_list]

D = [[lent.seeing, lent.exposition] for lent in lentilles_list]
import pandas as pd
import numpy as np
df = pd.DataFrame(np.array(D), columns=["Seeing", "Exposition"])
file_lentilles2 = "./source/Lentilles.csv"
data_lentilles2 = pd.read_csv(file_lentilles2, header=0, sep=";", skiprows=[1])

df = pd.concat([data_lentilles2, df], axis=1)
df.drop(["_RAB1950", "_DEB1950", "RAJ2000", "DEJ2000", "CFHTLS",
        "Simbad", "n_ID", "Rank"], axis=1, inplace=True)
df.columns = ["Galactic longitude", "Galactic latitude", "Right ascension", "Declination",
              "Ecliptic longitude", "Ecliptic latitude", "ID", "g-band magnitude", "r-band magnitude", "i-band magnitude", "redshift", "redshift uncertainity", "redshift from literature",
              "Arc Radius $R_A$ (arcsec)", r"$R_A$ ($h^{-1}kpc$)", "Detection Type", "Field", "Seeing", "Exposition"]


# -- Carte de corrélation - pearson
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(df.corr(method='pearson'), annot=True, vmin=-1,
            vmax=+1, cmap=cmap, linewidths=.5)
plt.title(f"Matrice de corrélation - r de Pearson (n={len(df)})")
plt.show()

# -- Carte de corrélation - spearman
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(df.corr(method='spearman'), annot=True, vmin=-1,
            vmax=+1, cmap=cmap, linewidths=.5)
plt.title(rf"Matrice de corrélation - $\rho$ de spearman (n={len(df)})")
plt.show()


# ---
def make_sub_band_plot():
    dfm = df[df["redshift"] < 0.5]
    dfp = df[df["redshift"] >= 0.5]

    fig, axes = plt.subplots(3,3)

    # x+y legends
    axes[0,0].set_ylabel("g-r")
    axes[1, 0].set_ylabel("g-i")
    axes[2, 0].set_ylabel("r-i")
    axes[2, 0].set_xlabel("g-r")
    axes[2, 1].set_xlabel("g-i")
    axes[2, 2].set_xlabel("r-i")

    # 0-1
    axes[0, 1].scatter(x=dfm["g-band magnitude"] - dfm["i-band magnitude"],
                    y=dfm["g-band magnitude"]-dfm["r-band magnitude"],
                    c="blue", marker="+",
                    label="z<0.5")
    axes[0, 1].scatter(x=dfp["g-band magnitude"] - dfp["i-band magnitude"],
                    y=dfp["g-band magnitude"]-dfp["r-band magnitude"],
                    c="red", marker="+",
                    label=r"z$\geq$0.5")
    # 0-2
    axes[0, 2].scatter(x=dfm["r-band magnitude"] - dfm["i-band magnitude"],
                    y=dfm["g-band magnitude"]-dfm["r-band magnitude"],
                    c="blue", marker="+")
    axes[0, 2].scatter(x=dfp["r-band magnitude"] - dfp["i-band magnitude"],
                    y=dfp["g-band magnitude"]-dfp["r-band magnitude"],
                    c="red", marker="+")

    # 1-0
    axes[1, 0].scatter(x=dfm["g-band magnitude"] - dfm["r-band magnitude"],
                    y=dfm["g-band magnitude"]-dfm["i-band magnitude"],
                    c="blue", marker="+")
    axes[1, 0].scatter(x=dfp["g-band magnitude"] - dfp["r-band magnitude"],
                    y=dfp["g-band magnitude"]-dfp["i-band magnitude"],
                    c="red", marker="+")

    # 1-2
    axes[1, 2].scatter(x=dfm["r-band magnitude"] - dfm["i-band magnitude"],
                    y=dfm["g-band magnitude"]-dfm["i-band magnitude"],
                    c="blue", marker="+")
    axes[1, 2].scatter(x=dfp["r-band magnitude"] - dfp["i-band magnitude"],
                    y=dfp["g-band magnitude"]-dfp["i-band magnitude"],
                    c="red", marker="+")
    
    # 2-0
    axes[2, 0].scatter(x=dfm["g-band magnitude"] - dfm["r-band magnitude"],
                       y=dfm["r-band magnitude"]-dfm["i-band magnitude"],
                       c="blue", marker="+")
    axes[2, 0].scatter(x=dfp["g-band magnitude"] - dfp["r-band magnitude"],
                       y=dfp["r-band magnitude"]-dfp["i-band magnitude"],
                       c="red", marker="+")
    
    # 2-1
    axes[2, 1].scatter(x=dfm["g-band magnitude"] - dfm["i-band magnitude"],
                       y=dfm["r-band magnitude"]-dfm["i-band magnitude"],
                       c="blue", marker="+")
    axes[2, 1].scatter(x=dfp["g-band magnitude"] - dfp["i-band magnitude"],
                       y=dfp["r-band magnitude"]-dfp["i-band magnitude"],
                       c="red", marker="+")


    fig.legend()
    fig.suptitle("Plots des différences entre bande photométrique en fonction du redshift pour les lentilles gravitationnelles")
    plt.show()
    
make_sub_band_plot()


"""
model = sm.ols(formula = "rad ~ z + expo + z*expo", data=df, missing='drop')
results = model.fit()
print(results.summary())


model = sm.ols(formula="Rad ~ zph + gmag + rmag*imag + e_zph",
               data=data_lentilles2, missing='drop')
model = sm.ols(formula="Rad ~ .", data=data_lentilles2, missing='drop')

results = model.fit()
print(results.summary())
model2 = sm.ols(formula="Rad ~ imag*rmag + RA", data=data_lentilles2, missing='drop')
results2 = model2.fit()
print(results2.summary())

c = data_lentilles2.corr()


plt.scatter(data_lentilles2["Rad"], data_lentilles2["zph"])
plt.show()
plt.scatter(df["see"], df["expo"])
"""