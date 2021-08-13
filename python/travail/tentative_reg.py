import statsmodels.api as sm
import statsmodels.formula.api as sm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
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
              "Ecliptic longitude", "Ecliptic latitude", "ID", "g_band_magnitude", "r_band_magnitude", "i_band_magnitude", "redshift", "redshift uncertainity", "redshift from literature",
              "Arc Radius $R_A$ (arcsec)", r"$R_A$ ($h^{-1}kpc$)", "Detection Type", "Field", "Seeing", "Exposition"]

#len(df[df['Field'] == 'W4']) 

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
    axes[0, 1].scatter(x=dfm["g_band_magnitude"] - dfm["i_band_magnitude"],
                    y=dfm["g_band_magnitude"]-dfm["r_band_magnitude"],
                    c="blue", marker="+",
                    label="z<0.5")
    axes[0, 1].scatter(x=dfp["g_band_magnitude"] - dfp["i_band_magnitude"],
                    y=dfp["g_band_magnitude"]-dfp["r_band_magnitude"],
                    c="red", marker="+",
                    label=r"z$\geq$0.5")
    # 0-2
    axes[0, 2].scatter(x=dfm["r_band_magnitude"] - dfm["i_band_magnitude"],
                    y=dfm["g_band_magnitude"]-dfm["r_band_magnitude"],
                    c="blue", marker="+")
    axes[0, 2].scatter(x=dfp["r_band_magnitude"] - dfp["i_band_magnitude"],
                    y=dfp["g_band_magnitude"]-dfp["r_band_magnitude"],
                    c="red", marker="+")

    # 1-0
    axes[1, 0].scatter(x=dfm["g_band_magnitude"] - dfm["r_band_magnitude"],
                    y=dfm["g_band_magnitude"]-dfm["i_band_magnitude"],
                    c="blue", marker="+")
    axes[1, 0].scatter(x=dfp["g_band_magnitude"] - dfp["r_band_magnitude"],
                    y=dfp["g_band_magnitude"]-dfp["i_band_magnitude"],
                    c="red", marker="+")

    # 1-2
    axes[1, 2].scatter(x=dfm["r_band_magnitude"] - dfm["i_band_magnitude"],
                    y=dfm["g_band_magnitude"]-dfm["i_band_magnitude"],
                    c="blue", marker="+")
    axes[1, 2].scatter(x=dfp["r_band_magnitude"] - dfp["i_band_magnitude"],
                    y=dfp["g_band_magnitude"]-dfp["i_band_magnitude"],
                    c="red", marker="+")
    
    # 2-0
    axes[2, 0].scatter(x=dfm["g_band_magnitude"] - dfm["r_band_magnitude"],
                       y=dfm["r_band_magnitude"]-dfm["i_band_magnitude"],
                       c="blue", marker="+")
    axes[2, 0].scatter(x=dfp["g_band_magnitude"] - dfp["r_band_magnitude"],
                       y=dfp["r_band_magnitude"]-dfp["i_band_magnitude"],
                       c="red", marker="+")
    
    # 2-1
    axes[2, 1].scatter(x=dfm["g_band_magnitude"] - dfm["i_band_magnitude"],
                       y=dfm["r_band_magnitude"]-dfm["i_band_magnitude"],
                       c="blue", marker="+")
    axes[2, 1].scatter(x=dfp["g_band_magnitude"] - dfp["i_band_magnitude"],
                       y=dfp["r_band_magnitude"]-dfp["i_band_magnitude"],
                       c="red", marker="+")


    fig.legend()
    fig.suptitle("Plots des différences entre bandes photométriques en fonction du redshift pour les lentilles gravitationnelles")
    plt.show()
    
make_sub_band_plot()

df2 = df[['Right ascension', 'Declination', 'g_band_magnitude',
          'r_band_magnitude', 'i_band_magnitude', 'redshift', 'Arc Radius $R_A$ (arcsec)', 'Field', 'Seeing', 'Exposition']]
df2["gMi"] = df2['g_band_magnitude']-df2['i_band_magnitude']
df2["gMr"] = df2['g_band_magnitude']-df2['r_band_magnitude']
df2["rMi"] = df2['r_band_magnitude']-df2['i_band_magnitude']

df2.to_csv(r"data_reg.csv", index=False)

df_train, df_test = sklearn.model_selection.train_test_split(df2, test_size=0.2, shuffle=True)
Z_train = df_train[['redshift']]
Z_test = df_test[['redshift']]
df_train2 = df_train.drop('redshift', axis = 1, inplace = False)
df_test2 = df_test.drop('redshift', axis = 1, inplace = False)

import pymc3 as pm
formula = "redshift ~ gMi + gMr + rMi"
with pm.Model() as normal_model:
    
    family = pm.glm.families.Normal()
    
    pm.GLM.from_formula(formula, data = df_train, family = family)
    
    normal_trace = pm.sample(draws=2000, chains=2, tune=500, njobs=-1)
pm.traceplot(normal_trace)


# models
dfb = df[["g_band_magnitude", "r_band_magnitude", "i_band_magnitude", "redshift"]]
model = sm.ols("redshift ~ g_band_magnitude + r_band_magnitude + i_band_magnitude", data=dfb).fit()
print(model.summary())



# ---- correlation a 2 points
import astroML.correlation as amlc
def plot_2pts_corr(N_bootstrap=5000, rd_seed=90):
    fig, axes = plt.subplots(2, 2)

    # _bins = np.logspace(-1, .5, 20)
    _bins = np.linspace(0, 5, 20)

    def errorfill(x, y, yerr, color='blue', alpha_fill=0.3, ax=None):
        ax = ax if ax is not None else plt.gca()
        if color is None:
            color = next(ax._get_lines.prop_cycler)['color']
        if np.isscalar(yerr) or len(yerr) == len(y):
            ymin = y - yerr
            ymax = y + yerr
        elif len(yerr) == 2:
            ymin, ymax = yerr
        ax.plot(x, y, color=color)
        ax.fill_between(x, ymax, ymin, color=color,
                        alpha=alpha_fill, label='erreur')


    def correct_corr(corr):
        n_corr = list()
        for v in corr:
            if v > 1:
                n_corr.append(1)
            elif v < -1:
                n_corr.append(-1)
            else:
                n_corr.append(v)
        return np.array(n_corr)


    # W1
    dfw1 = df[df["Field"] == "W1"]
    #dfd1 = df[(df["Field"] == "D1") & (df["Arc Radius $R_A$ (arcsec)"] >= 2.8)]
    # dfw1 = pd.concat([dfw1, dfd1], axis=0)

    corr, dcorr, bootstraps = amlc.bootstrap_two_point_angular(
        ra=dfw1["Right ascension"], dec=dfw1["Declination"], bins=_bins, method="landy-szalay", Nbootstraps=N_bootstrap,  random_state=rd_seed)
    corr = correct_corr(corr)


    axes[0, 0].hlines(y=0, xmin=min(_bins), xmax=max(_bins), colors='red', linestyles='dotted')
    axes[0, 0].hlines(y=1, xmin=min(_bins), xmax=max(_bins), colors='red', linestyles='dotted')
    axes[0, 0].hlines(y=-1, xmin=min(_bins), xmax=max(_bins), colors='red', linestyles='dotted')

    axes[0, 0].hlines(y=np.nanmean(corr), xmin=min(_bins), xmax=max(_bins), colors='black', linestyles='solid', label=f'mean correlation = {round(np.nanmean(corr), 2)}')
    axes[0, 0].hlines(y=np.nanmedian(corr), xmin=min(_bins), xmax=max(_bins), colors='grey', linestyles='solid', label=f'median correlation = {round(np.nanmedian(corr), 2)}')


    axes[0, 0].errorbar(_bins[1: len(_bins)], corr, yerr=dcorr, label='corrélation à 2 points', color='blue')

    #errorfill(_bins[1: len(_bins)], corr, yerr=dcorr)

    axes[0, 0].legend(loc='lower right')
    axes[0, 0].set_title(f"W1 (n={len(dfw1)})")
    # W2
    dfw2 = df[df["Field"] == "W2"]

    corr, dcorr, bootstraps = amlc.bootstrap_two_point_angular(
        ra=dfw2["Right ascension"], dec=dfw2["Declination"], bins=_bins, method="landy-szalay", Nbootstraps=N_bootstrap,  random_state=rd_seed)
    corr = correct_corr(corr)

    axes[0, 1].hlines(y=0, xmin=min(_bins), xmax=max(_bins), colors='red', linestyles='dotted')
    axes[0, 1].hlines(y=1, xmin=min(_bins), xmax=max(
        _bins), colors='red', linestyles='dotted')
    axes[0, 1].hlines(y=-1, xmin=min(_bins), xmax=max(_bins),
            colors='red', linestyles='dotted')

    axes[0, 1].hlines(y=np.nanmean(corr), xmin=min(_bins), xmax=max(_bins), colors='black',
            linestyles='solid', label=f'mean correlation = {round(np.nanmean(corr), 2)}')
    axes[0, 1].hlines(y=np.nanmedian(corr), xmin=min(_bins), xmax=max(_bins), colors='grey',
            linestyles='solid', label=f'median correlation = {round(np.nanmedian(corr), 2)}')


    axes[0, 1].errorbar(_bins[1: len(_bins)], corr, yerr=dcorr,
                label='corrélation à 2 points', color='blue')

    #errorfill(_bins[1: len(_bins)], corr, yerr=dcorr)

    axes[0, 1].legend(loc='lower right')
    axes[0, 1].set_title(f"W2 (n={len(dfw2)})")
    # W3 
    dfw3 = df[df["Field"] == "W3"]
    #dfd3 = df[df["Field"] == "D3"]
    #dfw1 = pd.concat([dfw3, dfd3], axis=0)

    corr, dcorr, bootstraps = amlc.bootstrap_two_point_angular(
        ra=dfw3["Right ascension"], dec=dfw3["Declination"], bins=_bins, method="landy-szalay", Nbootstraps=N_bootstrap, random_state=rd_seed)
    corr = correct_corr(corr)

    axes[1, 0].hlines(y=0, xmin=min(_bins), xmax=max(
        _bins), colors='red', linestyles='dotted')
    axes[1, 0].hlines(y=1, xmin=min(_bins), xmax=max(
        _bins), colors='red', linestyles='dotted')
    axes[1, 0].hlines(y=-1, xmin=min(_bins), xmax=max(_bins),
            colors='red', linestyles='dotted')

    axes[1, 0].hlines(y=np.nanmean(corr), xmin=min(_bins), xmax=max(_bins), colors='black',
            linestyles='solid', label=f'mean correlation = {round(np.nanmean(corr), 2)}')
    axes[1, 0].hlines(y=np.nanmedian(corr), xmin=min(_bins), xmax=max(_bins), colors='grey',
            linestyles='solid', label=f'median correlation = {round(np.nanmedian(corr), 2)}')


    axes[1, 0].errorbar(_bins[1: len(_bins)], corr, yerr=dcorr,
                label='corrélation à 2 points', color='blue')

    # errorfill(_bins[1: len(_bins)], corr, yerr=dcorr)

    axes[1, 0].legend(loc='lower right')
    axes[1, 0].set_title(f"W3 (n={len(dfw3)})")

    #W4
    dfw4 = df[df["Field"] == "W4"]

    corr, dcorr, bootstraps = amlc.bootstrap_two_point_angular(
        ra=dfw4["Right ascension"], dec=dfw4["Declination"], bins=_bins, method="landy-szalay", Nbootstraps=N_bootstrap, random_state=rd_seed)
    corr = correct_corr(corr)

    axes[1, 1].hlines(y=0, xmin=min(_bins), xmax=max(
        _bins), colors='red', linestyles='dotted')
    axes[1, 1].hlines(y=1, xmin=min(_bins), xmax=max(
        _bins), colors='red', linestyles='dotted')
    axes[1, 1].hlines(y=-1, xmin=min(_bins), xmax=max(_bins),
            colors='red', linestyles='dotted')

    axes[1, 1].hlines(y=np.nanmean(corr), xmin=min(_bins), xmax=max(_bins), colors='black',
            linestyles='solid', label=f'mean correlation = {round(np.nanmean(corr), 2)}')
    axes[1, 1].hlines(y=np.nanmedian(corr), xmin=min(_bins), xmax=max(_bins), colors='grey',
            linestyles='solid', label=f'median correlation = {round(np.nanmedian(corr), 2)}')


    axes[1, 1].errorbar(_bins[1: len(_bins)], corr, yerr=dcorr,
                label='corrélation à 2 points', color='blue')

    # errorfill(_bins[1: len(_bins)], corr, yerr=dcorr)

    axes[1, 1].legend(loc='lower right')
    axes[1, 1].set_title(f"W4 (n={len(dfw4)})")
    
    fig.suptitle("Fonction de corrélation à 2 points $\zeta$ par estimateur de Landy-Szalay sur les différents champs W1-4")
    fig.supxlabel("r")
    fig.supylabel("Corrélation")
    
    plt.show()
plot_2pts_corr(rd_seed=15)

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
