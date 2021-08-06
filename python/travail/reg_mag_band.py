import statsmodels.api as sm
import statsmodels.formula.api as sm
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import sklearn
import numpy as np
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


df["gMi"] = df["g_band_magnitude"] - df["i_band_magnitude"]
df["gMr"] = df["g_band_magnitude"] - df["r_band_magnitude"]
df["rMi"] = df["r_band_magnitude"] - df["i_band_magnitude"]


#--- Plot de ce qu'on va étudier

def make_base_plot(data_df=df, Y="None", _color='blue', _title=str()):
    
    
    given_color = False
    if not isinstance(_color, str):
        given_color = True
        n_color = []
        for i in range(len(_color)):
            v = _color[i]
            if v == 0:
                n_color.append("blue")
            elif v ==1:
                n_color.append("yellow")
            elif v == 2:
                n_color.append("red")
            elif v == 3:
                n_color.append("cyan")
        _color = n_color
        
    given_Y = False
    if not isinstance(Y, str):
        given_Y = True
        Y = np.array(Y)
        q0, q1, q2, q3 = np.quantile(Y, 0.25), np.quantile(Y, 0.50), np.quantile(Y, 0.75), max(Y)
        colormap = np.array(['blue', 'cyan', 'yellow', 'red'])
        Y_cat = []
        for v in Y:
            v0, v1, v2, v3 = abs(v-q0), abs(v-q1), abs(v-q2), abs(v-q3)
            m = min([v0, v1, v2, v3])
            if v0 == m: Y_cat.append(0)
            elif v1 == m: Y_cat.append(1)
            elif v2 == m: Y_cat.append(2)
            elif v3 == m: Y_cat.append(3)
        categories = np.array(Y_cat)
        _color=colormap[categories]
        
    fig, axes = plt.subplots(3, 3)

    axes[0, 0].set_yticklabels([])
    axes[0, 0].set_xticklabels([])
    axes[0, 0].get_xaxis().set_visible(False)
    axes[1, 1].axis('off')
    axes[2, 2].set_yticklabels([])
    axes[2, 2].set_xticklabels([])
    axes[2, 2].get_yaxis().set_visible(False)

    # x+y legends
    axes[0, 0].set_ylabel("g-r")
    axes[1, 0].set_ylabel("g-i")
    axes[2, 0].set_ylabel("r-i")
    axes[2, 0].set_xlabel("g-r")
    axes[2, 1].set_xlabel("g-i")
    axes[2, 2].set_xlabel("r-i")
    


    # 0-1
    axes[0, 1].scatter(x=data_df["gMi"],
                       y=data_df["gMr"],
                       c=_color, marker="+")

    # 0-2
    axes[0, 2].scatter(x=data_df["rMi"],
                       y=data_df["gMr"],
                       c=_color, marker="+")


    # 1-0
    axes[1, 0].scatter(x=data_df["gMr"],
                       y=data_df["gMi"],
                       c=_color, marker="+")


    # 1-2
    axes[1, 2].scatter(x=data_df["rMi"],
                       y=data_df["gMi"],
                       c=_color, marker="+")


    # 2-0
    axes[2, 0].scatter(x=data_df["gMr"],
                       y=data_df["rMi"],
                       c=_color, marker="+")


    # 2-1
    axes[2, 1].scatter(x=data_df["gMi"],
                       y=data_df["rMi"],
                       c=_color, marker="+")
    
    if given_Y:
        if not isinstance(q3, (float, int, np.int64)): q3 = q3[0]
        blue_patch = mpatches.Patch(color='blue', label=r"z$\leq${}".format(round(q0, 2)))
        cyan_patch = mpatches.Patch(color='cyan', label=r"z$\approx${}".format(round(q1, 2)))
        yellow_patch = mpatches.Patch(color='yellow', label=r"z$\approx${}".format(round(q2, 2)))
        red_patch = mpatches.Patch(color='red', label=r"z$\approx${}".format(round(q3, 2)))
        fig.legend(handles=[blue_patch, cyan_patch, yellow_patch, red_patch],
                   loc='center')
    elif given_color:
        blue_patch = mpatches.Patch(color='blue', label=r"class 0")
        cyan_patch = mpatches.Patch(color='cyan', label=r"class 3")
        yellow_patch = mpatches.Patch(color='yellow', label=r"class 1")
        red_patch = mpatches.Patch(color='red', label=r"class 2")
        fig.legend(handles=[blue_patch, cyan_patch, yellow_patch, red_patch],
                   loc='center')
        
    fig.suptitle(_title)
    plt.show()


from sklearn.model_selection import train_test_split
X = df[["gMi", "gMr", "rMi"]]
Y = df[["redshift"]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state=99)
make_base_plot(data_df=X, Y=Y,
               _title="Plots des différences entre bandes photométriques pour les lentilles gravitationnelles en fonction du redshift")


# ---- Régression linéaire
import statsmodels.api as sm

linear_model = sm.OLS(Y, X)
results = linear_model.fit()
results.summary()
print(results.summary().as_latex())
y_pred_linear = results.predict(X_test)
y_pred_linear

make_base_plot(data_df=X_test, Y=y_pred_linear,
               _title="Plots des différences entre bandes photométriques pour les lentilles gravitationnelles en fonction du redshift, \ncolorée selon les prévisions via la régression linéaire sur l'chantillon de test")

# ---- Gaussian Model
from sklearn.mixture import GaussianMixture

aic_values = []
bic_values = []
for i in range(2, 50):
    gmm = GaussianMixture(n_components=i, covariance_type='full')
    gmm.fit(X)
    aic_values.append(gmm.aic(X))
    bic_values.append(gmm.bic(X))
plt.plot(range(2, 50), aic_values, label = 'aic')
plt.plot(range(2, 50), bic_values, label = 'bic')
plt.legend()
plt.title("Ajustement du modèle GMM sur la population de lentilles")
plt.xlabel("Nombre de composants")
plt.ylabel("Valeur du critère")
plt.show()
aic_values.index(min(aic_values))+2
bic_values.index(min(bic_values))+2
# on choisit donc 3 components

gmm = GaussianMixture(n_components=4, covariance_type='full')
gmm.fit(X)
y_pred = gmm.predict(X)

make_base_plot(_color=y_pred, _title="Plots des différences entre bandes photométriques pour les lentilles gravitationnelles en fonction du redshift, coloré selon le modèle GMM")


df["predict_GMM"] = y_pred
means = df.groupby('predict_GMM')['redshift'].mean()
# df["z_GMM"] = 0
# df.loc[df.predict_GMM == 0, 'z_GMM'] = means[0]
# df.loc[df.predict_GMM == 1, 'z_GMM'] = means[1]
# df.loc[df.predict_GMM == 2, 'z_GMM'] = means[2]
# df.loc[df.predict_GMM == 3, 'z_GMM'] = means[3]

