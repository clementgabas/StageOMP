import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from python.plotfunctions import *
from scipy.stats import norm
import scipy.stats as scs




lentilles_list = pickle.load(open("./source_pickle/lentilles_list", "rb"))
obj_list = pickle.load(open("./source_pickle/fields", "rb"))

# -- listes de données seeing
seeing_list_lentille = np.array([lentille.seeing for lentille in lentilles_list])
seeing_list_obj = np.array([obj.seeing for obj in obj_list])

seeing_list_lentille_noNA = seeing_list_lentille[~np.isnan(seeing_list_lentille)]
seeing_list_obj_noNA = seeing_list_obj[~np.isnan(seeing_list_obj)]


def centrer_reduit(array):
    return (array-np.mean(array))/np.std(array)

seeing_list_lentille_01 = centrer_reduit(seeing_list_lentille_noNA)
seeing_list_obj_01 = centrer_reduit(seeing_list_obj_noNA)

# --- 1er histogrammes de seeing
bins = np.linspace(0, 1, 30)

plt.hist(seeing_list_lentille_noNA, bins, alpha=0.5, density=True,
         label=f"Répartition empirique des lentilles (n = {len(seeing_list_lentille_noNA)})")
plt.hist(seeing_list_obj_noNA, bins, alpha=0.5, density=True,
         label=f"Répartition théorique des champs (n = {len(seeing_list_obj_noNA)})")
plt.legend(loc='upper right')
plt.title(
    "Densité de répartition des lentilles en fonction du seeing sur les champs W1-4")
plt.xlabel("Seeing")
plt.ylabel("Densité")
plt.show()

# --- 1er histogrammes de seeing avec tracé de densité
bw_dict = {'bw': 0.3}
sns.distplot(seeing_list_lentille_noNA, kde=True, bins=bins, rug=True, kde_kws=bw_dict,
             label=f"empirique (n = {len(seeing_list_lentille_noNA)})")
sns.distplot(seeing_list_obj_noNA, kde=True, bins=bins, rug=True, kde_kws=bw_dict,
             label=f"théorique (n = {len(seeing_list_obj_noNA)})")
plt.legend(loc='upper right')
plt.title(
    "Densité de répartition des lentilles en fonction du seeing sur les champs W1-4")
plt.suptitle(
    f"Non parametric Parzen-Rosenblatt estimator - estimation par noyau non paramétrique - bw = {bw_dict['bw']}")
plt.xlabel("Seeing")
plt.ylabel("Densité")
plt.show()

# --- étude de la normalité des 2 échantillons

#qqplots
fig, (axe0, axe1) = plt.subplots(1, 2)
make_global_title(
    fig,
    title="QQ-plot des répartition empiriques et thoériques des lentilles selon le seeing")
sm.qqplot(seeing_list_lentille_01, line='45', ax=axe0, )
sm.qqplot(seeing_list_obj_01, line='45', ax=axe1)
axe0.title.set_text(f"Lentilles (n = {len(seeing_list_lentille_01)})")
axe1.title.set_text(f"Champs (n = {len(seeing_list_obj_01)})")

plt.show()

#theorical normal
mu_l, std_l = norm.fit(seeing_list_lentille_noNA)
mu_w, std_w = norm.fit(seeing_list_obj_noNA)

plt.hist(seeing_list_lentille_noNA, bins, alpha=0.5, density=True,
         label=f"Répartition empirique des lentilles (n = {len(seeing_list_lentille_noNA)})", color = 'blue')
plt.hist(seeing_list_obj_noNA, bins, alpha=0.5, density=True,
         label=f"Répartition théorique des champs (n = {len(seeing_list_obj_noNA)})", color = 'orange')
x = np.linspace(0, 1, 100)
plt.plot(x, norm.pdf(x, mu_l, std_l), 'blue', linewidth=2, label = f"Densité gausienne lentilles, mu = {round(mu_l,2)}, std = {round(std_l,3)}")
plt.plot(x, norm.pdf(x, mu_w, std_w), 'orange', linewidth=2, label = f"Densité gausienne fields, mu = {round(mu_w,2)}, std = {round(std_w,3)}")

plt.legend(loc='upper left')
plt.title(
    "Densité de répartition des lentilles en fonction du seeing sur les champs W1-4")
plt.xlabel("Seeing")
plt.ylabel("Densité")
plt.show()

# T test de Welch
_, p_value = scs.ttest_ind(seeing_list_lentille_01,
                           seeing_list_obj_01, equal_var=False)
p_value 
# on accepte H_0
