# Chargement des packages et classes 

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from python.plotfunctions import *
from scipy.stats import norm
import scipy.stats as scs

#--- load classes
from python.objets.wobjet import WDegCarre
from python.objets.lentille import LentilleGravita

# Lien pour charger les données pickle
lentilles_list = pickle.load(open("./source_pickle/lentilles_list2", "rb"))
obj_list = pickle.load(open("./source_pickle/fields", "rb"))


# Useful functions
def centrer_reduit(array):
    return (array-np.mean(array))/np.std(array)

# Création des listes pour afficher le seeing
# -- listes de données seeing
seeing_list_lentille = np.array([lentille.seeing for lentille in lentilles_list])
seeing_list_obj = np.array([obj.seeing for obj in obj_list])

seeing_list_lentille_noNA = seeing_list_lentille[~np.isnan(seeing_list_lentille)]
seeing_list_obj_noNA = seeing_list_obj[~np.isnan(seeing_list_obj)]


seeing_list_lentille_01 = centrer_reduit(seeing_list_lentille_noNA)
seeing_list_obj_01 = centrer_reduit(seeing_list_obj_noNA)

# --- 1er histogrammes de seeing
bins = np.linspace(0, 1, 30)

plt.hist(seeing_list_lentille_noNA, bins, alpha=0.5, density=True,
         label=f"Répartition empirique des lentilles (n = {len(seeing_list_lentille_noNA)})")
plt.hist(seeing_list_obj_noNA, bins, alpha=0.5, density=True,
         label=f"Répartition théorique des champs (n = {len(seeing_list_obj_noNA)})")
plt.legend(loc='upper left')
plt.title(
    "Densité de répartition des lentilles en fonction du seeing sur les champs W1-4")
plt.xlabel("Seeing (arc sec)")
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
    title="QQ-plot des répartition empiriques et théoriques des lentilles selon le seeing")
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
plt.plot(x, norm.pdf(x, mu_l, std_l), 'blue', linewidth=2, label = f"Densité gausienne lentilles, $\mu$ = {round(mu_l,2)}, $\sigma$ = {round(std_l,3)}")
plt.plot(x, norm.pdf(x, mu_w, std_w), 'orange', linewidth=2,
         label=f"Densité gausienne fields, $\mu$ = {round(mu_w,2)}, $\sigma$ = {round(std_w,3)}")

plt.legend(loc='upper left')
plt.title(
    "Densité de répartition des lentilles en fonction du seeing sur les champs W1-4")
plt.xlabel("Seeing (arc sec)")
plt.ylabel("Densité")
plt.show()

# T test de Welch
_, p_value = scs.ttest_ind(seeing_list_lentille_01,
                           seeing_list_obj_01, equal_var=False)
p_value 
# on accepte H_0


# ---- Exposition

expo_list_lentille = np.array([lentille.exposition for lentille in lentilles_list])
expo_list_obj = np.array([obj.exposition for obj in obj_list])

expo_list_lentille_noNA = expo_list_lentille[~np.isnan(
    expo_list_lentille)]
expo_list_obj_noNA = expo_list_obj[~np.isnan(expo_list_obj)]

expo_list_lentille_01 = centrer_reduit(expo_list_lentille_noNA)
expo_list_obj_01 = centrer_reduit(expo_list_obj_noNA)

# --- 1er histogrammes d'exposition
bins = np.linspace(40, 160, 8)

plt.hist(expo_list_lentille_noNA, bins, alpha=0.5, density=True,
         label=f"Répartition empirique des lentilles (n = {len(expo_list_lentille_noNA)})")
plt.hist(expo_list_obj_noNA, bins, alpha=0.5, density=True,
         label=f"Répartition théorique des champs (n = {len(expo_list_obj_noNA)})")
plt.legend(loc='upper right')
plt.title(
    "Densité de répartition des lentilles en fonction de l'exposition sur les champs W1-4")
plt.xlabel("Exposition (min)")
plt.ylabel("Densité")
plt.gca().invert_xaxis()
plt.show()

# --- étude de la normalité des 2 échantillons

#qqplots
fig, (axe0, axe1) = plt.subplots(1, 2)
make_global_title(
    fig,
    title="QQ-plot des répartition empiriques et théoriques des lentilles selon l'exposition")
sm.qqplot(expo_list_lentille_01, line='45', ax=axe0, )
sm.qqplot(expo_list_obj_01, line='45', ax=axe1)
axe0.title.set_text(f"Lentilles (n = {len(expo_list_lentille_01)})")
axe1.title.set_text(f"Champs (n = {len(expo_list_obj_01)})")

plt.show()

# -- Wilcoxon non paramétrique
N = 10_000
p_values = np.zeros(N)
for i in range(N):
    np.random.shuffle(expo_list_obj_noNA)
    expo_list_obj_noNA_len = expo_list_obj_noNA[0:len(expo_list_lentille_noNA)]

    _, p_values[i] = scs.wilcoxon(x=expo_list_lentille_noNA, y=expo_list_obj_noNA_len, alternative="two-sided")
np.mean(p_values)





# -- Rappel de la distribution spatiale des lentilles
raL, decL = [lent.ra for lent in lentilles_list], [lent.dec for lent in lentilles_list]

fig, axes = plt.subplots(2, 2)
plot_sub_data(axes, [0, 0], raL, decL, (30, 40), (-14, -2), "W1")
plot_sub_data(axes, [0, 1], raL, decL, (131, 138), (-6, 0), "W2")
plot_sub_data(axes, [1, 0], raL, decL, (205, 225), (50, 60), "W3")
plot_sub_data(axes, [1, 1], raL, decL, (328, 338), (-2, 5), "W4")

make_global_title(
    fig,
    title="Portion de champs et dispositions des lentilles gravitationelles",
    x_title="Ascension droite (deg)",
    y_title="Déclinaison (deg)")

plt.show()

# -- Distribution spatiale des lentilles en fonction du redshift
raL_m, decL_m = [lent.ra for lent in lentilles_list if lent.z < 0.5], [lent.dec for lent in lentilles_list if lent.z < 0.5]
raL_p, decL_p = [lent.ra for lent in lentilles_list if lent.z >= 0.5], [lent.dec for lent in lentilles_list if lent.z >= 0.5]


fig, axes = plt.subplots(2, 2)
plot_sub_data(axes, [0, 0], raL_p, decL_p, (30, 40), (-14, -2), "W1")
plot_sub_data(axes, [0, 0], raL_m, decL_m, (30, 40),
              (-14, -2), "W1",_marker='o', color='blue')

plot_sub_data(axes, [0, 1], raL_p, decL_p, (131, 138), (-6, 0), "W2")
plot_sub_data(axes, [0, 1], raL_m, decL_m, (131, 138),
              (-6, 0), "W2", _marker='o', color='blue')   

plot_sub_data(axes, [1, 0], raL_p, decL_p, (205, 225), (50, 60), "W3")
plot_sub_data(axes, [1, 0], raL_m, decL_m, (205, 225),
              (50, 60), "W3", _marker='o', color='blue')

plot_sub_data(axes, [1, 1], raL_p, decL_p, (328, 338), (-2, 5), "W4", _label = 'redshift$\geq$0.5')
plot_sub_data(axes, [1, 1], raL_m, decL_m, (328, 338),
              (-2, 5), "W4", _marker='o', color='blue', _label = 'redshift < 0.5')
fig.legend()

make_global_title(
    fig,
    title="Portion de champs et dispositions des lentilles gravitationelles en fonction du redshift",
    x_title="Ascension droite (deg)",
    y_title="Déclinaison (deg)")

plt.show()


# -- Histograme de seeing en fonction du redshift

see_m, see_p = [lent.seeing for lent in lentilles_list if lent.z < 0.5], [lent.seeing for lent in lentilles_list if lent.z >= 0.5]

ax1 = plt.subplot(221)
ax1.hist(see_m, density=True)
ax1.set_title(f"z<0.5 (n={len(see_m)})")
ax1.set_xlabel("Seeing (arc sec)")
ax1.set_yabel("Densité")


ax2 = plt.subplot(222)
ax2.hist(see_p, density=True)
ax2.set_title(f"z$\geq$0.5 (n={len(see_p)})")
ax2.set_xlabel("Seeing (arc sec)")
ax2.set_yabel("Densité")


ax3 = plt.subplot(212)
ax3.hist(see_m, alpha=0.5, density=True, label=f"z<0.5 (n={len(see_m)})")
ax3.hist(see_p, alpha=0.5, density=True, label=f"z$\geq$0.5 (n={len(see_p)})")
ax3.set_xlabel("Seeing (arc sec)")
ax3.set_yabel("Densité")
ax3.legend()
plt.suptitle("Histogrammes de densité de seeing des lentilles en fonction du redshift")

plt.show()

# -- Histograme de l'exposition en fonction du redshift

expo_m, expo_p = [lent.exposition for lent in lentilles_list if lent.z < 0.5], [lent.exposition for lent in lentilles_list if lent.z >= 0.5]

ax1 = plt.subplot(221)
ax1.hist(expo_m, density=True)
ax1.set_title(f"z<0.5 (n={len(expo_m)})")
ax1.set_xlabel("Exposition (min)")
ax1.set_yabel("Densité")


ax2 = plt.subplot(222)
ax2.hist(expo_p, density=True)
ax2.set_title(f"z$\geq$0.5 (n={len(expo_p)})")
ax2.set_xlabel("Exposition (min)")
ax2.set_yabel("Densité")


ax3 = plt.subplot(212)
ax3.hist(expo_m, alpha=0.5, density=True, label=f"z<0.5 (n={len(expo_m)})")
ax3.hist(expo_p, alpha=0.5, density=True, label=f"z$\geq$0.5 (n={len(expo_p)})")
ax3.set_xlabel("Exposition (min)")
ax3.set_yabel("Densité")
ax3.legend()
plt.suptitle("Histogrammes de densité d'exposition des lentilles en fonction du redshift")

plt.show()






# -- Distribution spatiale des lentilles en fonction du rayon d'einstein
R = np.array([l.rad for l in lentilles_list])
# np.nanmedian(R)

raL_m, decL_m = [lent.ra for lent in lentilles_list if lent.rad < np.nanmedian(R)], [lent.dec for lent in lentilles_list if lent.rad < np.nanmedian(R)]
raL_p, decL_p = [lent.ra for lent in lentilles_list if lent.rad >= np.nanmedian(R)], [lent.dec for lent in lentilles_list if lent.rad >= np.nanmedian(R)]


fig, axes = plt.subplots(2, 2)
plot_sub_data(axes, [0, 0], raL_m, decL_m, (30, 40), (-14, -2), "W1")
plot_sub_data(axes, [0, 0], raL_p, decL_p, (30, 40),
              (-14, -2), "W1",_marker='o', color='blue')

plot_sub_data(axes, [0, 1], raL_m, decL_m, (131, 138), (-6, 0), "W2")
plot_sub_data(axes, [0, 1], raL_p, decL_p, (131, 138),
              (-6, 0), "W2", _marker='o', color='blue')   

plot_sub_data(axes, [1, 0], raL_m, decL_m, (205, 225), (50, 60), "W3")
plot_sub_data(axes, [1, 0], raL_p, decL_p, (205, 225),
              (50, 60), "W3", _marker='o', color='blue')

plot_sub_data(axes, [1, 1], raL_m, decL_m, (328, 338), (-2, 5), "W4", _label=f"$R_E$< {np.nanmedian(R)}")
plot_sub_data(axes, [1, 1], raL_p, decL_p, (328, 338),
              (-2, 5), "W4", _marker='o', color='blue', _label = f"$R_E\geq$ {np.nanmedian(R)}")
fig.legend()

make_global_title(
    fig,
    title="Portion de champs et dispositions des lentilles gravitationelles en fonction du rayon d'Einstein",
    x_title="Ascension droite (deg)",
    y_title="Déclinaison (deg)")

plt.show()


# -- Histograme de seeing en fonction du rayon d'Einstein

see_m, see_p = [lent.seeing for lent in lentilles_list if lent.rad < np.nanmedian(R)], [lent.seeing for lent in lentilles_list if lent.rad >= np.nanmedian(R)]

ax1 = plt.subplot(221)
ax1.hist(see_m, density=True)
ax1.set_title(f"$R_E$<{np.nanmedian(R)} (n={len(see_m)})")
ax1.set_xlabel("Seeing (arc sec)")
ax1.set_yabel("Densité")


ax2 = plt.subplot(222)
ax2.hist(see_p, density=True)
ax2.set_title(f"$R_E\geq$ {np.nanmedian(R)} (n={len(see_p)})")
ax2.set_xlabel("Seeing (arc sec)")
ax2.set_yabel("Densité")


ax3 = plt.subplot(212)
ax3.hist(see_m, alpha=0.5, density=True, label=f"$R_E$<{np.nanmedian(R)} (n={len(see_m)})")
ax3.hist(see_p, alpha=0.5, density=True, label=f"$R_E\geq$ {np.nanmedian(R)} (n={len(see_p)})")
ax3.set_xlabel("Seeing (arc sec)")
ax3.set_yabel("Densité")
ax3.legend()
plt.suptitle("Histogrammes de densité de seeing des lentilles en fonction du rayon d'Einstein")

plt.show()

# -- Histograme de l'exposition en fonction du rayon d'Einstein

expo_m, expo_p = [lent.exposition for lent in lentilles_list if lent.rad < np.nanmedian(R)], [lent.exposition for lent in lentilles_list if lent.rad >= np.nanmedian(R)]

ax1 = plt.subplot(221)
ax1.hist(expo_m, density=True)
ax1.set_title(f"$R_E$<{np.nanmedian(R)} (n={len(expo_m)})")
ax1.set_xlabel("Exposition (min)")
ax1.set_yabel("Densité")


ax2 = plt.subplot(222)
ax2.hist(expo_p, density=True)
ax2.set_title(f"$R_E\geq$ {np.nanmedian(R)}(n={len(expo_p)})")
ax2.set_xlabel("Exposition (min)")
ax2.set_yabel("Densité")


ax3 = plt.subplot(212)
ax3.hist(expo_m, alpha=0.5, density=True,
         label=f"$R_E$<{np.nanmedian(R)} (n={len(expo_m)})")
ax3.hist(expo_p, alpha=0.5, density=True,
         label=f"$R_E\geq$ {np.nanmedian(R)}(n={len(expo_p)})")
ax3.set_xlabel("Exposition (min)")
ax3.set_yabel("Densité")
ax3.legend()
plt.suptitle("Histogrammes de densité d'exposition des lentilles en fonction du rayon d'Einstein")

plt.show()



# -- Etude selon le rayon d'Einstein
bins = np.linspace(0, 20, 50)
z_ech_w1 = [
    lentille.rad for lentille in lentilles_list if lentille.w in ("W1", "D1")]
z_ech_w2 = [lentille.rad for lentille in lentilles_list if lentille.w == "W2"]
z_ech_w3 = [
    lentille.rad for lentille in lentilles_list if lentille.w in ("W3", "D3")]
z_ech_w4 = [lentille.rad for lentille in lentilles_list if lentille.w == "W4"]

fig, axes = plt.subplots(2, 2)
hist_sub_data(axes, [0, 0], z_ech_w1, "W1", _bins=bins)
hist_sub_data(axes, [0, 1], z_ech_w2, "W2", _bins=bins)
hist_sub_data(axes, [1, 0], z_ech_w3, "W3", _bins=bins)
hist_sub_data(axes, [1, 1], z_ech_w4, "W4", _bins=bins)

make_global_title(fig, "Histogramme des rayon d'Einstein des lentilles")
plt.show()

# -- Etude selon le redshift
bins = np.linspace(0, 2, 20)
z_ech_w1 = [lentille.z for lentille in lentilles_list if lentille.w in ("W1", "D1")]
z_ech_w2 = [lentille.z for lentille in lentilles_list if lentille.w == "W2"]
z_ech_w3 = [lentille.z for lentille in lentilles_list if lentille.w in ("W3", "D3")]
z_ech_w4 = [lentille.z for lentille in lentilles_list if lentille.w == "W4"]

fig, axes = plt.subplots(2, 2)
hist_sub_data(axes, [0, 0], z_ech_w1, "W1", _bins = bins)
hist_sub_data(axes, [0, 1], z_ech_w2, "W2", _bins=bins)
hist_sub_data(axes, [1, 0], z_ech_w3, "W3", _bins=bins)
hist_sub_data(axes, [1, 1], z_ech_w4, "W4", _bins=bins)

make_global_title(fig, "Histogramme des redshift des lentilles")
plt.show()


ra, dec, z, rac = [lentille.ra for lentille in lentilles_list], [lentille.dec for lentille in lentilles_list], [lentille.z for lentille in lentilles_list], [lentille.rad for lentille in lentilles_list]


fig, axes = plt.subplots(2, 2)

# rayon einstein
plot_sub_data(axes, [0, 0], ra, dec, (30, 40), (-14, -2), "W1", color=rac, _marker = 'o')
plot_sub_data(axes, [0, 1], ra, dec, (131, 138), (-6, 0), "W2", color=rac, _marker = 'o')
plot_sub_data(axes, [1, 0], ra, dec, (205, 225), (50, 60), "W3", color=rac, _marker = 'o')
sc4 = axes[1, 1].scatter(ra, dec, marker='o', c=rac)
plt.colorbar(sc4)
axes[1, 1].set_xlim((328, 338))
axes[1, 1].set_ylim((-2, 5))
axes[1, 1].set_title("W4")

# redshift
plot_sub_data(axes, [0, 0], ra, dec, (30, 40), (-14, -2), "W1", color = z)
plot_sub_data(axes, [0, 1], ra, dec, (131, 138), (-6, 0), "W2", color=z)
sc3 = axes[1, 0].scatter(ra, dec, marker='+', c=z)
axes[1, 0].set_xlim((205, 225))
axes[1, 0].set_ylim((50, 60))
axes[1, 0].set_title("W3")
plt.colorbar(sc3)
plot_sub_data(axes, [1, 1], ra, dec, (328, 338), (-2, 5), "W4", color=z)


make_global_title(fig, "Disposition des lentilles et redshift et rayon d'einstein")

# plt.scatter(x=[lentille.ra for lentille in lentilles_list], y=[lentille.dec for lentille in lentilles_list], marker='+', c=[lentille.z for lentille in lentilles_list])
# plt.show()

# fig, axes = plt.subplots(2, 2)


plt.show()





# ---------------------------
"""lentilles_list_1 = [lentille for lentille in lentilles_list if lentille.w == 1]

for i in range(len(lentilles_list_1)):
    curr_lent = lentilles_list_1[i]
    curr_lent_ppv = curr_lent.closest_object(
        lentilles_list_1[0:i]+lentilles_list_1[i+1:])
    curr_lent.set_ppv(curr_lent_ppv)
"""

"""
dit_list = [lentille.distance(lentille.ppv) for lentille in lentilles_list]
bins = np.linspace(0, 4, 40)

plt.hist(dit_list, bins, density = True)
plt.show()
"""


"""
C = [[o.ra, o.dec,] for o in obj_list]
X = [o.ra for o in obj_list]
Y = [o.dec for o in obj_list]
Z = np.array([o.seeing for o in obj_list])
plt.pcolormesh(X=X, Y=Y, cmap=Z, shading='nearest', vmin=Z.min(), vmax=Z.max())
plt.show()
"""
