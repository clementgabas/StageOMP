import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns
from astropy.io import fits
from astropy.table import Table
import pandas as pd
import math
import random as rd

file_lentilles = "./source/Lentilles.fits"
with fits.open(file_lentilles, memmap=True) as hdulist_Lentilles:
    hdulist_Lentilles.info()
    lentilles_df = pd.DataFrame(hdulist_Lentilles[1].data)

file_lentilles_2 = "./source/Lentilles.csv"
lentilles_df_2 = pd.read_csv(
    file_lentilles_2, sep=";", header=[0], skiprows=[1])

lentilles_W1_df = lentilles_df_2[lentilles_df_2['Fld'] == "W1"]
lentilles_D1_df = lentilles_df_2[lentilles_df_2['Fld'] == "D1"]
lentilles_WD1_df = pd.concat([lentilles_W1_df, lentilles_D1_df], )
lentilles_W2_df = lentilles_df_2[lentilles_df_2['Fld'] == "W2"]
lentilles_W3_df = lentilles_df_2[lentilles_df_2['Fld'] == "W3"]
lentilles_D3_df = lentilles_df_2[lentilles_df_2['Fld'] == "D3"]
lentilles_WD3_df = pd.concat([lentilles_W3_df, lentilles_D3_df], )
lentilles_W4_df = lentilles_df_2[lentilles_df_2['Fld'] == "W4"]


def tirage_lenses(n_lenses, loc_coord):
    xs = [loc[0] for loc in loc_coord]
    ys = [loc[1] for loc in loc_coord]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    out_list = list()

    for i in range(n_lenses):
        x = rd.uniform(x_min, x_max)
        y = rd.uniform(y_min, y_max)
        out_list.append([x, y])
    return out_list


def make_sub(x_min, x_max, n_sub):
    sub_x = [x_min]
    eps_x = (x_max - x_min) / (n_sub)
    for i in range(n_sub - 1):
        curr_x = sub_x[-1]
        sub_x.append(round(curr_x + eps_x, 4))
    sub_x.append(x_max)

    return sub_x


def make_subdivision(loc_coord, n_sub_col, n_sub_row):
    xs = [loc[0] for loc in loc_coord]
    ys = [loc[1] for loc in loc_coord]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    sub_x = make_sub(x_min, x_max, n_sub_col)
    sub_y = make_sub(y_min, y_max, n_sub_row)
    list_of_loc = list()
    for i in range(n_sub_col * n_sub_row):
        applic_x0, applic_x1 = i % n_sub_col, (i % n_sub_col) + 1
        applic_y0, applic_y1 = i // n_sub_col, (i // n_sub_col) + 1
        x0, x1 = sub_x[applic_x0], sub_x[applic_x1]
        y0, y1 = sub_y[applic_y0], sub_y[applic_y1]
        curr_list = [[x0, y0], [x0, y1], [x1, y1], [x1, y0]]
        list_of_loc.append(curr_list)
    return list_of_loc


# make_subdivision(W1_loc, 10, 10)


def find_the_cell(lense_coord, list_of_cells):
    # Les cells sont numérotées de 0 à n_col*n_row-1 en partant d'en bas à gauche et en repllisant d'abord les lignes
    # puis les colonnes
    # ------- |3|4|5| |0|1|2| -------
    for cell in list_of_cells:
        x_min, x_max = cell[0][0], cell[3][0]
        y_min, y_max = cell[0][1], cell[1][1]
        lense_x, lense_y = lense_coord[0], lense_coord[1]
        if (x_min <= lense_x <= x_max) and (y_min <= lense_y <= y_max):
            return cell


# find_the_cell([32, -9], [W1_loc])

def make_dict(subdivisions, lenses_list):
    results_dict = dict()
    for cell in subdivisions:
        results_dict[str(cell)] = 0
    for lense in lenses_list:
        cell = find_the_cell(lense_coord=lense, list_of_cells=subdivisions)
        results_dict[str(cell)] += 1
    return results_dict

def make_simulation(coord_champ, n_col, n_row, n_lenses, subdivisions):
    lenses_list = tirage_lenses(n_lenses=n_lenses, loc_coord=coord_champ)
    return make_dict(subdivisions=subdivisions, lenses_list=lenses_list)


def make_multiple_simulation(n_simul, coord_champ, n_col, n_row, n_lenses):
    subdivisions = make_subdivision(loc_coord=coord_champ, n_sub_col=n_col, n_sub_row=n_row)

    result_dict_global = make_simulation(coord_champ=coord_champ, n_col=n_col, n_row=n_row, n_lenses=n_lenses,
                                         subdivisions=subdivisions)
    result_dict_global_for_std = result_dict_global.copy()
    for key, value in result_dict_global_for_std.items():
        result_dict_global_for_std[key] = [value]


    for i in range(n_simul - 1):
        result_dict = make_simulation(coord_champ=coord_champ, n_col=n_col, n_row=n_row, n_lenses=n_lenses,
                                      subdivisions=subdivisions)
        for key, value in result_dict.items():
            result_dict_global[key] += value
            result_dict_global_for_std[key].append(value)

    std_dict={}
    for key, value in result_dict_global_for_std.items():
        std_dict[key] = np.std(value)
        
    mean_dict = {k: v / n_simul for k, v in result_dict_global.items()}
    
    return (mean_dict, std_dict)


def make_substraction(simul_d, real_d):
    for key, value in simul_d.items():
        real_d[key] -= simul_d[key]
    return real_d

def make_division(std_d, real_d):
    out_d = {}
    for key, value in std_d.items():
        out_d[key] = real_d[key]/std_d[key]
    return out_d

def make_draw_values(value, dist_method):
    if dist_method=='absolue':
        limit_valueP, limit_valueM = 1.1, -1.1
    elif dist_method == 'std':
        limit_valueP, limit_valueM = 2, 0.5
    _color, _l, _face = 'royalblue', 1, "None"
    if value >= limit_valueP:
        _color, _l, _face = 'red', 2, 'red'
    elif value <= limit_valueM:
        _color, _l, _face = 'green', 2, 'green'
    return (_color, _l, _face)
    

def plot_count_in_cell_dict(count_in_cell_dict, n_lenses, lense_coord='None', title='', D_field_coord=[], dist_method="absolue", _legend=True):
    fig, ax = plt.subplots()

    rectangles = []
    count=0
    palier = n_lenses/len(count_in_cell_dict)

    for key, value in count_in_cell_dict.items():
        count+=1
        rectangle = eval(key)
        x0, y0 = rectangle[0]
        x1, y1 = rectangle[2]
        width = x1-x0
        height = y1-y0
        _color, _l, _face = make_draw_values(value, dist_method=dist_method)
        curr_Rectangle = patches.Rectangle(xy=(x0, y0), width=width, height=height, linewidth=_l, edgecolor=_color, facecolor=_face, alpha=0.2)
        rectangles.append([value, curr_Rectangle])
        if count == 1:
            x_min, y_min = x0, y0
        elif count == len(count_in_cell_dict):
            x_max, y_max = x1, y1
            
    W_Rectangle = patches.Rectangle(xy=(x_min, y_min), width=x_max-x_min, height=y_max-y_min, linewidth=1, edgecolor="black", facecolor="None")
    rectangles.append(["", W_Rectangle])
    if D_field_coord:
        Dx_min, Dy_min = D_field_coord[0]
        Dx_max, Dy_max = D_field_coord[2]
        D_Rectangle = patches.Rectangle(xy=(Dx_min, Dy_min), width=Dx_max-Dx_min, height=Dy_max-Dy_min, linewidth=2, edgecolor="grey", facecolor="None", linestyle="--")
        rectangles.append(["", D_Rectangle])
    
    
    for r in rectangles:
        ax.add_artist(r[1])
        if isinstance(r[0], (int, float)):
            rx, ry = r[1].get_xy()
            cx = rx + r[1].get_width()/2.0
            cy = ry + r[1].get_height()/2.0
            ax.annotate(round(r[0], 3), (cx, cy), color='black', weight='bold', fontsize=6, ha='center', va='center')
    ax.set_xlim((x_min-.5, x_max+.5))
    ax.set_ylim((y_min-.5, y_max+.5))
    ax.invert_xaxis()
    
    if lense_coord == 'None':
        sca = "+"
    else:
        sca = ax.scatter(x=lense_coord[0], y=lense_coord[1], marker='+', c='black', label="lentilles")
    
    plt.suptitle(title, fontsize=15)
    plt.title(f"Densité attendue d'objet par subdivision : {round(palier, 3)}")
    plt.xlabel("RA")
    plt.ylabel("DEC")
    
    red_patch = patches.Patch(color='red', label='Surdensité locale')
    green_patch = patches.Patch(color='green', label='Sousdensité locale')
    blue_patch = patches.Patch(edgecolor='blue', facecolor='None', label='Densité locale normale')
    black_patch = patches.Patch(edgecolor='black', facecolor='None', label='Contours du champ W')
    grey_patch = patches.Patch(
        edgecolor='grey', facecolor='None', label='Contours du champ D', linestyle="--")
    
    if _legend:
        if not D_field_coord:
            fig.legend(handles=[red_patch, green_patch, blue_patch, sca, black_patch])
        else:
            fig.legend(handles=[red_patch, green_patch, blue_patch, sca, black_patch, grey_patch])
        
    ax.set_aspect('equal')
    plt.show()
    
    
def plot_simulations(n_simul=20_000, coord_champ=W1_loc, n_col=6, n_row=10, lentilles_df=lentilles_WD1_df, title='', coord_W=[], dist_method="absolue", _legend=True):
    n_lenses = len(lentilles_df)
    print("making simulations...")
    mean_dict, std_dict = make_multiple_simulation(n_simul=n_simul, coord_champ=coord_champ, n_col=n_col, n_row=n_row, n_lenses=n_lenses)

    sub = make_subdivision(loc_coord=coord_champ, n_sub_col=n_col, n_sub_row=n_row)
    c1 = [[row["_RAJ2000"], row["_DEJ2000"]] for index, row in lentilles_df.iterrows()]
    c1_x, c1_y = [row["_RAJ2000"] for index, row in lentilles_df.iterrows()], [row["_DEJ2000"] for index, row in lentilles_df.iterrows()]
    d1 = make_dict(subdivisions=sub, lenses_list=c1)

    
    if dist_method.lower() == "absolue":
        o1 = make_substraction(mean_dict, d1)
        plot_count_in_cell_dict(count_in_cell_dict=o1, n_lenses=n_lenses, dist_method="absolue",
                                lense_coord=[c1_x, c1_y], title=title, D_field_coord=coord_W, _legend=_legend)
    elif dist_method.lower() == "std":
        o1 = make_division(mean_dict, d1)
        plot_count_in_cell_dict(count_in_cell_dict=o1, n_lenses=n_lenses, dist_method="std",
                                lense_coord=[c1_x, c1_y], title=title, D_field_coord=coord_W, _legend=_legend)


# Champs W1
# on enregistr eles points dans cet ordre : [coin gauche bas, coin gauche haut, coin droite haut, coin droite bas]
W1_loc = [[30, -11.5], [30, -3.5], [39, -3.5], [39, -11.5]]
D1_loc = [[35.9958, -4.9944], [35.9958, -3.9944], [36.9958, -3.9944], [36.9958, -4.9944]]
plot_simulations(n_simul=20_000, coord_champ=W1_loc, n_col=6, n_row=10, lentilles_df=lentilles_WD1_df, coord_W=D1_loc,
                 title="Surdensité des lentilles dans les champs W1 & D1 \n20 000 simulations selon une loi uniforme $\mathcal{U}_{W1}$")


# Champs W2
# on enregistr eles points dans cet ordre : [coin gauche bas, coin gauche haut, coin droite haut, coin droite bas]
W2_loc = [[131.5, -4.25], [131.5, 0.75], [136.5, 0.75], [136.5, -4.25]]
plot_simulations(n_simul=20_000, coord_champ=W2_loc, n_col=5, n_row=5, lentilles_df=lentilles_W2_df,
                 title="Surdensité des lentilles dans le champ W2 \n20 000 simulations selon une loi uniforme $\mathcal{U}_{W2}$")

# Champs W3
# on enregistr eles points dans cet ordre : [coin gauche bas, coin gauche haut, coin droite haut, coin droite bas]
W3_loc = [[209, 51], [219, 58], [219, 58], [209, 51]]
D3_loc = [[214.25, 52.25], [214.25, 53.25], [215.25, 53.25], [215.25, 52.25]]

plot_simulations(n_simul=20_000, coord_champ=W3_loc, n_col=7, n_row=7, lentilles_df=lentilles_W3_df, coord_W=D3_loc,
                 title="Surdensité des lentilles dans le champ W3 \n20 000 simulations selon une loi uniforme $\mathcal{U}_{W3}$")


# Moins de finesse
plot_simulations(n_simul=20_000, coord_champ=W1_loc, n_col=5, n_row=5, lentilles_df=lentilles_WD1_df, coord_W=D1_loc,
                 title="Surdensité des lentilles dans les champs W1 & D1 \n20 000 simulations selon une loi uniforme $\mathcal{U}_{W1}$")
plot_simulations(n_simul=20_000, coord_champ=W2_loc, n_col=3, n_row=3, lentilles_df=lentilles_W2_df,
                 title="Surdensité des lentilles dans le champ W2 \n20 000 simulations selon une loi uniforme $\mathcal{U}_{W2}$")
plot_simulations(n_simul=20_000, coord_champ=W3_loc, n_col=5, n_row=5, lentilles_df=lentilles_W3_df, coord_W=D3_loc,
                 title="Surdensité des lentilles dans le champ W3 et D3 \n20 000 simulations selon une loi uniforme $\mathcal{U}_{W3}$")

# Plus de finesse
plot_simulations(n_simul=20_000, coord_champ=W1_loc, n_col=15, n_row=15, lentilles_df=lentilles_WD1_df, coord_W=D1_loc,
                 title="Pics de surdensité des lentilles dans les champs W1 & D1 \n20 000 simulations selon une loi uniforme $\mathcal{U}_{W1}$")
plot_simulations(n_simul=20_000, coord_champ=W2_loc, n_col=8, n_row=8, lentilles_df=lentilles_W2_df,
                 title="Pics de surdensité des lentilles dans le champ W2 \n20 000 simulations selon une loi uniforme $\mathcal{U}_{W2}$")
plot_simulations(n_simul=20_000, coord_champ=W3_loc, n_col=10, n_row=10, lentilles_df=lentilles_W3_df, coord_W=D3_loc,
                 title="Pics de surdensité des lentilles dans le champ W3 et D3 \n20 000 simulations selon une loi uniforme $\mathcal{U}_{W3}$")


# std
plot_simulations(n_simul=20_000, coord_champ=W1_loc, n_col=5, n_row=5, lentilles_df=lentilles_WD1_df, coord_W=D1_loc, dist_method="std",
                 title="Surdensité des lentilles dans les champs W1 & D1 \n20 000 simulations selon une loi uniforme $\mathcal{U}_{W1}$",)
plot_simulations(n_simul=20_000, coord_champ=W2_loc, n_col=3, n_row=3, lentilles_df=lentilles_W2_df, dist_method="std",
                 title="Surdensité des lentilles dans les champ W2  \n20 000 simulations selon une loi uniforme $\mathcal{U}_{W1}$",)
plot_simulations(n_simul=20_000, coord_champ=W3_loc, n_col=5, n_row=5, lentilles_df=lentilles_W3_df, coord_W=D3_loc, dist_method="std",
                 title="Surdensité des lentilles dans le champ W3 et D3 \n20 000 simulations selon une loi uniforme $\mathcal{U}_{W3}$")
