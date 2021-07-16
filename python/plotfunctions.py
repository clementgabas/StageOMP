# --- Plot functions
def plot_sub_data(subplot_axe_object, axe: [int, int], data1, data2, xlim: (int, int), ylim: (int, int), title: str, invert_x: bool = True, invert_y: bool = False):
    subplot_axe_object[axe[0], axe[1]].set_xlim(xlim)
    subplot_axe_object[axe[0], axe[1]].set_ylim(ylim)
    subplot_axe_object[axe[0], axe[1]].scatter(
        data1, data2, marker="+", c='red')
    subplot_axe_object[axe[0], axe[1]].title.set_text(title)
    if invert_x:
        subplot_axe_object[axe[0], axe[1]].invert_xaxis()
    if invert_y:
        subplot_axe_object[axe[0], axe[1]].invert_yaxis()


def make_global_title(subplot_fig_object, title="", x_title="", y_title=""):
    subplot_fig_object.suptitle(title)
    subplot_fig_object.supxlabel(x_title)
    subplot_fig_object.supylabel(y_title)
