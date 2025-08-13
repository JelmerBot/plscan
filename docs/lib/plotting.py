import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def configure_matplotlib():
    mpl.rcParams["text.color"] = "black"
    mpl.rcParams["xtick.color"] = "black"
    mpl.rcParams["ytick.color"] = "black"
    mpl.rcParams["axes.labelcolor"] = "black"
    mpl.rcParams["xtick.bottom"] = True
    mpl.rcParams["ytick.left"] = True
    mpl.rcParams["ytick.major.size"] = mpl.rcParams["xtick.major.size"]
    mpl.rcParams["ytick.major.width"] = mpl.rcParams["xtick.major.width"]
    mpl.rcParams["font.size"] = 8
    mpl.rcParams["axes.labelsize"] = 8
    mpl.rcParams["axes.titlesize"] = 10
    mpl.rcParams["legend.fontsize"] = 8
    mpl.rcParams["legend.title_fontsize"] = 8
    mpl.rcParams["xtick.labelsize"] = 8
    mpl.rcParams["ytick.labelsize"] = 8
    mpl.rcParams["axes.unicode_minus"] = True
    mpl.rcParams["axes.spines.left"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.bottom"] = False
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.format"] = "png"
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams[
        "text.latex.preamble"
    ] = """
    \\usepackage{libertine}
    \\renewcommand\\sfdefault{ppl}
    """


def sized_fig(width=0.5, aspect=0.618, dpi=None):
    """Create a figure with width as fraction of an A4 page."""
    if dpi is None:
        dpi = 150
    page_width_cm = 13.9
    inch = 2.54
    w = width * page_width_cm
    h = aspect * w
    return plt.figure(figsize=(w / inch, h / inch), dpi=dpi)


def highpassColors(cmap_name="Blues", min_frac=0.2):
    """Clamp blues colomap to avoid close-to-white colors."""

    # Retrieve blues colormap
    cmap = plt.get_cmap(cmap_name)
    if min_frac == 0.0:
        return cmap

    # Extract color segments
    if not hasattr(cmap, "_segmentdata"):
        raise ValueError(f"{cmap_name} is not a LinearSegmentedColormap.")
    segments = cmap._segmentdata
    values = np.asarray([segments["red"][i, 0] for i in range(9)])
    colors = [
        (segments["red"][i, 1], segments["green"][i, 1], segments["blue"][i, 1])
        for i in range(9)
    ]

    # Remove segments below min_frac
    idx = np.where(values > min_frac)[0][0] - 1
    values = values[idx:]
    colors = colors[idx:]

    # Set first segment to be min_frac's color
    values[0] = min_frac
    colors[0] = cmap(min_frac)

    # Scale resulting values to range [0, 1]
    values -= min_frac
    values /= 1 - min_frac

    # Build resulting colormap
    return LinearSegmentedColormap.from_list(
        f"HP{cmap_name}", list(zip(values, colors))
    )


def subplot_fractions(n_plots, n_colorbars, colorbar_fraction=0.15):
    """
    Computes subplot widths as fractions of a page, reserving spaces
    for colorbars. Make sure each subplots content-area is sized
    equally.
    """
    colorbar_ratio = (colorbar_fraction) / (1 - colorbar_fraction)
    composition = n_plots + n_colorbars * colorbar_ratio
    plot_frac = 1 / composition
    colorbar_frac = colorbar_ratio * plot_frac
    return plot_frac, plot_frac + colorbar_frac


def frame_off():
    """Disables frames and ticks, sets aspect ratio to 1."""
    plt.xticks([])
    plt.yticks([])
    for spine in plt.gca().spines.values():
        spine.set_visible(False)


def adjust_legend_subtitles(legend):
    """
    Make invisible-handle "subtitles" entries look more like titles.
    Adapted from seaborn.utils.
    """
    # Legend title not in rcParams until 3.0
    font_size = plt.rcParams.get("legend.title_fontsize", None)
    vpackers = legend.findobj(mpl.offsetbox.VPacker)
    for vpacker in vpackers[:-1]:
        hpackers = vpacker.get_children()
        for hpack in hpackers:
            draw_area, text_area = hpack.get_children()
            handles = draw_area.get_children()
            if not all(artist.get_visible() for artist in handles):
                draw_area.set_width(0)
                for text in text_area.get_children():
                    if font_size is not None:
                        text.set_size(font_size)
