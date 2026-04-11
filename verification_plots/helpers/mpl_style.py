import matplotlib.pyplot as plt


def apply_mpl_style():
    plt.style.use(["science", "grid"])

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "font.sans-serif": ["cmss10"],
        "font.monospace": ["cmtt10"],
        "axes.formatter.use_mathtext": True,
        "font.size": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.titlesize": 20,
        "figure.titlesize": 20,
        # "text.latex.preamble": r"""
        #     \usepackage[T1]{fontenc}
        #     \usepackage{amsmath}
        #     \renewcommand{\familydefault}{\sfdefault}
        #     \usepackage{sansmath}
        #     \sansmath
        # """,
    })