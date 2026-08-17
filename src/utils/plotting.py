"""Matplotlib defaults: serif fonts, LaTeX only if latex + cm-super work."""

import matplotlib.pyplot as plt


def setup_matplotlib(usetex=True):
    """Enable serif fonts. Fall back to mathtext if LaTeX cannot run.

    Matplotlib usetex needs the TeX Live `cm-super` package
    (`sudo apt install cm-super`).
    """
    plt.rc("font", family="serif")
    if not usetex:
        plt.rc("text", usetex=False)
        return False

    plt.rc("text", usetex=True)
    fig, ax = plt.subplots()
    try:
        ax.set_title("test")
        fig.canvas.draw()
        plt.close(fig)
        return True
    except Exception:
        plt.close(fig)
        plt.rc("text", usetex=False)
        return False
