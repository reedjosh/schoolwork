import click
import errno
import matplotlib
from math import exp as e

try:
    matplotlib.use('agg')
except:
    # Some IDE, most likely
    pass

import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.misc
import scipy.stats
import scipy.integrate as integrate
import seaborn as sns

def safe_make(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def plot_cdf_discrete(x, y, label=None):
    """Plots a CDF with discrete probabilities "y" at "x" points.
    """
    ax = plt.gca()

    # Plot the continuous parts
    xerr = np.zeros((2, x.shape[0]))
    xerr[1, :] = 1.
    handles = plt.errorbar(x, y, xerr=xerr, fmt='o', label=label)

    # Keep color the same
    c = handles[0].get_color()
    c = matplotlib.colors.to_rgba(c, True)  # required by LineCollection

    # Plot the connecting (discontinuous) bits
    y_aug = np.r_[[0.], y]
    lines = [ [(x[i], y_aug[i]), (x[i], y_aug[i+1])]
            for i in range(x.shape[0]) ]
    lc = matplotlib.collections.LineCollection(lines, colors=c,
            linestyles='--')
    ax.add_collection(lc)

    # Endcaps
    lines = [[(x[0] - 1000, 0.), (x[0], 0.)],
            [(x[-1], 1.), (x[-1] + 1000, 1.)]]
    lc = matplotlib.collections.LineCollection(lines, colors=c)
    ax.add_collection(lc)


def plot_pdf_discrete(x, y, label=None):
    ax = plt.gca()

    # Assume zero everywhere
    handles = plt.plot(x, np.zeros_like(y), label=label)
    c = handles[0].get_color()

    # Plot point masses as arrows of differing heights
    for xv, yv in zip(x, y):
        ax.arrow(xv, 0, 0, yv, width=0.02, head_length=0.03, head_width=0.08,
                length_includes_head=True,
                ec=c, fc=c)

    # End caps
    lines = [[(x[0] - 1000, 0.), (x[0], 0.)],
            [(x[-1], 0.), (x[-1] + 1000, 0.)]]
    lc = matplotlib.collections.LineCollection(lines, colors=c)
    ax.add_collection(lc)


def plot_scipy_dist(a, b, dist):
    """Plots dist on [a, b]."""
    x = np.linspace(a, b, 101)
    pdf = dist.pdf(x)
    cdf = dist.cdf(x)

    handles = plt.plot(x, pdf, label="PDF")
    cpdf = matplotlib.colors.to_rgba(handles[0].get_color(), True)
    handles = plt.plot(x, cdf, label="CDF")
    ccdf = matplotlib.colors.to_rgba(handles[0].get_color(), True)

    # Add end caps
    x2 = np.linspace(plt.gca().get_xlim()[0], a - 1e-6, 5)
    x3 = np.linspace(b + 1e-6, plt.gca().get_xlim()[1], 5)
    plt.plot(x2, dist.pdf(x2), color=cpdf)
    plt.plot(x3, dist.pdf(x3), color=cpdf)
    plt.plot(x2, dist.cdf(x2), color=ccdf)
    plt.plot(x3, dist.cdf(x3), color=ccdf)


def rescale():
    """Rescale axes based only on labeled artists.
    """
    import matplotlib
    ax = plt.gca()
    ax.dataLim = matplotlib.transforms.Bbox.unit()
    ax.dataLim.set_points(np.asarray([[1e8, 1e8], [-1e8, -1e8]]))
    for obj, _label in zip(*ax.get_legend_handles_labels()):
        xy = None
        if isinstance(obj, matplotlib.lines.Line2D):
            xy = np.vstack(obj.get_data()).T
        elif isinstance(obj, matplotlib.container.ErrorbarContainer):
            xy = np.vstack(obj.get_children()[0].get_data()).T
        elif isinstance(obj, matplotlib.patches.Rectangle):
            # Assume e.g. histogram; use all rectangles
            for obj2 in ax.get_children():
                if isinstance(obj2, matplotlib.patches.Rectangle):
                    ax.dataLim.update_from_data_xy(
                            obj2.get_bbox().get_points(), ignore=False)
        else:
            raise NotImplementedError(obj)
        if xy is not None:
            ax.dataLim.update_from_data_xy(xy, ignore=False)
    ax.autoscale_view()
    plt.tight_layout()


@click.command()
def main():
    """Note that all images will end up in the 'img' folder.
    """
    safe_make('img')
    matplotlib.rcParams['savefig.dpi'] = 150
    sns.set_style('whitegrid')
        
    def plot_210_CDF():
        """Plot the CDF for problem 2.10.
        """
        def f(x):
            """ This is the function for the CDF of problem 2.10.
            """
            A = 1.429
            if x<1:
                return 0.0
            elif x<2:
                return integrate.quad(lambda x: A*e(-x), 1, x)[0]
            elif x<3:                             
                return integrate.quad(lambda x: A*e(-x), 1, x)[0]+1/4
            elif x<4:                             
                return integrate.quad(lambda x: A*e(-x), 1, x)[0]+1/2
            else:
                return 1.0

        x1=np.linspace(0,0.999,200)
        x2=np.linspace(1,1.999,200)
        x3=np.linspace(2,2.999,200)
        x4=np.linspace(3,3.999,200)
        x5=np.linspace(4,4.999,200)
         
        y1 = np.vectorize(f)(x1)
        plt.legend()
        y2 = np.vectorize(f)(x2)
        y3 = np.vectorize(f)(x3)
        y4 = np.vectorize(f)(x4)
        y5 = np.vectorize(f)(x5)


        plt.figure()
        plt.plot(x1, y1)
        plt.plot(x2, y2)
        plt.plot(x3, y3)
        plt.plot(x4, y4)
        plt.plot(x5, y5)
        plt.xlabel('x')
        plt.ylabel('$F_x(x)$')
        #rescale()
        #plt.ylabel('$P[S \leq k]$')
        plt.title("2.10 (CDF)")
        plt.savefig('img/plot_CDF_210.png')
        rescale()
        plt.legend()



if __name__ == '__main__':
    main()

