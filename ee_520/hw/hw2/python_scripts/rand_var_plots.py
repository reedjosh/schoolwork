import click
import errno
import matplotlib

try:
    matplotlib.use('Agg')
except:
    # Some IDE, most likely
    pass
import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.misc
import scipy.stats
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


    def f(x):
        if x <= 1.0:
            return (x/2)**2
        elif x<=2:
            return x/4
        elif x<=10:
            return 1/2
        elif x<=20:
            return x/20
        else:
            return 1
    x=np.linspace(0,21,200)
    
    f2 = np.vectorize(f)
    y = f2(x)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('$F_x(x)$')
    #plt.legend()
    #rescale()
    plt.title("Wait Times (CDF)")
    plt.show() 
    plt.savefig('img/plot-wait-times-cdf.png')

    def f(x):
        if x <= 1.0:
            return (x/2)
        elif x<=2:
            return 1/4
        elif x<=10:
            return 0
        elif x<=20:
            return 1/20
        else:
            return 0
    x=np.linspace(0,21,200)
    
    f2 = np.vectorize(f)
    y = f2(x)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('$F_x(x)$')
    #plt.legend()
    #rescale()
    plt.title("Wait Times (PDF)")
    plt.show() 
    plt.savefig('img/plot-wait-times-pdf.png')


    # My Binomial Plot for Class Project
    plt.figure()
    n = 10
    k = np.arange(n+1)
    p = 0.33
    q = 0.66
    y_actual_noint = scipy.misc.comb(n, k) * p ** k * (1 - p) ** (n - k)
    y_actual = np.cumsum(y_actual_noint)
    plot_cdf_discrete(k, y_actual, label=r"$n = {}$ Binomial Law".format(n))
    print(k)
    print(y_actual)

    n = 15
    k = np.arange(n+1)
    p = 0.33
    q = 0.66
    y_actual_noint = scipy.misc.comb(n, k) * p ** k * (1 - p) ** (n - k)
    y_actual = np.cumsum(y_actual_noint)
    plot_cdf_discrete(k, y_actual, label=r"$n = {}$ Binomial Law".format(n))
    print(k)


    plt.xlabel('k')
    plt.ylabel('$P[S \leq k]$')
    rescale()
    plt.legend()
    plt.savefig('img/plot-discrete-binomapprox.png')



if __name__ == '__main__':
    main()

