import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from typing import Union, List
from enum import Enum

# TODO (tedious): refactor pyplot to use axes rather than plt
# TODO: improve specs for functions

# Setup plot style

sns.set_style("darkgrid")
plt.rc("axes", labelsize=14)  # fontsize of the x and y labels
plt.rc("font", size=12)  # controls default text sizes
COLOR_PALETTE = np.array(sns.color_palette(palette="Accent"))
COLOR_PALETTE[:5, :] = COLOR_PALETTE[[4, 6, 2, 1, 0], :]

# Covariance Scatter plot


# taken from https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    print(pearson)
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# Categorical Data Pie Charts


def aggregate_infreq(
    counts, namemap=None, max_size=None, other_tot_prop=None, other_cutoff=None
):
    """Takes as input a column of categorical data, returns proportions along with an 'other' bin for low frequency values
    Supports multiple ways of specifying other bin. max_size trumps other_prop trumps other_cutoff.
    May return `None` for the other bin if it was unnecessary
    """
    total_count = counts.sum()
    assert total_count > 0, "total_count should be positive"
    freqs = counts / total_count

    if namemap is not None:
        freqs.rename(namemap, inplace=True)

    if max_size is not None:
        if len(freqs) < max_size:
            return freqs, None
        freqs.sort_values(inplace=True)
        small_vals = freqs[max_size - 2 :]
        freqs.drop(small_vals.index, inplace=True)
        freqs["Other"] = small_vals.sum()
        return freqs, small_vals

    elif other_tot_prop is not None:
        # Aggregate low frequency values
        freqs.sort_values(inplace=True)
        net_sum = 0
        i = 0
        while net_sum < other_tot_prop:
            net_sum += freqs[i]
            i += 1

        small_vals = freqs[:i]
        freqs.drop(small_vals.index, inplace=True)
        freqs.at["Other"] = net_sum

        return freqs, small_vals

    elif other_cutoff is not None:
        small_vals = freqs[freqs <= other_cutoff]
        freqs.drop(small_vals.index, inplace=True)
        freqs["Other"] = small_vals.sum()
        return freqs, small_vals

    return freqs, None


def column_pie_chart(
    df,
    column,
    plot_title,
    color_palette,
    save_path,
    aggregation_kwargs,
    namemap=None,
    pop_other=True,
    has_legend=False,
):
    column_counts = df[column].value_counts()
    # TODO: make it generate a null label automatically if missing value is already a column
    assert (
        "Missing Value" not in column_counts.index
    ), "'Missing Value' must not already be a column value"
    missing_count = df[column].isnull().astype(np.int32).sum()
    if missing_count > 0:
        column_counts["Missing Value"] = missing_count

    main, infreq = aggregate_infreq(column_counts, namemap, **aggregation_kwargs)
    max_len = 30
    main.rename(
        {name: name[:max_len] + "." for name in main.index if len(name) > max_len},
        inplace=True,
    )

    if pop_other:
        explode = np.zeros(len(main))
        explode[-1] = 0.1
    else:
        explode = None

    # TODO: probably shouldn't hardcode default value 1.1

    main.plot.pie(
        y=0,
        title=plot_title,
        labeldistance=None if has_legend else 1.1,
        autopct="%.0f%%",
        colors=color_palette,
        explode=explode,
        pctdistance=0.75,
        # shadow=True,
        startangle=0,
    )
    if has_legend:
        plt.legend(bbox_to_anchor=(0.65, 1), fontsize=11, loc="upper left")
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.8)

    plt.axis("off")
    plt.savefig(save_path)
    plt.close()

    print(f"{plot_title} infrequent values {infreq}")


# Loan funnel analysis

# TODO: refactor to just use strings. KISS
FunnelType = Enum("FunnelType", ["totals", "proportions", "permeability"])
PlotStyle = Enum("PlotStyle", ["bar", "scatter", "line"])


def division_prop_uncertainty(a, b, a_uncertainty, b_uncertainty):
    return np.sqrt(
        np.square(np.divide(a_uncertainty, b))
        + np.square(np.divide(a * b_uncertainty, np.square(b)))
    )


def gen_funnel_data(funnel_type: FunnelType, data, stages, stage_namemap):
    """Generates the desired funnel metrics and uncertainties"""
    stage_names = [stage_namemap[stage] for stage in stages]

    totals = np.array(
        [[df[stage_id].notnull().sum() for stage_id in stages] for df in data]
    )
    total_uncertainties = np.sqrt(totals)

    if funnel_type == FunnelType.totals:
        return stage_names, totals, total_uncertainties

    # uncertainty of a / b = sqrt((delta_a / b)^2 + (a * delta_b/ b^2)^2)
    b = totals[:, 0].reshape((-1, 1))
    b_uncertainty = total_uncertainties[:, 0].reshape((-1, 1))

    props = np.divide(totals, b)
    prop_uncertainties = division_prop_uncertainty(
        totals, b, total_uncertainties, b_uncertainty
    )
    if funnel_type == FunnelType.proportions:
        return stage_names, props, prop_uncertainties

    shifted = np.roll(props, 1, axis=-1)
    shifted_uncertainties = np.roll(prop_uncertainties, 1, axis=-1)
    permeabilities = np.divide(props, shifted)[:, 1:]
    permeability_uncertainties = division_prop_uncertainty(
        props, shifted, prop_uncertainties, shifted_uncertainties
    )[:, 1:]

    if funnel_type == FunnelType.permeability:
        return (stage_names[:-1], permeabilities, permeability_uncertainties)

    raise ValueError(f"{funnel_type} is not supported!")


def bar_plot(
    ax, xs, ys, plot_multiple, palette, series_labels=None, total_bar_width=0.8
):
    bin_spacing = xs[1] - xs[0]
    total_bar_width *= bin_spacing
    if plot_multiple:
        num_to_plot = len(ys)
        bar_width = total_bar_width / num_to_plot
        offsets = (
            i * bar_width - bin_spacing * total_bar_width / 2
            for i in range(num_to_plot)
        )
        labels = iter(series_labels)
        for series in ys:
            ax.bar(
                xs + next(offsets),
                series,
                width=bar_width,
                align="center",
                color=next(palette),
                label=next(labels),
            )
        ax.legend(loc="upper right")
    else:
        assert xs is not None, "ahhhhhhh"
        ax.bar(xs, ys[0], width=total_bar_width, color=next(palette))
    print([series[-1] for series in ys])


def line_plot(ax, xs, ys, plot_multiple, palette, series_labels=None):
    if plot_multiple:
        for series in ys:
            ax.plot(xs, series, color=next(palette), label=next(series_labels))
        ax.legend(loc="upper right")
    else:
        ax.plot(xs, ys[0], color=next(palette))


def error_scatter(
    ax, xs, ys, uncertainties, plot_multiple, palette, series_labels=None
):
    if plot_multiple:
        for series, errs in zip(ys, uncertainties):
            ax.errorbar(
                xs,
                series,
                yerr=errs,
                marker="o",
                capsize=2,
                ls="none",
                elinewidth=2,
                color=next(palette),
                label=next(series_labels),
            )
        ax.legend(loc="upper right")
    else:
        ax.errorbar(
            xs,
            ys[0],
            yerr=uncertainties[0],
            marker="o",
            capsize=2,
            ls="none",
            color=next(palette),
            label=next(series_labels),
        )


def box_plot(ax, data, palette, labels, orientation="vert"):
    # rectangular box plot
    o_map = {"vert": True, "hor": False}
    bplot = ax.boxplot(data, vert=o_map[orientation], patch_artist=True, labels=labels)
    for patch in bplot["boxes"]:
        patch.set_facecolor(next(palette))


def funnel_plot(
    data: Union[pd.DataFrame, List[pd.DataFrame]],
    stages,
    stages_namemap,
    funnel_name,
    save_path,
    logging=False,
    labels=None,
    funnel_type=FunnelType.proportions,
    plot_style=PlotStyle.scatter,
):
    """all `data` DataFrames must have columns matching the properties in stages"""

    plot_multiple = isinstance(data, list) or isinstance(data, tuple)
    if plot_multiple:
        assert len(data) == len(labels), "There must be a label for each data series"
    else:
        data = [data]

    xs, ys, uncertainties = gen_funnel_data(funnel_type, data, stages, stages_namemap)

    if logging:
        print(xs)
        print(ys)
        print(uncertainties)

    fig, ax = plt.subplots(1, 1)
    palette = iter(COLOR_PALETTE)
    labels = iter(labels)

    bins = xs
    if isinstance(xs[0], str) and plot_style == PlotStyle.bar:
        # for bars with no inherent meaning, we use integers
        bins = np.array(range(len(xs)))

    match plot_style:
        case PlotStyle.line:
            line_plot(ax, xs, ys, plot_multiple, palette, labels)
        case PlotStyle.scatter:
            error_scatter(ax, xs, ys, uncertainties, plot_multiple, palette, labels)
        case PlotStyle.bar:
            bar_plot(ax, bins, ys, plot_multiple, palette, labels)

    ax.set_title(f"{funnel_name} Funnel {str(funnel_type.name).capitalize()}")
    ax.set_xlabel("Stage")
    ax.set_ylabel(f"{funnel_name} {funnel_type.name.capitalize()}")
    if funnel_type in (FunnelType.proportions, FunnelType.permeability):
        ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xticks(ticks=range(len(xs)), labels=xs, rotation=30)
    fig.savefig(
        save_path,
        bbox_inches="tight",
    )
