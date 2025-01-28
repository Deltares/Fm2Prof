"""Plot styles for Fm2Prof output figures."""

from __future__ import annotations

import locale
import warnings
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

from fm2prof.output._types import StyleGuide

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend


register_matplotlib_converters()

COLORSCHEMES = {
    "Deltares": ["#000000", "#00cc96", "#0d38e0"],
    "Koeln": [
        "#000000",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E444",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
    ],  # https://jfly.uni-koeln.de/color/
    "PaulTolVibrant": [
        "#0077BB",
        "#33BBEE",
        "#009988",
        "#EE7733",
        "#CC3311",
        "#EE3377",
        "#BBBBBB",
    ],
}


class PlotStyles:
    """Class for handling and applying plot styles."""

    my_fmt = mdates.DateFormatter("%d-%b")
    monthlocator = mdates.MonthLocator(bymonthday=(1, 10, 20))
    daylocator = mdates.DayLocator(interval=5)
    colorscheme = COLORSCHEMES["Koeln"]

    @staticmethod
    def set_locale(locale_string: str) -> None:
        """Set locale."""
        try:
            locale.setlocale(locale.LC_TIME, locale_string)
        except locale.Error as err:
            # known error on linux fix:
            # export LC_ALL="en_US.UTF-8" & export LC_CTYPE="en_US.UTF-8" & sudo dpkg-reconfigure locales
            err_msg = f"could not set locale to {locale_string}"
            raise locale.Error(err_msg) from err

    @staticmethod
    def _is_timeaxis(axis: Axes) -> bool:
        try:
            label_string = axis.get_ticklabels()[0].get_text().replace("−", "-")  # noqa:RUF001
            # if label_string is empty (e.g. because of twin_axis, return false)
            if label_string:
                float(label_string)

        except ValueError:
            return True
        except IndexError:
            return False
        return False

    @classmethod
    def van_veen(
        cls,
        fig: Figure | None = None,
        *,
        use_legend: bool = True,
        extra_labels: list | None = None,
        ax_align_legend: plt.Axes | None = None,
    ) -> None:
        """Apply van veen plotstyle."""
        warnings.warn(  # noqa: B028
            "This function is deprecated and will be removed on future versions."
            "Use PlotStyle.apply(fig, style='van_veen') instead",
            category=DeprecationWarning,
        )
        cls.apply(
            fig=fig,
            style="van_veen",
            use_legend=use_legend,
            extra_labels=extra_labels,
            ax_align_legend=ax_align_legend,
        )

    @classmethod
    def apply(
        cls,
        fig: Figure | None = None,
        style: str = "sito",
        extra_labels: list | None = None,
        ax_align_legend: plt.Axes | None = None,
        *,
        use_legend: bool = True,
    ) -> tuple[Figure, Legend]:
        """Apply style to figure."""
        styles: dict[str, StyleGuide] = {
            "sito": StyleGuide(
                font={"family": "Franca, Arial", "weight": "normal", "size": 16},
                major_grid={
                    "visible": True,
                    "which": "major",
                    "linestyle": "--",
                    "linewidth": 1.0,
                    "color": "#BBBBBB",
                },
                minor_grid={
                    "visible": True,
                    "which": "minor",
                    "linestyle": "--",
                    "linewidth": 1.0,
                    "color": "#BBBBBB",
                },
                spine_width=1,
            ),
            "van_veen": StyleGuide(
                font={"family": "Bahnschrift", "weight": "normal", "size": 18},
                major_grid={"visible": True, "which": "major", "linestyle": "-", "linewidth": 1, "color": "k"},
                minor_grid={"visible": True, "which": "minor", "linestyle": "-", "linewidth": 0.5, "color": "k"},
                spine_width=2,
            ),
        }

        if style not in styles:
            err_msg = f"unknown style {style}. Options are {list(styles.keys())}"
            raise KeyError(err_msg)

        style_guide: StyleGuide = styles.get(style)

        if not fig:
            return cls._initiate(style_guide)
        cls._initiate(style_guide)

        return cls._style_figure(
            style_guide=style_guide,
            fig=fig,
            use_legend=use_legend,
            extra_labels=extra_labels,
            ax_align_legend=ax_align_legend,
        )

    @classmethod
    def _initiate(cls, style_guide: StyleGuide, locale: str = "nl_NL.UTF-8") -> None:
        PlotStyles.set_locale(locale)

        # Color style
        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
            color=cls.colorscheme * 3,
            linestyle=["-"] * len(cls.colorscheme) + ["--"] * len(cls.colorscheme) + ["-."] * len(cls.colorscheme),
        )

        # Font style
        font = style_guide.font

        # not all fonts support the unicode minus, so disable this option
        mpl.rc("font", **font)
        mpl.rcParams["axes.unicode_minus"] = False

    @classmethod
    def _style_figure(
        cls,
        style_guide: StyleGuide,
        fig: Figure | None,
        extra_labels: list | None,
        ax_align_legend: Axes | None,
        *,
        use_legend: bool,
    ) -> tuple[Figure, Legend] | tuple[Figure, list] | None:
        if ax_align_legend is None:
            ax_align_legend = fig.axes[0]

        # this forces labels to be generated. Necessary to detect datetimes
        fig.canvas.draw()

        # Set styles for each axis
        legend_title = r"Toelichting"
        handles = []
        labels = []

        for ax in fig.axes:
            # Enable grid grid
            ax.grid(**style_guide.major_grid)
            ax.grid(**style_guide.minor_grid)

            for spine in ax.spines.values():
                spine.set_linewidth(style_guide.spine_width)

            if cls._is_timeaxis(ax.xaxis):
                ax.xaxis.set_major_formatter(cls.my_fmt)
                ax.xaxis.set_major_locator(cls.monthlocator)
            if cls._is_timeaxis(ax.yaxis):
                ax.yaxis.set_major_formatter(cls.my_fmt)
                ax.yaxis.set_major_locator(cls.monthlocator)

            ax.patch.set_visible(False)
            h, lab = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(lab)

        if extra_labels:
            handles.extend(extra_labels[0])
            labels.extend(extra_labels[1])
        fig.tight_layout()
        if use_legend:
            lgd = fig.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(1.0, ax_align_legend.get_position().y1),
                bbox_transform=fig.transFigure,
                edgecolor="k",
                facecolor="white",
                framealpha=1,
                borderaxespad=0,
                title=legend_title.upper(),
            )

            return fig, lgd
        return fig, handles, labels
