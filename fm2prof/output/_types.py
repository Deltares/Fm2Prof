from typing import NamedTuple

import matplotlib as mpl


class FigureOutput(NamedTuple):
    fig: mpl.figure.Figure
    axes: mpl.axes.Axes
    legend: mpl.legend.Legend


class StyleGuide(NamedTuple):
    font: dict
    major_grid: dict
    minor_grid: dict
    spine_width: float
