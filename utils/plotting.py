import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd 

FIGURES_PATH = "PaperSeek-Report/figures/"
DATA_PATH = "PaperSeek-Report/data/"
COLORS = ["#179c7d", "#005b7f", "#a6bbc8", "#39c1cd", "#b2b235", "#f58220", "#008598"]
DEFAULT_LAYOUT = dict(
    title_x=0.5,
    title_font_family="Modern Computer",
    font_family="Modern Computer",
    showlegend=True,
    legend_title="",
    height=550,
    width=750,
    xaxis_showgrid=True,
    yaxis_showgrid=True,
    xaxis_tickfont_size=15,
    yaxis_tickfont_size=15,
    legend_font_size=25,
    legend_itemsizing="constant",
    legend_x=0.5,
    title_font_size=30,
    colorway=COLORS,
)
pio.templates.default = "presentation"
pio.templates[pio.templates.default].layout.colorway = COLORS


def save_plot(fig: go.Figure, filename: str, layout: dict = {}):
    path = FIGURES_PATH + filename + ".pdf"
    # Update only the adjusted value in the default layout
    layout = (lambda d: d.update(layout) or d)(DEFAULT_LAYOUT.copy())
    fig.update_layout(**layout)
    fig.write_image(path)


def save_data(df: pd.DataFrame, filename: str):
    path = DATA_PATH + filename + ".csv"
    df.to_csv(path, index=False)

def read_data(filename: str):
    path = DATA_PATH + filename + ".csv"
    return pd.read_csv(path)