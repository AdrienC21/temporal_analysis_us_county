"""
Create automatically the python script (script_to_convert.py)
in the correct format.
It will be then converted into a notebook "Analysis" when running the
script run_script_to_notebook.py
"""

type_data_list = ["COVID-19", "All Causes", "Excess Mortality"]
features_list = [("political_leaning", "Political Leaning", 0, -0.5, 0.5),
                 ("obesity", "Obesity %", 35, None, None),
                 ("pct_jail", "Jail population %", 0, None, None),
                 ("pct_nursing", "Nursing population %", 0, None, None),
                 ("income", "Median Household Income", 0, None, None)]
SHOW_FIGURES = False

script = """
#| # Imports and load data

# upgrade excel package to load specific files
# %pip install -U xlrd
# county choropleth graph
# %pip install -U geopandas
# %pip install -U pyshp
# %pip install -U shapely
# %pip install -U plotly-geo
# %pip install -U xgboost
# %pip install -U lightgbm
# %pip install -U scikit-learn
# %pip install -U tslearn

#-------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm

#-------------------------------

community_color_dic = {"African American South": "#457a59",
                       "Aging Farmlands": "#7a3842",
                       "Big Cities": "#c43b82",
                       "College Towns": "#c44244",
                       "Evangelical Hubs": "#82477f",
                       "Exurbs": "#fcb93a",
                       "Graying America": "#2e547a",
                       "Hispanic Centers": "#1f8fba",
                       "LDS Enclaves": "#3a2c70",
                       "Middle Suburbs": "#699246",
                       "Military Posts": "#abbf48",
                       "Native American Lands": "#eacd3f",
                       "Rural Middle America": "#3a9c9b",
                       "Urban Suburbs": "#f08031",
                       "Working Class Country": "#86563e"}
acp_dic = {1: "Exurbs",
           2: "Graying America",
           3: "African American South",
           4: "Evangelical Hubs",
           5: "Working Class Country",
           6: "Military Posts",
           7: "Urban Suburbs",
           8: "Hispanic Centers",
           9: "Native American Lands",
           10: "Rural Middle America",
           11: "College Towns",
           12: "LDS Enclaves",
           13: "Aging Farmlands",
           14: "Big Cities",
           15: "Middle Suburbs"}

#| Execute the cells below to directly import the datasets (instead of calculate everything)

type_data = "COVID-19"  # default type: All Causes, COVID-19, Excess Mortality

county_databases = {}

# Load all causes dataset
county_database = pd.read_csv("county_database_all_causes.csv")
county_database.index = county_database.FIPS
county_database2 = pd.read_csv("county_database2_all_causes.csv")
county_database2.index = county_database2.FIPS
county_database2_imputed = pd.read_csv("county_database2_imputed_all_causes.csv")
county_database2_imputed.index = county_database2_imputed.FIPS

county_database.drop(columns=["FIPS.1"], inplace=True)
county_database2.drop(columns=["FIPS.1"], inplace=True)
county_database2_imputed.drop(columns=["FIPS.1"], inplace=True)

county_databases["All Causes"] = {}
county_databases["All Causes"]["county_database"] = county_database
county_databases["All Causes"]["county_database2"] = county_database2
county_databases["All Causes"]["county_database2_imputed"] = county_database2_imputed

# Load COVID-19 dataset
county_database = pd.read_csv("county_database_covid19.csv")
county_database.index = county_database.FIPS
county_database2 = pd.read_csv("county_database2_covid19.csv")
county_database2.index = county_database2.FIPS
county_database2_imputed = pd.read_csv("county_database2_imputed_covid19.csv")
county_database2_imputed.index = county_database2_imputed.FIPS

county_database.drop(columns=["FIPS.1"], inplace=True)
county_database2.drop(columns=["FIPS.1"], inplace=True)
county_database2_imputed.drop(columns=["FIPS.1"], inplace=True)

county_databases["COVID-19"] = {}
county_databases["COVID-19"]["county_database"] = county_database
county_databases["COVID-19"]["county_database2"] = county_database2
county_databases["COVID-19"]["county_database2_imputed"] = county_database2_imputed

# Load Excess Mortality dataset
county_database = pd.read_csv("county_database_excess_mortality.csv")
county_database.index = county_database.FIPS
county_database2 = pd.read_csv("county_database2_excess_mortality.csv")
county_database2.index = county_database2.FIPS
county_database2_imputed = pd.read_csv("county_database2_imputed_excess_mortality.csv")
county_database2_imputed.index = county_database2_imputed.FIPS

county_database.drop(columns=["FIPS.1"], inplace=True)
county_database2.drop(columns=["FIPS.1"], inplace=True)
county_database2_imputed.drop(columns=["FIPS.1"], inplace=True)

county_databases["Excess Mortality"] = {}
county_databases["Excess Mortality"]["county_database"] = county_database
county_databases["Excess Mortality"]["county_database2"] = county_database2
county_databases["Excess Mortality"]["county_database2_imputed"] = county_database2_imputed

with open("feature_selection", "rb") as fp:  # load feature selection
  features = pickle.load(fp)
  selected_columns_list = features[0]
  X_selected_columns_list = features[1]

# load default dataset
county_database = county_databases[type_data]["county_database"]
county_database2 = county_databases[type_data]["county_database2"]
county_database2_imputed = county_databases[type_data]["county_database2_imputed"]

#| # Functions and additional imports

#| Initialization

import geopandas
import shapely
import shapefile
import plotly
from plotly.figure_factory._county_choropleth import create_choropleth

import plotly.express as px
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties_json = json.load(response)
# duplicate the county with no 0 at the beginning of the FIPS
list_of_geo = counties_json["features"]
list_of_id = [v["id"] for v in counties_json["features"]]
for cty in counties_json["features"]:
  new_id = str(int(cty["id"]))
  if not(new_id in list_of_id):
    list_of_id.append(new_id)
    new_cty = cty.copy()
    new_cty["id"] = new_id
    list_of_geo.append(new_cty)
counties_json["features"] = list_of_geo

def create_custom_choropleth(county_database, counties_json, label_col,
                             label_display, color_continuous_scale=None,
                             range_color=None):
  if range_color is None:
    range_color_min = county_database[label_col].min()
    range_color_max = county_database[label_col].max()
  else:
    range_color_min = range_color[0]
    range_color_max = range_color[1]
  fig = px.choropleth(county_database, geojson=counties_json, locations="FIPS",
                      color=label_col,
                      color_continuous_scale=color_continuous_scale,
                      range_color=(range_color_min, range_color_max),
                      scope="usa",
                      labels={label_col:label_display},
                      title="USA by {}".format(label_display)
                      )
  fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
  fig.show()

#| Custom function to plot the scatter plots with trends

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

def custom_scatter_plot(feat_col="political_leaning",
                        feat_name="Political Leaning", *,
                        type_data="COVID-19",
                        cmid=None,
                        min_val=None, max_val=None,
                        filter=None, filter_threshold=None,
                        show_figures=True,
                        **kwargs):

  additional_text = ""  # for the title
  height = 600  # height of the plot

  county_db_to_use = county_databases[type_data]["county_database2_imputed"]

  if "filter_equality" in kwargs:
    county_db_to_use = county_db_to_use[county_db_to_use[kwargs.get("filter_equality")] == kwargs.get("filter_equality_value")]
    additional_text = additional_text + kwargs.get("additional_text", "")

  if filter is None:
    county_db = county_db_to_use
    county_db2 = county_db_to_use  # not used
    rows = 2
  else:
    county_db = county_db_to_use[county_db_to_use[filter] <= filter_threshold]
    county_db2 = county_db_to_use[county_db_to_use[filter] > filter_threshold]
    rows = 4

  if cmid is None:
    cmid = 0

  # shared_yaxes="all" or missing from kwargs
  shared_yaxes = kwargs.get("shared_yaxes", False)
  fig = make_subplots(rows=rows, cols=3,
                      subplot_titles=tuple(["Period {i}".format(i=i) for i in range(1, 7)]),
                      shared_yaxes=shared_yaxes)
  for per in range(6):
    fig.add_trace(
        go.Scatter(x=county_db[feat_col],
                   y=county_db[f"deathRate_period{per+1}"],
                   mode="markers",
                   marker_color=county_db[feat_col],
                   marker_colorscale="rdbu_r",
                   marker={"cmid": cmid},
                   name=f"Period {per+1}",
                   showlegend=False),
        row=(per//3 + 1), col=(per%3 + 1)
    )

    dt = county_db[[feat_col, f"deathRate_period{per+1}", "total_pop"]].dropna()
    lr = LinearRegression(fit_intercept=True)
    lr.fit(dt[feat_col].to_numpy().reshape(-1, 1),
          dt[f"deathRate_period{per+1}"].to_numpy(),
          np.log(dt["total_pop"]).to_numpy())  # log total pop sample weight

    slope = round(lr.coef_[0], 2)

    if min_val is None:
      min_val2 = county_db[feat_col].min()
    else:
      min_val2 = min_val
    if max_val is None:
      max_val2 = county_db[feat_col].max()
    else:
      max_val2 = max_val
    y_pred = lr.predict(np.linspace(min_val2, max_val2, 1000).reshape(-1, 1))
    fig.append_trace(go.Scatter(x=np.linspace(min_val2, max_val2, 1000), y=y_pred,
                                name=f"Regression Period {per+1}. Slope={slope}",
                                line=dict(color="black", width=4,
                                          dash="dash")
                                ), row=(per//3 + 1), col=(per%3 + 1))

    if not(filter is None):
      fig.add_trace(
        go.Scatter(x=county_db2[feat_col],
                   y=county_db2[f"deathRate_period{per+1}"],
                   mode="markers",
                   marker_color=county_db2[feat_col],
                   marker_colorscale="rdbu_r",
                   marker={"cmid": cmid},
                   name=f"Period {per+1}, {filter} above",
                   showlegend=False),
        row=(per//3 + 3), col=(per%3 + 1)
      )
      dt = county_db2[[feat_col, f"deathRate_period{per+1}", "total_pop"]].dropna()
      lr = LinearRegression(fit_intercept=True)
      lr.fit(dt[feat_col].to_numpy().reshape(-1, 1),
            dt[f"deathRate_period{per+1}"].to_numpy(),
            np.log(dt["total_pop"]).to_numpy())  # log total pop sample weight

      slope = round(lr.coef_[0], 2)

      if min_val is None:
        min_val2 = county_db2[feat_col].min()
      else:
        min_val2 = min_val
      if max_val is None:
        max_val2 = county_db2[feat_col].max()
      else:
        max_val2 = max_val
      y_pred = lr.predict(np.linspace(min_val2, max_val2, 1000).reshape(-1, 1))
      fig.append_trace(go.Scatter(x=np.linspace(min_val2, max_val2, 1000), y=y_pred,
                                  name=f"Regression Period {per+1}, {filter} above. Slope={slope}",
                                  line=dict(color="black", width=4,
                                            dash="dash")
                                  ), row=(per//3 + 3), col=(per%3 + 1))
      additional_text = additional_text + f"<br>{filter} <= {filter_threshold} compared to {filter} above"
      height = 1200  # higher image is mandatory as we add more rows
  title_text = f"{type_data} Crude Death Rate and {feat_name} by period{additional_text}" if type_data != "Excess Mortality" else f"{type_data} and {feat_name} by period{additional_text}"
  yaxis_title = f"{type_data} crude death rate" if type_data != "Excess Mortality" else f"{type_data}"
  fig.update_layout(height=height, width=800,
                    title_text=title_text,
                    xaxis_title=f"{feat_name}",
                    yaxis_title=yaxis_title,
                    legend_title="Slopes")
  filename = "Plot/Analysis/{feat_col}_{type_data}".format(feat_col=feat_col,
                                                           type_data=type_data)
  if not(filter is None):
    filename += "filter_" + str(filter)
  if "filter_equality" in kwargs:
    filename += "_" + str(kwargs.get("filter_equality")) + "_" + str(kwargs.get("filter_equality_value"))
  else:
    filename += "_national"
  fig.write_image(filename + ".png")
  if show_figures:
    fig.show()

#| Function for the stringency index (as it is a function of the period)

def custom_scatter_plot_stringency(*,
                                   min_val=None, max_val=None,
                                   type_data="COVID-19",
                                   filter=None, filter_threshold=None,
                                   show_figures=True):
  feat_name = "Stringency Index"

  county_db_to_use = county_databases[type_data]["county_database2_imputed"]

  if filter is None:
    county_db = county_db_to_use
    county_db2 = county_db_to_use  # not used
    rows = 2
  else:
    county_db = county_db_to_use[county_db_to_use[filter] <= filter_threshold]
    county_db2 = county_db_to_use[county_db_to_use[filter] > filter_threshold]
    rows = 4

  fig = make_subplots(rows=rows, cols=3,
                      subplot_titles=tuple(["Period {i}".format(i=i) for i in range(1, 7)]),
                      shared_yaxes="all")
  for per in range(6):
    fig.add_trace(
        go.Scatter(x=county_db[f"StringencyIndex{per+1}_mean"],
                   y=county_db[f"deathRate_period{per+1}"],
                   mode="markers",
                   marker_color=county_db[f"StringencyIndex{per+1}_mean"],
                   marker_colorscale="rdbu_r",
                   marker={"cmid": 37},
                   name=f"Period {per+1}",
                   showlegend=False),
        row=(per//3 + 1), col=(per%3 + 1)
    )

    dt = county_db[[f"StringencyIndex{per+1}_mean", f"deathRate_period{per+1}", "total_pop"]].dropna()
    lr = LinearRegression(fit_intercept=True)
    lr.fit(dt[f"StringencyIndex{per+1}_mean"].to_numpy().reshape(-1, 1),
           dt[f"deathRate_period{per+1}"].to_numpy(),
           np.log(dt["total_pop"]).to_numpy())  # log total pop sample weight

    slope = round(lr.coef_[0], 2)

    if min_val is None:
      min_val2 = county_db[f"StringencyIndex{per+1}_mean"].min()
    else:
      min_val2 = min_val
    if max_val is None:
      max_val2 = county_db[f"StringencyIndex{per+1}_mean"].max()
    else:
      max_val2 = max_val

    y_pred = lr.predict(np.linspace(min_val2, max_val2, 1000).reshape(-1, 1))
    fig.append_trace(go.Scatter(x=np.linspace(min_val2, max_val2, 1000), y=y_pred,
                                name=f"Regression Period {per+1}. Slope={slope}",
                                line=dict(color="black", width=4,
                                          dash="dash")
                                ), row=(per//3 + 1), col=(per%3 + 1))
    additional_text = ""  # for the title
    height = 600  # height of the plot

    if not(filter is None):
      fig.add_trace(
        go.Scatter(x=county_db2[f"StringencyIndex{per+1}_mean"],
                   y=county_db2[f"deathRate_period{per+1}"],
                   mode="markers",
                   marker_color=county_db2[f"StringencyIndex{per+1}_mean"],
                   marker_colorscale="rdbu_r",
                   marker={"cmid": 37},
                   name=f"Period {per+1}, {filter} above",
                   showlegend=False),
        row=(per//3 + 3), col=(per%3 + 1)
      )
      dt = county_db2[[f"StringencyIndex{per+1}_mean", f"deathRate_period{per+1}", "total_pop"]].dropna()
      lr = LinearRegression(fit_intercept=True)
      lr.fit(dt[f"StringencyIndex{per+1}_mean"].to_numpy().reshape(-1, 1),
            dt[f"deathRate_period{per+1}"].to_numpy(),
            np.log(dt["total_pop"]).to_numpy())  # log total pop sample weight

      slope = round(lr.coef_[0], 2)

      if min_val is None:
        min_val2 = county_db2[f"StringencyIndex{per+1}_mean"].min()
      else:
        min_val2 = min_val
      if max_val is None:
        max_val2 = county_db2[f"StringencyIndex{per+1}_mean"].max()
      else:
        max_val2 = max_val
      y_pred = lr.predict(np.linspace(min_val2, max_val2, 1000).reshape(-1, 1))
      fig.append_trace(go.Scatter(x=np.linspace(min_val2, max_val2, 1000), y=y_pred,
                                  name=f"Regression Period {per+1}, {filter} above. Slope={slope}",
                                  line=dict(color="black", width=4,
                                            dash="dash")
                                  ), row=(per//3 + 3), col=(per%3 + 1))
      additional_text = f"<br>{filter} <= {filter_threshold} compared to {filter} above"
      height = 1200  # higher image is mandatory as we add more rows
  title_text = f"{type_data} Crude Death Rate and {feat_name} by period{additional_text}" if type_data != "Excess Mortality" else f"{type_data} and {feat_name} by period{additional_text}"
  yaxis_title = f"{type_data} crude death rate" if type_data != "Excess Mortality" else f"{type_data}"
  fig.update_layout(height=height, width=800,
                    title_text=title_text,
                    xaxis_title=f"{feat_name}",
                    yaxis_title=yaxis_title,
                    legend_title="Slopes")
  fig.write_image("Plot/Analysis/StringencyIndex_{type_data}_filter-{filter}.png".format(type_data=type_data,
                                                                                         filter=filter))
  if show_figures:
    fig.show()

#| # Plots
"""

script += """
#| ## Analysis at the national level
"""


def add_to_script(feature, feature_name, cmid, min_val, max_val):
    global SHOW_FIGURES
    s = ""
    s += f"""
#| ### {feature_name}
"""
    # add box plot
    s += """
#| Box plot
"""
    s += f"""
fig = go.Figure()
fig.add_trace(go.Box(y=county_database["{feature}"],
                     name="National"))
"""
    if feature == "political_leaning":
      s += """
fig.add_hline(y=0, line_dash="dash")
"""
    s += f"""
fig.update_layout(title_text="Box Plot {feature_name} - National level",
                  yaxis_title="{feature_name}")
fig.write_image("Plot/Analysis/boxplot_{feature}_national.png")
"""
    if SHOW_FIGURES:
      s += """
fig.show()
"""

    for type_data in type_data_list:

      s += f"""
#| {type_data}
"""
      s += """
kwargs = {"shared_yaxes": "all"}
"""
      s += f"""
custom_scatter_plot(feat_col="{feature}",
                    feat_name="{feature_name}",
                    type_data="{type_data}",
                    min_val={min_val}, max_val={max_val},
                    cmid={cmid},
                    show_figures={SHOW_FIGURES},
                    **kwargs)
"""
    return s


for feature, feature_name, cmid, min_val, max_val in features_list:
    script += add_to_script(feature, feature_name, cmid, min_val, max_val)

script += f"""
#| ### Stringency index (at each period)
custom_scatter_plot_stringency(type_data="COVID-19",
                               show_figures={SHOW_FIGURES})
custom_scatter_plot_stringency(type_data="All Causes",
                               show_figures={SHOW_FIGURES})
custom_scatter_plot_stringency(type_data="Excess Mortality",
                               show_figures={SHOW_FIGURES})
"""

script += """
#| ## Analysis at the community level

import plotly.express as px
import math
# rename communities
fig = px.pie(county_database, names="acp_name", title="Number of counties per communities",
             color="acp_name",
             color_discrete_map=community_color_dic,
             width=1200, height=720)
fig.write_image("Plot/Analysis/pie-chart-number-communities.png")
"""
if SHOW_FIGURES:
  script += """
fig.show()
"""

script += """
#-------------------------------

import plotly.express as px
# rename communities
bypop = county_database.groupby(by="acp_name")["total_pop"].sum().to_frame()
bypop.reset_index(drop=False, inplace=True)
bypop
fig = px.pie(bypop, names="acp_name", values="total_pop",
             title="Population per communities",
             color="acp_name",
             color_discrete_map=community_color_dic,
             width=1200, height=720)
fig.update_traces(textposition="inside", textinfo="percent+label")
fig.update_layout(legend_title="American Communities (id & name)")
fig.write_image("Plot/Analysis/pie-chart-population-communities.png")
"""

if SHOW_FIGURES:
  script += """
fig.show()
"""

script += """
#| Scatter plots
"""


def add_to_script_communities(feature, feature_name, cmid, min_val, max_val):
    global SHOW_FIGURES
    s = ""
    s += f"""
#| ### {feature_name}
"""
    # add box plot
    s += """
#| Box plot
"""
    s += f"""
fig = go.Figure()
for acp_nb in range(1, 16):
    acp_name = acp_dic[acp_nb]
    fig.add_trace(go.Box(y=county_database[county_database["acp"] == acp_nb]["{feature}"],
                         name=acp_name,
                         marker_color=community_color_dic[acp_name]))
"""
    if feature == "political_leaning":
      s += """
fig.add_hline(y=0, line_dash="dash")
"""
    s += f"""
fig.update_layout(title_text="Box Plot {feature_name} - Community level",
                  yaxis_title="{feature_name}",
                  width=1500, height=500)
fig.write_image("Plot/Analysis/boxplot_{feature}_communities.png")
"""
    if SHOW_FIGURES:
      s += """
fig.show()
"""

    for acp_nb in range(1, 16):
        for type_data in type_data_list:

            s += f"""
#| Community type {acp_nb} - {type_data}

acp_nb = {acp_nb}
acp_name = county_database[county_database["acp"] == acp_nb]["acp_name"].iloc[0]
"""
            s += """
kwargs = {"filter_equality": "acp",
          "filter_equality_value": acp_nb,
          "additional_text": "<br>American Communities: " + acp_name,
          "shared_yaxes": "all"}
"""
            s += f"""
custom_scatter_plot(feat_col="{feature}",
                    feat_name="{feature_name}",
                    type_data="{type_data}",
                    min_val={min_val}, max_val={max_val},
                    cmid={cmid},
                    show_figures={SHOW_FIGURES},
                    **kwargs)
"""
    return s


for feature, feature_name, cmid, min_val, max_val in features_list:
    script += add_to_script_communities(feature, feature_name,
                                        cmid, min_val, max_val)

with open("script_to_convert.py", "w+") as f:
    f.write(script)
