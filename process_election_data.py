import json
import pandas as pd

# Extract total dem & rep votes for each county by exploring all precints
with open("precincts-with-results.geojson", "r") as f:
    data = json.load(f)

result = {}
for _, v in enumerate(data["features"]):
    dic = v["properties"]
    county = dic["GEOID"].split("-")[0]
    if len(county) > 5:
        county = county[:5]
    county = str(int(county))
    votes_dem = dic["votes_dem"]
    votes_rep = dic["votes_rep"]
    # votes_total = dic["votes_total"]
    if votes_dem is None:
        votes_dem = 0
    if votes_rep is None:
        votes_rep = 0
    # if votes_total is None:
    #     votes_total = 0
    if county in result.keys():
        result[county]["votes_dem"] += votes_dem
        result[county]["votes_rep"] += votes_rep
        # result[county]["votes_total"] += votes_total
    else:
        result[county] = {"votes_dem": votes_dem,
                          "votes_rep": votes_rep}  # "votes_total": votes_total

# Add total population of each county to the dictionnary
# From American Community Survey - United States Census
df = pd.read_csv("American Community Survey - United States Census/cp05.csv", encoding="latin")
def transfo_county(id):
    res = id.split("US")[1]
    return str(int(res))
df["county_id"] = df["GEOID"].apply(lambda x: transfo_county(x))
for county in result.keys():
    df_county = df[df["county_id"] == county]
    if df_county.empty:  # No data
        total_pop = None
    else:
        total_pop = int(df_county[df_county["TITLE"] == "Total population"]["EST_1418"].iloc[0].replace(",", ""))
    result[county]["total_population"] = total_pop

# Calculate political leaning
for county in result.keys():
    if result[county]["total_population"] is None:
        result[county]["political_leaning"] = None
    else:
        result[county]["political_leaning"] = (result[county]["votes_rep"] - result[county]["votes_dem"]) / result[county]["total_population"]

# Save data
df_result = pd.DataFrame(result)
df_result.to_csv("political_leaning.csv")
