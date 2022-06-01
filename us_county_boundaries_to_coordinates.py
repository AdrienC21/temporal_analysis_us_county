import pandas as pd
import os

os.chdir(r"C:\Users\acarr\Documents\MIT Internship\American Communities")

df = pd.read_csv("us-county-boundaries.csv", sep=";", encoding="latin")
df = df[["GEOID", "Geo Point"]]
df["GEOID"] = df["GEOID"].apply(lambda x: str(int(x)))
df["Lat"] = df["Geo Point"].apply(lambda x: float(x.split(",")[1]))
df["Long_"] = df["Geo Point"].apply(lambda x: float(x.split(",")[0]))
df.drop(columns=["Geo Point"], inplace=True)
df.to_csv("us_county_coordinates.csv")
