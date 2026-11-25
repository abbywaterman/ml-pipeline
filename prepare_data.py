
import io
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd

url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

response = urllib.request.urlopen(url)
outer_zip = zipfile.ZipFile(io.BytesIO(response.read()))

inner_zip = zipfile.ZipFile(
    io.BytesIO(outer_zip.read("bank.zip"))
)

df = pd.read_csv(
    inner_zip.open("bank.csv"),
    sep=";"
)

# Remove information unavailable before the sales call
df = df.drop(columns=["duration"])

Path("data").mkdir(exist_ok=True)

df.to_csv("data/bank.csv", index=False)

print("Dataset saved successfully!")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
print(df["y"].value_counts())
