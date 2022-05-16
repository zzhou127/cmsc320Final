'''
specs of laptops are poorly labelled, indicats consumers either don't know or care of hardware. Compared to custom pc like origin, specs are the 1st and most important thing for the customers.
https://www.notebookcheck.net/Mobile-Processors-Benchmark-List.2436.0.html?type=&sort=&or=0&cinebench_r20_single=1&cinebench_r20_multi=1&cinebench_r23_single=1&cinebench_r23_multi=1&cpu_fullname=1
qhqws47nyvgze2mq3qx4jadt
7Y75, 7Y57
486, typo processor
'''
import re
from time import sleep
import requests, json, pandas as pd, numpy as np

#api key is from a public forum
params = {
  "apiKey": "qhqws47nyvgze2mq3qx4jadt",
  "sort": "manufacturer.asc",
  "show": "regularPrice,salePrice,shippingCost,customerReviewAverage,customerReviewCount,details.name,details.value,weight",
  # "showOptions": "&show=manufacturer,customerReviewAverage,customerReviewCount,name,description,details.name,details.value,features.feature,longDescription,releaseDate,weight",
  "format": "json",
  "pageSize": "100",
  "cursorMark": "*"
}
json1 = {"nextCursorMark": "*"}
# data = pd.DataFrame(columns=("cpu","cpuNum","ram","storage","gpu","screnSize","Resolution","os","brand"))
# detailsPat = re.compile(r"Screen Size|Screen Resolution|brand|System Memory \(RAM\)|Total Storage Capacity|Processor Model|Processor Model Number|^Graphics$|Operating System", flags=re.I)
columns = ("price","msrp","cpu","cpuNum","ram","storage","gpu","screnSize","Resolution","os","brand","name","weight","rating","ratingLen")
labels = ("cpu","cpuNum","ram","storage","gpu","screnSize","Resolution","os","brand","name","name","weight",)
details = ("Processor Model","Processor Model Number","System Memory (RAM)","Total Storage Capacity","Graphics","Screen Size","Screen Resolution","Operating System","Brand","Model Number","Product Name","Product Weight")
products = []

def parse(product):
  df = pd.DataFrame(product["details"])
  name = df.name
  dict1 = {}

  if product["shippingCost"] == "":
    product["shippingCost"] = 0
  dict1["price"] = int(product["salePrice"]) + int(product["shippingCost"])
  dict1["msrp"] = product["regularPrice"]
  dict1["rating"] = product["customerReviewAverage"]
  dict1["ratingLen"] = product["customerReviewCount"]
  #There are 3 types of storage, emmc, ssd, hdd
  for label, detail in zip(labels, details):
    val = df[name == detail].value
    if len(val) == 0:
      dict1[label] = np.nan
    else:
      dict1[label] = val.iloc[0]
  return dict1

#bestbuy api only allows 100 results per page, and there can be 100+ products.
#cursorMark tells bestbuy API to return the next n results, and this is done untill there are no more products left.
while 1:
  params["cursorMark"] = json1["nextCursorMark"]
  res = requests.get("https://api.bestbuy.com/v1/products((categoryPath.id=abcat0502000))", params=params)
  #contains metadata, and the data to analyze which is "products"
  json1 = json.loads(res.text)
  if json1.get("nextCursorMark") == None:
    break
  products.extend(json1["products"])
  #prevent getting timed out
  sleep(1)

data = pd.DataFrame(list(map(parse, products)), columns=columns)
# data = pd.concat((data, ), ignore_index=True)
data.to_csv("./bestbuy.csv", index=False)

tablePat = re.compile(r"<table.*</table>", flags=re.M|re.S)
rowPat = re.compile(r"<tr.*?</tr>", flags=re.M|re.S|re.I)
cellPat = re.compile(r"<td.*?</td>", flags=re.S|re.I)
namePat = re.compile(r"<a.*>(.+)</a>")
valPat = re.compile(r"<span.*?(\d+(\.\d+)?)</span>")
columns = ("cpu","r20Single","r20Multi","r23Single","r23Multi")
res = requests.get("https://www.notebookcheck.net/Mobile-Processors-Benchmark-List.2436.0.html?type=&sort=&or=0&cinebench_r20_single=1&cinebench_r20_multi=1&cinebench_r23_single=1&cinebench_r23_multi=1&cpu_fullname=1")

def ParseNotebookcheck(html, columns):
  Benchmarks = []
  table = tablePat.search(html).group(0)
  rows = rowPat.findall(table)
  for row in rows:
    cells = cellPat.findall(row)
    dict1 = {}
    #skip header rows and cpu/gpu with no name
    name = namePat.search(cells[1])
    if valPat.search(cells[0]) == None or name == None:
      continue
    dict1[columns[0]] = name.group(1)
    for cell, col in zip(cells[2:], columns[1:]):
      val = valPat.search(cell)
      if val == None:
        dict1[col] = np.nan
      else:
        dict1[col] = val.group(1)
    #check if there is atleast 1 data exists
    for col in columns[1:]:
      if isinstance(dict1[col], str):
        Benchmarks.append(dict1)
        break
  return Benchmarks

data = pd.DataFrame(ParseNotebookcheck(res.text, columns), columns=columns)
data.to_csv("cpuBenchmark.csv", index=False)

columns = ("gpu","FireStrike","TimeSpy")
res = requests.get("https://www.notebookcheck.net/Mobile-Graphics-Cards-Benchmark-List.844.0.html?type=&sort=&multiplegpus=1&archive=1&or=0&3dmark13_fire_gpu=1&3dmark13_time_spy_gpu=1&gpu_fullname=1")
data = pd.DataFrame(ParseNotebookcheck(res.text, columns), columns=columns)
data.to_csv("gpuBenchmark.csv", index=False)