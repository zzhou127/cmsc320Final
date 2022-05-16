"""
ram and storage are in GB, weight is in pounds

cpu model num not found or is just a generation and/or i/r series are ignored
cpu series are intel celeron, intel pentium, amd Athlon, intels i3, i5, i7, ryzen r3, r5, r7.
Amd calls it's ryzen series as ryzen 3/5/7, this porjects calls it r3/5/7
generation are a number like the intel 12th gen or ryzen 5000 series.
issues with giving single and multi core the same weight.

gpu series are intel HD Graphics, Intel Iris Xe Graphics, nvidia MX series, nvidias gtx and rtx series, AMD vega series,
Intel calls integreaed gpus igpu, AMD calls cpu with gpu as APU, this project will call them igpus because speficily discussing about the igpu's preformance and not the cpu.
Nvidia quadro and A series exists, very specialized workloads, buyer either knows and needs it or will never know about it
"""
import re
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

bestbuy = pd.read_csv("./bestbuy.csv")
cpuBenchmark = pd.read_csv("./cpuBenchmark.csv")
gpuBenchmark = pd.read_csv("./gpuBenchmark.csv")
cpuNumPat = re.compile(r"(\w{0,2}\d{4,5}((HK|HX|HQ)|(\w?\d?)))")
#filter cpus, MT = MediaTek
cpuFilterPat = re.compile(r"(Apple.*)|(Not provided)|(MT)", flags=re.I)
getIntPat = re.compile(r"(\d+)")
getFloatPat = re.compile(r"(\d+(\.\d+)?)")

# cleaning up bestbuy
#parsing cpu model number
def parseCpuNum(df, col):
  series = df[col].str.extract(cpuNumPat)
  #only want the 1st group
  series = series[0]
  series[series.isnull()] = df[col][series.isnull()]
  df["cpuNum"] = series
parseCpuNum(bestbuy, "cpuNum")
#filter cpus
bestbuy = bestbuy[bestbuy.cpuNum.str.match(cpuFilterPat) == False]

#parsing storage, ram, weight values by stripping units and extracting int val
def parseUnits(bestbuy, col, type1):
  if type1 == int:
    series = bestbuy[col].str.extract(getIntPat, expand=False)
    #only want the 1st group
  elif type1 == float:
    series = bestbuy[col].str.extract(getFloatPat)
    series = series[0]
  bestbuy[col] = series.astype(type1)
parseUnits(bestbuy, "ram", int)
parseUnits(bestbuy, "storage", int)
parseUnits(bestbuy, "weight", float)

#all unqiue names
# set(bestbuy.cpu)
labels = ("celeron", "pentium", "Athlon", "i3", "i5", "i7", "i9", "r3", "r5", "r7", "r9")
labelColors = ("brown","orange","yellow","green","blue","purple","red","black","grey","cyan","magenta")
labelColors = dict((k,v) for k,v in zip(labels,labelColors))
def createSeries(bestbuy):
  name = "cpuSeries"
  cpuSeries = ("celeron", "pentium", "Athlon", "i3", "i5", "i7", "i9", "ryzen 3", "ryzen 5", "ryzen 7", "ryzen 9")
  bestbuy[name] = np.nan
  
  for series, label in zip(cpuSeries, labels):
    bestbuy.loc[bestbuy.cpu.str.match(rf"(.*{series}.*)", flags=re.I) == True, name] = label
createSeries(bestbuy)

#cleaning up cpuBenchmark
parseCpuNum(cpuBenchmark, "cpu")
#create aggeragate preformace index
#normalize each numerical col so values are between [0, 1]
def createPrefIdx(df):
  pref = pd.Series([0 for _ in range(len(df.index))], dtype=float)
  i = 0
  for colName, colData in df.items():
    if colData.dtype == int or colData.dtype == float:
      i += 1
      currPref = colData / colData.max()
      #if missing data add current pref / i so it treats it as if nothing changed
      #ex, curr pref for this cpu is 2 from 2 dif benchmarks, 3rd benchmark is mssing. add 2 / 2 = 1 to curr pref, which is now 3. dividing by number of benchmarks, 3 / 3 = 1, which is the same average as 2 / 2 = 1.
      currPref[currPref.isnull()] = pref[currPref.isnull()] / i
      pref += currPref
  df["pref"] = pref
createPrefIdx(cpuBenchmark)

def linkCpuPref(bestbuy, benchmark):
  name = "cpuPref"
  bestbuy[name] = np.nan
  cpuFound = benchmark.loc[benchmark.cpuNum.isin(bestbuy.cpuNum)]

  for i, cpu in cpuFound.iterrows():
    bestbuy.loc[bestbuy.cpuNum == cpu.cpuNum, name] = cpu.pref
linkCpuPref(bestbuy, cpuBenchmark)
bestbuy = bestbuy[bestbuy.cpuPref.isnull() == False]

# cpuBySeries = bestbuy.groupby("cpuSeries")
# for series, group in cpuBySeries:
#   plt.scatter(group.price, group.cpuPref, label=series, c=labelColors[series])
# #graph labels
# plt.xlabel("price")
# plt.ylabel("cpu preformance")
# plt.title(f"price to cpu preformance")
# plt.ylim(0)
# plt.xlim(0)
# plt.legend(loc="upper left")
# plt.show()

# celeron = cpuBySeries.get_group("celeron")
# for os, group in celeron.groupby("os"):
#   if os == "Chrome OS":
#     plt.scatter(group.price, group.cpuPref, label=os, c="red")
#   else:
#     plt.scatter(group.price, group.cpuPref, label="Windows", c="blue")
# plt.xlabel("price")
# plt.ylabel("cpu preformance")
# plt.title(f"price to cpu preformance")
# plt.ylim(0)
# plt.xlim(0)
# plt.legend(loc="upper left")
# plt.show()

gpuSeries = (
  r"Intel (u?hd) Graphics [^Xe]", r"Intel Iris (xe) Graphics", r"NVIDIA GeForce (MX\d{3})",
  r"NVIDIA (GeForce [GR]TX \d{3,4}M?(( Super)|( Ti)|( Max-Q))*)",
  r"NVIDIA (Quadro ([GR]TX)? [a-z]?\d{3,4}[a-z]?( Max-Q)?)",
  r"(NVIDIA RTX A\d000)",
  r"(Vega \d{1,2})"
)
def createSeries(df, labels):
  name = "gpuSeries"
  df[name] = np.nan
  for label in labels:
    match = df.gpu.str.extract(rf".*{label}.*", flags=re.I, expand=False)
    if isinstance(match, pd.DataFrame):
      match = match[0]
    boolList = (match.isnull() == False)
    df.loc[boolList, name] = match[boolList]
  
  return df
createSeries(bestbuy, gpuSeries)
createSeries(gpuBenchmark, gpuSeries)

print(set(bestbuy.loc[bestbuy.gpu.str.match(r".*(Vega \d{1,2}).*", flags=re.I) == True].cpu))
#there are only 3000 series and microsoft edition, AMD Vega pref changes between cpu generations, using 3000 series vega since that is the only generation known.
gpuBenchmark.loc[gpuBenchmark.gpu.str.match(r".*Vega \d{1,2}.*Ryzen [45]000.*") == True, "gpuSeries"] = np.nan

createPrefIdx(gpuBenchmark)
#add gpu and overall(cpu + gpu) preformance
def linkPref(bestbuy, gpuBenchmark):
  name = "gpuPref"
  name2 = "pref"
  bestbuy[name] = np.nan
  bestbuy[name2] = bestbuy.cpuPref
  gpuFound = gpuBenchmark.loc[
    gpuBenchmark.gpuSeries.isin(
      bestbuy[bestbuy.gpuSeries.isnull() == False].gpuSeries
    )
  ]
  for i, gpu in gpuFound.iterrows():
    boolList = (bestbuy.gpuSeries == gpu.gpuSeries)
    bestbuy.loc[boolList, name] = gpu.pref
    bestbuy.loc[boolList, name2] += gpu.pref
linkPref(bestbuy, gpuBenchmark)

gpuBySeries = gpuBenchmark.groupby("gpuSeries")
for series in ("Xe", "HD", "UHD"):
  group = gpuBySeries.get_group(series)
  gpuBenchmark.loc[group.index, "pref"] = group.pref.mean()
bestbuy = bestbuy[bestbuy.gpuSeries.isnull() == False]

cpuBySeries = bestbuy.groupby("cpuSeries")
for series, group in cpuBySeries:
  plt.scatter(group.price, group.pref, label=series, c=labelColors[series])
#graph labels
plt.xlabel("price")
plt.ylabel("overall preformance")
plt.title(f"price to overall preformance")
plt.ylim(0)
plt.xlim(0)
plt.legend(loc="upper left")
plt.show()

linReg = LinearRegression()
linReg.fit(np.array(bestbuy.price).reshape(-1,1), bestbuy.pref)
prices = np.linspace(0, bestbuy.price.max(), 100)

for series, group in cpuBySeries:
  plt.scatter(group.price, group.pref, label=series, c=labelColors[series])

plt.plot(prices, linReg.predict(prices.reshape(-1,1)))
#graph labels
plt.xlabel("price")
plt.ylabel("overall preformance")
plt.title(f"price to overall preformance")
plt.ylim(0)
plt.xlim(0)
plt.legend(loc="upper left")
plt.show()