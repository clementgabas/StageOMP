import statsmodels.api as sm
import statsmodels.formula.api as sm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from python.objets.wobjet import WDegCarre
from python.objets.lentille import LentilleGravita
lentilles_list = pickle.load(open("./source_pickle/lentilles_list2", "rb"))
obj_list = pickle.load(open("./source_pickle/fields", "rb"))


Y = [lent.rad for lent in lentilles_list]
X = [[lent.z, lent.seeing, lent.exposition] for lent in lentilles_list]

D = [[lent.rad, lent.z, lent.seeing, lent.exposition] for lent in lentilles_list]
import pandas as pd
import numpy as np
df = pd.DataFrame(np.array(D), columns=["rad", "z", "see", "expo"])


model = sm.ols(formula = "rad ~ z + expo + z*expo", data=df, missing='drop')
results = model.fit()
print(results.summary())

file_lentilles2 = "./source/Lentilles.csv"
data_lentilles2 = pd.read_csv(file_lentilles2, header=0, sep=";", skiprows=[1])
model = sm.ols(formula="Rad ~ zph + gmag + rmag*imag + e_zph",
               data=data_lentilles2, missing='drop')
model = sm.ols(formula="Rad ~ .", data=data_lentilles2, missing='drop')

results = model.fit()
print(results.summary())
model2 = sm.ols(formula="Rad ~ imag*rmag + RA", data=data_lentilles2, missing='drop')
results2 = model2.fit()
print(results2.summary())

c = data_lentilles2.corr()
sns.heatmap(c, annot=True)
plt.show()

plt.scatter(data_lentilles2["Rad"], data_lentilles2["zph"])
plt.show()
plt.scatter(df["see"], df["expo"])