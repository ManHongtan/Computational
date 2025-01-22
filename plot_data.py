import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl

#read dataset
df=pd.read_csv("C:/Users/ASUS/Downloads/SIF3012history.data.txt",skiprows=range(0,5),sep='\s+')
print(df)

age=df["star_age"]
Teff=df["log_Teff"]
Luminosity=df["log_L"]

mpl.xlabel("log T_$eff$")
mpl.ylabel("log L$_{Luminosity}$")
mpl.plot(Teff,Luminosity)
mpl.invert_xaxis()
mpl.show()