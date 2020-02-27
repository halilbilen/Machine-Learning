import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


myDataSet = pd.read_csv("hw_25000.csv",sep=",")

plt.scatter(myDataSet.Height,myDataSet.Weight)
plt.xlabel("weight")
plt.ylabel("height")
plt.title("hw_250000")
plt.show()