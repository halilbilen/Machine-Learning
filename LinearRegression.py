import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("hw_25000.csv",sep=",")


height=data.Height.values.reshape(-1,1)

weight=data.Weight.values.reshape(-1,1)

regression = LinearRegression()
regression.fit(height,weight)

print(regression.predict([[60]])) #bias
print(regression.predict([[62]]))
print(regression.predict([[64]]))
print(regression.predict([[66]]))
print(regression.predict([[68]]))
print(regression.predict([[70]]))

print("Intercept:",regression.intercept_)
print("Coef:",regression.coef_)

plt.scatter(data.Height,data.Weight)
#Gorsellestirme
x = np.arange(min(data.Height),max(data.Height)).reshape(-1,1)
plt.plot(x,regression.predict(x),color="red")

plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("hw_25000")
plt.show()



