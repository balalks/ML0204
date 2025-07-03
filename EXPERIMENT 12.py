import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25, 36, 49])

# Linear Regression
lin_model = LinearRegression()
lin_model.fit(x, y)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)
poly_model = LinearRegression()
poly_model.fit(x_poly, y)

plt.scatter(x, y, color='blue')
plt.plot(x, lin_model.predict(x), color='red', label='Linear')
plt.plot(x, poly_model.predict(x_poly), color='green', label='Polynomial')
plt.legend()
plt.title("Linear vs Polynomial Regression")
plt.show()
