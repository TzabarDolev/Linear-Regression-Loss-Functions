# Linear-Regression-Loss-Functions
Simple pyhon implementation of least squares and basic loss functions using basic linear regression

This file contains a simple implementation of Linear regression using Least Squares.
It also contains basic implementations of the following loss functions:
L1, L2, Huber, Log-Cosh.

The code is initilized with 60 points. 40 points are scattered randomly between 1 to 20 with a slope of 1 (x and y values are equal).
In addition, 20 points a scattered using Gaussian distribution with a mean=10 and sigma=3.

The model finds the least squares solution, followed by a calculation of the loss functions.

L1_loss=(1/n)*|y-y_pred|
L2_loss = ((1/n) * (y-y_pred)^2)
Huber_loss = where (|y - y_pred| < delta) -> 0.5 * (y - y_pred)^2, where: (|y - y_pred| >= delta) -> delta * |y - y_pred| - 0.5 * delta^2
LogCosh_loss = log(cosh(y-y_pred)

L1 is robust to outliers but it's derivatives are not continous.
L2 is difficult with outliers but it's gradients are naturally adaptive.
Huber loss combines the advantages of both L1 and L2.
Log Cosh is very similar to Huber loss but it is twice differentiable everywhere
