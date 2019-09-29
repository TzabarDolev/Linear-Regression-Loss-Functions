import numpy as np
import matplotlib.pyplot as plt
from random import gauss, randint

def main():
    mu = 10
    sigma = 3
    num_points = 20
    X = [0] * num_points*3
    Y = [0] * num_points*3
    print('Gaussian distribution: %d points, mu=%d, sigma=%d' % (num_points, mu, sigma))
    for val in range(0, num_points):
        X[val] = gauss(mu, sigma)
        Y[val] = gauss(mu, sigma)

    for val in range(num_points, 3*num_points):
        X[val] = randint(round(mu * 0.1), round(mu*2))
        Y[val] = X[val]
    print('Linear scattering with slope=1. %d points' % (num_points*2))
    LQ_model(X, Y, num_points)

def LQ_model(x, y, num_points):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    m_num = 0
    m_den = 0
    for i in range(len(x)):
        m_num += (x[i] - x_mean)*(y[i] - y_mean)
        m_den += (x[i] - x_mean)**2
    m = m_num / m_den
    b = y_mean - m*x_mean
    predictions(m, b, x, y, num_points)
    return

def predictions(m, b, x, y, num_points):
    delta = 1
    y_pred = np.dot(m, x) + b
    plt.figure(0)
    plt.scatter(x, y) # actual
    plt.plot([min(x), max(x)], [min(y_pred), max(y_pred)], color='red')
    plt.plot([min(x), max(x)], [min(y), max(y)], color='blue')
    plt.show()
    error = np.sum(np.square((y-y_pred)))
    print('least squares error: ' + str(error))
    loss_functions(y, y_pred, num_points, delta)

def loss_functions(y, y_pred, num_points, delta):
    # printing loss functions: L1, L2, Huber
    L1_loss=(1/(num_points*3))*np.sum(abs(y-y_pred))
    print('L1 loss: ' + str(L1_loss))

    L2_loss = (1/(num_points*3)) * np.sum(np.square(y-y_pred))
    print('L2 loss: ' + str(L2_loss))

    Huber_loss = np.sum(np.where(np.abs(y - y_pred) < delta, 0.5 * (np.square(y - y_pred)),
                    delta * np.abs(y - y_pred) - 0.5 * (delta ** 2)))
    print('Huber loss: ' + str(Huber_loss) + ', delta = ' + str(delta))

    LogCosh_loss = np.sum(np.log(np.cosh(y - y_pred)))
    print('Log Cosh loss: ' + str(LogCosh_loss))

    return

if __name__ == "__main__":
    main()
