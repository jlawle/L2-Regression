"""
 ECE 8550 Assignment 1 - Linear Regression
 John Lawler
 Feb 10 2022

 Purpose of assignment is to implement a linear regression model to predict death rate amongst several
 characteristics (population, pollutions, temperature, etc).
"""

import numpy as np
import matplotlib.pyplot as plt

# weights input is a vector 16 long
# xi input is vector 16 long of all features at a given sample
def predict(w, xi):
    sum1 = 0
    for j in range(len(w)):
        sum1 += w[j] * xi[j]
    return sum1

# calculate_loss(48, 16, weights, lambda_r, ynorms, norms)
def calculate_loss_l(n, d, w, l, y, x):
    # get summation of prediction function
    sum1 = 0
    for i in range(n):
        f_x = predict(w, x[i])
        sum1 += np.square(y[i] - f_x)

    # get summation of weights
    sum2 = 0
    for i in range(d):
        sum2 += abs(w[i])

    l_const = l/(2*d)
    n_const = 1/(2*n)
    loss = n_const*sum1 + l_const*sum2
    return loss

# pd_loss(48, 16, weights, lambda_l, ynorms, norms, feature # of concern (jth feature))
def pd_loss(n, d, w, l, y, x, j):
    sum1 = 0
    for i in range(n):
        f_x = predict(w, x[i])
        sum1 += (y[i] - f_x)*(-1*x[i][j])
    if w[j] == 0:
        pdw = 1
    else:
        pdw = w[j] / (np.sqrt(np.square(w[j])))
    pdloss = (1/n)*sum1 + l/(2*(d+1))*pdw
    return pdloss

# convergence criteria for batch gradient descent, where conv < e
def converge(prevJ, currJ):
    subtract = (prevJ - currJ) * 100
    sq = np.sqrt(np.square(subtract))
    nom = sq/prevJ
    return nom

# Calculate mean squared error versus testing set of data
def MSE(m, ytest, w, xtest):
    sum1=0
    for i in range(m):
        f_x = predict(w, xtest[i])
        sum1 += np.square(ytest[i] - f_x)
    return (1/(2*m))*sum1

# Read in data
data = []
lambda_l = 0.3
alpha = 0.01
eps = 0.001

# Open file
with open("data.txt") as file:
    lines = [line.rstrip() for line in file]

# Set the data to the array
for line in lines[19:]:
    if line[0] == ' ':
        line = line[1:]
    linedata = [float(i) for i in line.split(' ')[1:] if i != ' ' and i != '']
    data.append(linedata)

# *************************************************************************************
# Normalize data
data = np.array(data)
datanorm = []

# Iterate data across each col, calcuate norms per feature
for i in range(len(data[:,0])):
    featnorm = []
    if i == 16:
        break
    min = np.min(data[:,i])
    max = np.max(data[:,i])
    # Range over each X
    for x in data[:,i]:
        featnorm.append((x - min) / (max - min))
    datanorm.append(featnorm)

## seperate out norms nicely
norms = []
ynorms = []
for j in range(len(datanorm[0])):
    temp = []
    for i in range(len(datanorm)):
        temp.append(datanorm[i][j])
    ynorms.append(temp[-1])
    temp[-1] = 1
    norms.append(temp)

# *************************************************************************************
# Perform Batch Gradient Descent
weights = np.zeros(16)
Jkw = []
mse = []
k = 0
delta = 0
train_norms = norms[0:48]
test_norms = norms[48:]
train_norms_y = ynorms[0:48]
test_norms_y = ynorms[48:]


not_converged = True # reached when delta >= e
while not_converged:
    # iterate j = 1 ...
    new_weights = []
    for j in range(len(weights)):
        # determine pd of wj
        new_weight = weights[j] - (alpha * pd_loss(48, 16, weights, lambda_l,  train_norms_y, train_norms, j))
        new_weights.append(new_weight)

    # get loss
    Jk = calculate_loss_l(48, 16, weights, lambda_l, train_norms_y, train_norms)
    Jkw.append(Jk)

    # Convergence test
    if k != 0:
        delta = converge(Jkw[k-1], Jkw[k])
        if delta < eps:
            not_converged = False

    # Calculate mse against test data
    msek = MSE(12, test_norms_y, weights, test_norms)
    mse.append(msek)

    # Reset weights
    weights = new_weights
    k = k + 1

print("Final loss: ", Jk)
print("Final mse: ", msek)

# Print out weights less than 0.01
for i in range(len(weights)):
    if weights[i] < 0.01:
        print(weights[i])


plt.figure()
plt.plot(range(k), Jkw, label="Loss")
plt.plot(range(k), mse, label="MSE")
plt.xlabel("Iterations (k)")
plt.legend()
plt.title("Gradient Descent with Lasso Regularization")

plt.show()





