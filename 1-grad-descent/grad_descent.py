__author__ = 'Ashwin'

#Implement a python function gradient descent(X, y, alpha) that takes an input matrix X of shape (m,n),
#  a output vector y of shape (m,1), a scalar learning rate alpha and returns a vector theta of shape (n+1,1).
#
# theta should be the linear regression parameter vector found by a gradient descent algorithm on the given inputs.
#
# The function should also plot the value of the cost function J(?) vs the iternation number.

import numpy as np
import matplotlib.pyplot as plt
import math

#process matrix X - normalize each column to mean-0 and variance-1 by replacing values xij with (xij - xbar_i)/sigma_i
def preppedX(X):
    ncols = len(X)
    means = []
    stds = []
    outmat = []


    for col in range(0,ncols):
        means.append(np.mean(X[col]))
        stds.append(np.std(X[col]))

    for col in range(0,ncols):
        outcol = []
        currmean = means[col]
        currstd = stds[col]
        for val in range(0,len(X[col])):
            outcol.append((X[col][val]-currmean)/currstd)

        outmat.append(outcol)

    return outmat


max_iters = 500
gen_J_thresh = .1
plotln = True

#note: assuming columns normalized
def gradient_descent(X, y, alpha, plot=True, preprocess = False, tellconvtime = False, threshin = 0.):
    #setup
    #note - slight inefficiency in creating a new matrix Xrl, rather than just taking XT = transpose(prepped(X))
    #since we can do all our lookups from XT
    #however, this way will make things a little easier/more legible down the line since we have a prepped X to use.
    if(preprocess):
        Xrl = preppedX(X)
    else:
        Xrl = X
    XT = np.transpose(Xrl)

    n = len(Xrl)
    m = len(Xrl[0])
    alpha_to_use = alpha*1.0/m

    #set J threshold
    if(threshin == 0):
        J_thresh = gen_J_thresh*m
    else:
        J_thresh = threshin*m

    #initialize vectors
    theta = np.zeros(n+1)
    errs = np.zeros(m)
    d_theta = np.zeros(n+1)
    Js = []



    #loop through until convergence or max iteration number reached
    for iter in range(0,max_iters):

        # print(theta)
        #generate errors y - h_theta(x) for each data point(row of X)
        for pt in range(0,m):
            errs[pt] = y[pt] - np.dot(theta[:n],XT[pt][:]) - theta[n]

        #get update value for each predictor (column)
        for pred in range(0,n):
            currgrad = 0
            for pt in range(0,m):
                currgrad += errs[pt]*XT[pt][pred]
            theta[pred] += alpha*currgrad

        #update constant term = theta[n]
        pred = n
        currgrad = 0
        for pt in range(0,m):
            currgrad += errs[pt]*1
        theta[pred] += alpha*currgrad

        #calculate goal function (= residual sum of squares) J(theta) for previous step:
        J = np.linalg.norm(errs)**2
        if(plotln):
            Js.append(math.log(J))
        else:
            Js.append(J)


        if(J <= J_thresh):
            break
    #plot J
    if(plot):
        Jstr = "J"
        if(plotln):
            Jstr = "ln(J)"
        plt.plot(Js)
        plt.xlabel("Gradient Descent Step")
        plt.ylabel(Jstr)
        plt.title("Gradient Descent with Alpha = " + str(alpha) + "; " + Jstr +" versus Iteration")
        plt.show()
    if(tellconvtime):
        print("finished in " +str(iter) + " rounds to reach a J/m of " + str(J/m) + ": threshold was " + str(J_thresh/m))
    return theta

def test_gradient_desc():
    Xt = [[1,2,3,4,5]]
    yt = [2,3,4,5,6]

    alphat = .02

    gradient_descent(Xt,yt,alphat)