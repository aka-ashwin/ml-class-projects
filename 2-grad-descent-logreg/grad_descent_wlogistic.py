__author__ = 'Ashwin'

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


max_iters = 100
gen_J_thresh = 0.
plotln = False
return_scaled_to_data = True


#update formula : w += alpha * (y-h) * (h(1-h)) * x
            #rather than linear regression update formula of w += alpha(y-h)x
    #note for logistic reg, h = 1/(1+exp(-w^T x))

#now returns coefficients scaled to input X
def gradient_descent_log(X, y, alpha, plot=True, preprocess = False, tellconvtime = False, threshin = 0.,maxitersin = 0):
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

    #set num steps threshold
    if(maxitersin == 0):
        maxsteps = max_iters
    else:
        maxsteps = maxitersin

    #initialize vectors
    theta = np.zeros(n+1)
    errs = np.zeros(m)
    d_theta = np.zeros(n+1)
    Js = []
    hs = np.zeros(m)
    hfacs = np.zeros(m)
    currh = 1.0
    currexponent = 0.0
    currpow = 1.0

    #loop through until convergence or max iteration number reached
    for iter in range(0,maxsteps):

        # print(theta)
        #generate errors y - h_theta(x) for each data point(row of X)
        for pt in range(0,m):
            currexponent = -1.0*(np.dot(theta[:n],XT[pt][:]) + theta[n])
            currpow = np.exp(currexponent)
            currh =  1./(1. + currpow)
            hs[pt] = currh
            hfacs[pt] = currh*(1.-currh)

        for pt in range(0,m):
            errs[pt] = y[pt] - hs[pt]

        #get update value for each predictor (column)
        for pred in range(0,n):
            currgrad = 0
            for pt in range(0,m):
                currgrad += errs[pt]*hfacs[pt]*XT[pt][pred]
            theta[pred] += alpha*currgrad

        #update constant term = theta[n]
        pred = n
        currgrad = 0
        for pt in range(0,m):
            currgrad += errs[pt]*hfacs[pt]*1.
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

    if(preprocess and return_scaled_to_data):
        theta_final = []
        avgweight = 0.
        for i in range(0,len(X)):
            col = X[i]
            colavg = np.average(col)
            colstd = np.std(col)
            theta_final.append(theta[i]/colstd)
            avgweight += colavg*theta[i]/colstd

        const_final = theta[n] - avgweight
        theta_final = [const_final] + theta_final
    else:
        theta_final = theta

    return theta_final




#note: assuming columns normalized
def gradient_descent_lin(X, y, alpha, plot=True, preprocess = False, tellconvtime = False, threshin = 0.,maxitersin = 0):
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

    #set num steps threshold
    if(maxitersin == 0):
        maxsteps = max_iters
    else:
        maxsteps = maxitersin

    #initialize vectors
    theta = np.zeros(n+1)
    errs = np.zeros(m)
    d_theta = np.zeros(n+1)
    Js = []



    #loop through until convergence or max iteration number reached
    for iter in range(0,maxsteps):

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
    if(preprocess and return_scaled_to_data):
        theta_final = []
        avgweight = 0.
        for i in range(0,len(X)):
            col = X[i]
            colavg = np.average(col)
            colstd = np.std(col)
            theta_final.append(theta[i]/colstd)
            avgweight += colavg*theta[i]/colstd

        const_final = theta[n] - avgweight
        theta_final = [const_final] + theta_final
    else:
        theta_final = theta

    return theta_final


def gradient_descent(X, y, alpha, plot=True, preprocess = False, tellconvtime = False, threshin = 0.,maxitersin = 0, log = True):
    if(log):
        return gradient_descent_log(X, y, alpha, plot, preprocess, tellconvtime, threshin, maxitersin)
    else:
        return gradient_descent_lin(X, y, alpha, plot, preprocess, tellconvtime, threshin, maxitersin)



def test_gradient_desc():
    Xt = [[1,1,1,1,2,2,4,4]]
    yt = [0,0,0,1,0,1,1,1]

    alphat = .02

    gradient_descent_log(Xt,yt,alphat)