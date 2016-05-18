__author__ = 'Ashwin'


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model as sklin

import tabulate as tbl

import grad_descent as gd

#load data
df_full = pd.read_csv("prud_hw1.csv")


#1. replace missing values with means

#include blanks, i.e. "", as nas:
df_full.replace("", np.nan)

for key in df_full.keys():
    try:
        currmean = df_full[key].mean()
        df_full[key].fillna(currmean, inplace = True)
        # print("mean for " + key + " was: " + str(currmean))
    except:
        print("couldn't get mean for: " + key)
        continue
print('done!')

#2. create smaller dataframe for our simple analysis
df_small = df_full[['Ht','Wt','Response']][:1000]

#write dataframe for ease of use/reference
df_small.to_csv("prud_hw1_small.csv")
#]

#read df_small from file - used during actual data analysis for improved runtime
# df_small = pd.read_csv("prud_hw1_small.csv")

#gradient descent method

#cast values to matrices
X = [df_small['Ht'].tolist(),df_small['Wt'].tolist()]
# print(X[:10])
y = df_small['Response'].tolist()


#use gradient descent to get coefficients

#compare alphas, make graphs
show_all_coeffs = True
print_convergence_details = True
show_plot = True
opt_threshold = 0 #so that all graphs show equal number of iterations, no threshold used

test_alphas_old = [.0001,.001, .01]
test_alphas_next = [.0007,.0009,.0011,.0012,.0013]
test_alphas_close = [.00092,.00095,.00098,.00101]
test_alphas = [.00098,.00099,.000995,.001]

test_alphas_illustrative = [.0001, .0007, .001, .00105, .0013, .01]

alpha_final = .001

for alpha in test_alphas_illustrative:
    print("Alpha = " + str(alpha))
    t = gd.gradient_descent(X,y,alpha,show_plot,True,print_convergence_details, opt_threshold)
    if(show_all_coeffs):
        print("My Coeffs for scaled X (alpha = " + str(alpha) + "): " + str(t[:2]))
        print("My Constant for scaled + normalized X (alpha = " + str(alpha) + "): " + str(t[2]))
        print("---\n")


#use best alpha and compare to sklearn regression results

#use gradient descent for best alpha
alpha = alpha_final
show_final_plot = False
print_final_convergence = False
final_threshold = 5.381845807175 #a threshold of slightly greater than the optimal J-value found is used, so that algorithm doesn't iterate more than needed
t = gd.gradient_descent(X,y,alpha,show_final_plot,True,print_final_convergence)

#compare to sklearn regression results:
#(and compare results scaled to original data to sklearn results)
#learned sklearn syntax (and copied some) from http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#example-linear-model-plot-ols-py

#create X-transposes for sklearn fits
XT = np.transpose(gd.preppedX(X))
XT2 = np.transpose(X)

#sklearn fit on scaled/centered data
reg = sklin.LinearRegression()
reg.fit(XT,y)

#my fit rescaled t original (unscaled, uncentered) data
htsc = np.std(X[0])
wtsc = np.std(X[1])
my_unscaled_const =  str(t[2] - np.average(X[0])*t[0]/htsc - np.average(X[1])*t[1]/wtsc)
t_descaled = [t[0]/htsc, t[1]/wtsc, my_unscaled_const]

#sklearn fit on original data
reg2 = sklin.LinearRegression(fit_intercept = True)
reg2.fit(XT2,y)

skl1 = [reg.coef_[0],reg.coef_[1],reg.intercept_]
skl2 = [reg2.coef_[0],reg2.coef_[1],reg2.intercept_]
labels = ["Source", "Height Coeff", "Weight Coeff", "Constant"]
sources = [["Gradient Descent (scaled + centered vars)"], ["Sklearn (scaled + centered vars)"], ["Gradient Descent (scaled to original vars)"], ["Sklearn (for original vars)"]]

table = [sources[0]+np.ndarray.tolist(t), sources[1]+skl1, sources[2] + t_descaled, sources[3] + skl2 ]

#print summary table
print("Results for gradient descent with alpha = " + str(alpha_final)+":\n")
print(tbl.tabulate(table,headers =labels))