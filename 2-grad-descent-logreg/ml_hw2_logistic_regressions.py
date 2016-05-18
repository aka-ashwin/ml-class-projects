from __future__ import division # ensures that default division is real number division #taken from helper code
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

import grad_descent_wlogistic as gdl

#taken from helper code:
mpl.rc('figure', figsize=[10,6])

df = pd.read_csv('wdbc.data', header=None)
base_names = ['radius', 'texture', 'perimeter', 'area', 'smooth', 'compact', 'concav',
                 'conpoints', 'symmetry', 'fracdim']
names = ['m' + name for name in base_names]
names += ['s' + name for name in base_names]
names += ['e' + name for name in base_names]
names = ['id', 'class'] + names
df.columns = names
df['color'] = pd.Series([(0 if x == 'M' else 1) for x in df['class']])
my_color_map = mpl.colors.ListedColormap(['r', 'g'], 'mycolormap')

#end of 1st block of code taken from helper code

all_input_names = names[2:]
output_list = df['color'].tolist()

#test alphas over optionally-smaller portion of data set

trainlen = 569

numvarstest = 30
trainnames = []
for i in range(0,numvarstest):
    currname = all_input_names[i]
    trainnames.append(currname)

X = []
X2 = []

for name in trainnames:
    X.append(df[name].tolist()[:trainlen])
    X2.append(df[name].tolist()[trainlen:])


y = output_list[:trainlen]
y2 = output_list[trainlen:]

#test over different alphas
#originally used .00001,.0001,.001 but found that .01 converged equally well and more quickly
#.1 works well so far, but I am cautious about it because I know it diverged for hw 1
#.5 and .8 are oddly good, but the coefficients they yield are very high; I think I will stick to smaller coefficients
#using .01, as it converges quickly but with reasonably-sized coefficients

#note: the large-magnitude coefficients at higher learning rates may be due to collinearity of the variables,
# which makes the individual coefficient values much less illustrative than the overall sum of coefficients for each set of highly-correlated inputs

# alphas = [.0001,.001,.01,.1,.5,.8]
alphas = []
show_plot = True
print_convergence_details = True
opt_threshold = 0
maxiters = 1000
for alpha in alphas:
    print("alpha = " + str(alpha) + ": \n")
    t = gdl.gradient_descent(X,y,alpha,show_plot,True,print_convergence_details, opt_threshold, maxiters, True)
    print("\n\n t: " + str(t))


#test final alpha at different classification thresholds
alpha_finals = [.01]
test_thresholds = [.1,.3,.45,.5,.55,.7,.9]
use_separate_test_data = False

for alpha_final in alpha_finals:
    print("\n \n alpha = :" + str(alpha_final))
    t_final = gdl.gradient_descent(X,y,alpha_final,show_plot,True,print_convergence_details, opt_threshold, maxiters, True)

    print("final theta, scaled for original variables:\n")
    print(t_final[1:])
    print("\n final constant: " + str(t_final[0]))


    # test accuracy of final model
    goodcount = 0

    if(use_separate_test_data):
        Xtotest = X2
        ytotest = y2
        numtestrows = 569-trainlen
    else:
        Xtotest = X
        ytotest = y
        numtestrows = trainlen

    currpred = 0
    currpredval = 0
    for defthresh in test_thresholds:
        goodcount = 0.
        XT = np.transpose(np.asarray(Xtotest))
        for i in range(0,numtestrows):
            currpredval = np.dot(t_final[1:],XT[i]) + t_final[0]
            currpredval = 1./(1. + np.exp(-currpredval))
            if(currpredval >= defthresh):
                currpred = 1
            else:
                currpred = 0
            # print("prediction: " + str(currpred))
            # print("reality: " + str(y[i]))
            if(currpred == ytotest[i]):
                goodcount += 1.

        print("threshold = " + str(defthresh) + "; accuracy ratio: " + str(goodcount/numtestrows))


clf = LogisticRegression(C=100.)
clf.fit(df[trainnames][:trainlen], df['color'][:trainlen])

print("\n sklearn coeffs: " + str(clf.coef_) + "; intercept = " + str(clf.intercept_))

#
# clf_J = 0
# for i in range(0,10):
#     print(df[trainnames][i])
#     currpred = clf.predict(df[trainnames][i])
#     # print(currpred)
#     print(" predicted; ")
#     print(output_list[i])
#     # clf_J += (currpred - output_list[i])**2

clf_J = clf.score(df[trainnames][:],df['color'][:])

print("Accuracy for sklearn regression: " + str(clf_J))

#Problem 8 - run logistic regression on 'mradius' and 'mtexture' but with degree = 3; graph points and decision boundary


#below adapted from helper code
c1 = 'mradius'
c2 = 'mtexture'

plt.scatter(df[c1], df[c2], c = df['color'], cmap=my_color_map)
plt.xlabel(c1)
plt.ylabel(c2)

x = np.linspace(df[c1].min(), df[c1].max(), 1000)
y = np.linspace(df[c2].min(), df[c2].max(), 1000)
xx, yy = np.meshgrid(x,y)


topred = np.hstack(
            (xx.ravel().reshape(-1,1), yy.ravel().reshape(-1,1))
        )

polyFinal = PolynomialFeatures(degree=3,include_bias = False)
topred3 = polyFinal.fit_transform(topred)

clfFinal = LogisticRegression()
clfFinal.fit(polyFinal.fit_transform(df[[c1,c2]]),df['color'])

predictions3 = clfFinal.predict(topred3)

predictions3 = predictions3.reshape(xx.shape)

plt.contour(xx, yy, predictions3, [0.0])
plt.show()