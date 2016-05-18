__author__ = 'Ashwin'


import numpy as np

#N = number of possible response values
def qwk(y_gold,y_predict,N=8, quant_predictions = True, stuff_predictions = True, use_offset = True, fixed_offset = 1):
    n_entries = len(y_gold)

    if(quant_predictions):
        #round predictions to the nearest integer
        for i in range(0,n_entries):
            y_predict[i] = round(y_predict[i])
            #while we're at it, convert true values to integers as well
            #using round to avoid floats stored as eg 6.999999999996 getting cast to 6
            y_gold[i] = round(y_gold[i])

    goldmax = max(y_gold)
    goldmin = min(y_gold)
    if(stuff_predictions):
        #push in extreme predictions to extreme values of target data
        for i in range(0,n_entries):
            if(y_predict[i] > goldmax):
                y_predict[i] = goldmax
            elif(y_predict[i] < goldmin):
                y_predict[i] = goldmin

    offset = 0
    if(use_offset):
        if(fixed_offset != 0):
            offset = fixed_offset
        else:
            offset = goldmin

    # print("offset = " + offset)

    if(N==0):
        vals = []
        for val in y_gold:
            if(not (val in vals)):
                N+=1
                vals.append(val)

        if(not(quant_predictions and stuff_predictions)):
            for val in y_predict:
                if(not (val in vals)):
                    N+=1
                    vals.append(val)

    w = np.zeros((N,N))

    #build weight matrix w
    wnorm = (N-1.)**(-2)

    for i in range(0,N):
        for j in range(0,N):
            #add to weight matrix w
            w[i][j] = wnorm*(i-j)**2


    #build histogram matrix O
    #and histogram colums h_gold, h_predict
    O = np.zeros((N,N))
    h_gold = np.zeros(N)
    h_predict = np.zeros(N)


    for ind in range(0,n_entries):
        i = int(y_gold[ind] - offset)
        j = int(y_predict[ind] - offset)
        O[i][j] += 1
        h_gold[i] += 1
        h_predict[j] += 1

    E = np.outer(h_gold, h_predict)
    #scale E to have same sum as O (= number of entries)
    E = np.dot(np.sum(O)*1./np.sum(E),E)

    kupper = 0.
    klower = 0.
    for i in range(0,N):
        for j in range(0,N):
            kupper += w[i][j]*O[i][j]
            klower += w[i][j]*E[i][j]

    kappa = 1 - kupper*.1/klower
    return kappa



def test_qwk():

    ygold = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
    ypred = [3]*len(ygold)

    print(qwk(ygold,ypred))

# test_qwk()