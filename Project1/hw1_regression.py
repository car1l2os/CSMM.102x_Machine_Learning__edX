import numpy as np
import sys

lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter = ",")



# FOR TESTING WITH REAL DATA.
# lambda_input=1
# sigma2_input=2

# import pandas as pd
# # Load CSV and columns
# df = pd.read_csv("Housing.csv")
# y = df['price']
# X = df[['lotsize', 'bedrooms', 'bathrms', 'stories']]
# X=X.values.reshape(len(X),4)
# y=y.values.reshape(len(y),1)
# # Split the data into training/testing sets
# X_train = X[:-250]
# X_test = X[-250:]
 
# # Split the targets into training/testing sets
# y_train = y[:-250]
# y_test = y[-250:]



## Solution for Part 1
def part1(lambda_input, X, y):
    ## Input : Arguments to the function
    wRR = np.dot(np.linalg.inv( lambda_input + np.dot(X.T,X) ) , np.dot( X.T,y))
    ## Return : wRR, Final list of values to write in the file
    return wRR

wRR = part1(lambda_input, X_train, y_train)  # Assuming wRR is returned from the function
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file


## Solution for Part 2
def part2(lambda_input, sigma2_input, X_train, X_test):
    ## Input : Arguments to the function
    # start the calculation again at the top (for each x in x_test)
    active = []
    X = X_test.copy()
    Cov = np.linalg.inv( lambda_input + (1 /sigma2_input) * np.dot(X_train.T,X_train) )
    x_list = []
    for times in range(10):
        s_list = []
        
        for x in X:
            new_sigma2 = sigma2_input + np.dot(np.dot(x.T,Cov),x)
            s_list.append(new_sigma2)
        
        i = np.where(s_list ==np.max(s_list))[0][0]

        Cov = np.linalg.inv( np.linalg.inv(Cov) + (1 /sigma2_input) * np.inner(X[i],X[i].T)  )
        x_list.append(X[i])
        Xt = X.tolist()
        Xt.pop(i)
        X = np.array(Xt)
    for i in x_list:
        active.append(  np.where((X_test==i).mean(1)==1)[0][0]+1  )
    ## Return : active, Final list of values to write in the file 
    return active

active = part2(lambda_input, sigma2_input, X_train, X_test)  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output to file


