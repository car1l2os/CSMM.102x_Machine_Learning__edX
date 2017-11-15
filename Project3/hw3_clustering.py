from __future__ import division
import numpy as np
import sys
from sklearn.datasets import make_moons, make_circles, make_blobs, make_classification
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt # to plot images

Def CreateDataset():
    X, y = make_blobs(n_samples=1000, n_features=2, centers=5, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None)
    rng = np.random.RandomState(2)
    X += 0.005 * rng.uniform(size=X.shape)
    X += X*0.05

    plt.scatter(X[:,0], X[:,1], c=y, s=60)
    plt.show()
    np.savetxt("X.csv", X, delimiter=",") # write output to file
    np.savetxt("y.csv", y, delimiter=",") # write output to file



X = np.genfromtxt("X.csv", delimiter=",")
y = np.genfromtxt("y.csv")

# X_train = np.genfromtxt(sys.argv[1], delimiter=",")
# y_train = np.genfromtxt(sys.argv[2])

def Kmeans(X):
    cluster_num = 5
    n = len(X)
    total_points = range(n)
    ind = list(total_points)
    np.random.RandomState(0)
    np.random.shuffle(ind)
    centroids = X[ind[:5]]
    for it in np.arange(1,11,1):
        distances = np.linalg.norm(X[:,np.newaxis,:]-centroids, axis=2)**2
        print('Kmeans: ', distances.sum())
        c_i = np.argmin(distances, axis=1)
        
        # plt.hold(True)
        # plt.scatter(X[:,0], X[:,1], c=c_i, s=60)
        # plt.scatter(centroids[:,0], centroids[:,1], c=['red']*5, s=150, marker='*')
        # plt.hold(False)
        # plt.show()

        for i in range(cluster_num):
            centroids[i,:] = (X[c_i==i]).sum(0) / (c_i==i).sum()
        np.savetxt("centroids-"+str(it)+".csv", centroids, delimiter=",") # write output to file
    return c_i


def GaussianDistribution(x, u, D):
    exponential_term = np.exp(-0.5 *    (np.matmul((x-u) , np.linalg.pinv(D))  * (x-u)).sum(-1)    )
    return ( exponential_term / np.sqrt(np.linalg.det(D)) ).squeeze() 

def Initialization(X,k):
    n = len(X)
    total_points = range(n)
    ind = list(total_points)
    np.random.RandomState(0)
    np.random.shuffle(ind)
    u = X[ind[:5]]
    # D = np.array([np.eye(X.shape[1])]*k)
    D = np.array([np.cov(X.T)]*k)
    pi =  np.array([1./k] * k)
    phi = np.zeros([n,k])

    phi = Expectation_step(phi, u, D, pi, k)
    std = X.std(0)
    if ((np.abs(std)-1)<4).any():
        ind = np.where(X.std(1)>2)[0] # Assuming the data was normalized; peak detection
        if ind.any():
            mean = X.mean(0)
            _, c = np.where(X[ind]>4)
            X[ind,c] = mean[c]

    return u, D, pi, phi, X

def Expectation_step(phi, u, D, pi, k):
    for i in range(0,k):
        phi[:,i] = pi[i]*GaussianDistribution(X,u[i],D[i])
    phi = phi/phi.sum(1)[:,None]
    return phi

def MaximumLikelihood_step(X, k, u, D,phi):
    nk = phi.sum(0)
    pi = nk/X.shape[0]

    for ki in range(k):
        phi_ = np.tile(phi[:,ki],(X.shape[1],1)).T
        u[ki,:] = (phi_ * X).sum(0) / nk[ki]
        D[ki,:,:] =  np.matmul((phi_*(X - u[ki,:])).T , (X - u[ki,:]) ) / nk[ki]
        
    return u, D, pi

def EM_GMM(X):
    cluster_num = 5
    n = len(X)
    u, D, pi, phi, X = Initialization(X,cluster_num) # Initialize parameters and set outliers with the mean value in each position.

    for it in np.arange(1,11,1):
        print(it)
        # E-step
        phi = Expectation_step(phi, u, D, pi,cluster_num)
        # M-step
        u, D, pi = MaximumLikelihood_step(X,cluster_num, u, D,phi)

        for k in range(cluster_num):
            np.savetxt("Sigma-"+str(k+1)+"-"+str(it)+".csv", D[k], delimiter=",") # write output to file

        np.savetxt("pi-"+str(it)+".csv", u, delimiter=",") # write output to file
        np.savetxt("mu-"+str(it)+".csv", pi, delimiter=",") # write output to file
    return phi

y = np.uint8(y)
y0_ = Kmeans(X)
y1_ = EM_GMM(X)

y_ = y0_
m = confusion_matrix(y,y_)
print('K-means')
print(m)

y_ = y1_.argmax(1)
m = confusion_matrix(y,y_)
print('GMM')
print(m)
