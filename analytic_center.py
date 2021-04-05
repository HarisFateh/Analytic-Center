# modification 1.3
# librarires for computation
import numpy as np
import math
from sys import exit
import warnings
warnings.filterwarnings("ignore")
# libraries for plotting
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol


# problem description (in form of A^T*x <= b)
A = np.array([[3.2,5.1],
              [-6.3,7.7],
              [2.5,-4.8]])

b = np.array([[10.0],[13.0],[50]])

# function for guessing initial feasible value for the system
def guessn(A,b):
    
    n = len(A)
    xs = np.zeros(A.shape)
    b = np.reshape(b, [n,])

    for i in range(n-1):
        An = np.zeros([2,2])
        bn = np.zeros([1,2])
        An = A[i:i+2]
        bn = b[i:i+2]
        xn = np.linalg.solve(An, bn)
        xs[i][0] = xn[0]
        xs[i][1] = xn[1]

    An = np.zeros([2,2])
    bn = np.zeros([1,2])

    An[0] = A[0]
    An[1] = A[n-1]
    bn[0][0] = b[0]
    bn[0][1] = b[n-1]

    bn = np.transpose(bn)
    xn = np.linalg.solve(An, bn)

    xs[n-1][0] = xn[0]
    xs[n-1][1] = xn[1]
    aug = 0.1
    for i in range(n):
    
        if(min(b - np.matmul(A,xs[i]))>0):
            gn = xs[i]
            return(gn)

    for i in range(n):

        if(min(b - np.matmul(A,xs[i]+aug))> 0):
            gn = xs[i]+aug
            return(gn)
    
    for i in range(n):

        if(min(b - np.matmul(A,xs[i]-aug)) > 0):
            gn = xs[i]-aug
            return(gn)

        else:
            print('no feasible found')
            return(0)

# function for plotting the system of polyhedron
def plottn(A,b,xnp):
    
    An = np.zeros([2,2])
    bn = np.zeros([1,2])
    xs = np.zeros(A.shape)
    n = len(A)
    b = np.reshape(b, [n,])

    for i in range(n-1):
        An = A[i:i+2]
        bn = b[i:i+2]
        xn = np.linalg.solve(An, bn)
        xs[i:i+1] = xn

    An[0] = A[0]
    An[1] = A[n-1]
    bn[0] = b[0]
    bn[1] = b[n-1]
    xn = np.linalg.solve(An, bn)
    xs[n-1:n] = xn
    #print(xs)
    for i in range(n):
        plt.plot(xs[i][0],xs[i][1],'go',markersize=10)
        
    for i in range(len(xnp)):
        plt.plot(xnp[i][0],xnp[i][1],'go',markersize=10)
    
    plt.plot(xnp[len(xnp)-1][0],xnp[len(xnp)-1][1],'ro',markersize=10)

    x1n = np.zeros([n,1])
    y1n = np.zeros([n,1])

    x1n = np.append(x1n,[xs[0][0]])
    y1n = np.append(y1n,[xs[0][1]])

    for i in range(n):
        x1n[i] = xs[i][0]
        y1n[i] = xs[i][1]

    plt.fill(x1n,y1n,'red',alpha=0.5)

    for i in range(n):
        plt.plot(x1n,y1n,'k--')

    plt.show()

def analytic_center(A,b):
    # problem description (in form of A^T*x <= b)

    # initial point guessing empirical method (to start from one of the corner point of polyhedron)

    guess = guessn(A, b)

    x_0 = np.array([[guess[0]],[guess[1]]]) # feasible point choice (inside the domain of the inequalities)
    # printing the initial guess:
    print('the initial guess for the problem:\n',x_0)
    # functional description in the use of algorithm
    At = np.transpose(A)
    def Sinv2(s):
        s2 = np.multiply(s,s)
        s2 = 1/s2
        sn2 = np.zeros((m,m), float)
        np.fill_diagonal(sn2,s2)
        return sn2

    # algorithm description
    alpha = 0.01         # learning rate or change of step or iterative content
    beta = 0.5           # changing directional content
    eps = 0.000001       # thresholding or tolerance
    k = 20               # max iterations
    ls = 1               # convergence criteria

    # checking the feasibility of the initial guess
    if(min(b - np.matmul(A,x_0))<=0):
        print('point is not in domain')
        exit()

    # length description and initialization
    m = len(b)
    n = len(x_0)
    x = x_0
    i = 0
    xi = x_0

    # algorithm loop of the analytic center finding
    while(i<20 and (ls/2.0)>=eps):

        s = b - np.matmul(A, x)
        # creating s^-2
        sn2 = Sinv2(s)
        # creating s^-1
        sn1 = 1/s

        # computation of paramters for step delta"x"
        H = np.matmul(np.matmul(At,sn2),A)
        g = np.matmul(At,sn1)
        dx = -1*(np.linalg.lstsq(H,g)[0])
        gt = np.transpose(g)
        # updating of covergence criteria
        ls = (-1*(np.matmul(gt,dx)))
        
        # temporal holding variable of directional content of the step
        t = 1

        # bringing the point in the domain by the following step
        while(min(b - np.matmul(A, x+t*dx))<=0):
            t = t*beta

        # backtracking line search method implication as: f(x_k+1) - f(x_k) > alpha * t * delta(f(x_k+1))
        while(-1*np.sum(np.log(b - np.matmul(A, x+t*dx))) + np.sum(np.log(b - np.matmul(A, x))) - (alpha*t*np.matmul(gt,dx))>0):
            t = t*beta

        # printing the iterative step and value of "x"
        print('\nthe iteration step:',i+1)
        # computing the next value of "x"
        x = x + t*dx
        #xi[i][0] = x[0]
        #xi[i][1] = x[1]
        xi = np.append(xi,x)
        print(x)
        # iteration updating    
        i = i+1

    # printing the final and optimal value of the analytic center of the given polyhedron
    print('\nthe optimal value is:')
    print(x)
    xi = np.reshape(xi, [int(len(xi)/2),2])
    plottn(A, b, xi)

analytic_center(A,b)
