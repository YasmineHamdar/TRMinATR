#Import Libraries: 
import numpy as np 
from numpy import linalg as la
import matplotlib
import matplotlib.pyplot as plt
import random
import warnings
import cvxpy as cvx
import cvxopt as cvxop
import autograd.numpy as aunp
import autograd as au
from scipy import linalg as laSci
from scipy.optimize import linprog
import scipy.fftpack as fft
from cvxpy import atoms
from scipy.optimize import linprog

#Size of input vector
N = 5

#Defines the objective function
def objective_function(w):
    return -sum(w[i, 0] * np.log(w[i, 0]) for i in range(N))

#Defines the gradient of the objective function
def objective_gradient(w):
    G = np.zeros(N)
    for i in range(N):
        G[i] = -1 - np.log(w[i, 0])
    return np.matrix(G).T

#Defines the quadratic approximation of the objective function
def quad_function(p, g, B):
    return np.asscalar(la.multi_dot([g.T, p]) + 0.5 * la.multi_dot([p.T, B, p]))

#Initialization Method (I) for the gamma-variable gamma(k) to initialize B0
def initialization_method(s, y):
    #Compute numerator term
    first_term = la.multi_dot([y.T, y])
    #Compute the denominator term
    second_term = la.multi_dot([s.T, y])
    #Return the maximum between computed gamma value and 1
    return np.asscalar(first_term / second_term)

#Computes the k-th matrices psi_k and Mk
def compute_psi_and_M(B0, S, Y):
    #Compute S.T * Y
    tmp = np.matrix(la.multi_dot([S.T, Y]))
    #Get the lower triangular matrix
    Lk = np.tril(tmp, k=-1)
    #Get the diagonal matrix
    Dk = np.diag(np.diag(tmp))
    #Form the psi_k block    
    #psi_k = np.block([la.multi_dot([B0, S]), Y])
    #psi_k = np.column_stack((la.multi_dot([B0, S]), Y))    
    psi_k = np.concatenate((la.multi_dot([B0, S]), Y) ,axis=1)    
    #Form row 1 of the Mk matrix
    Mk_row1 = np.block([-la.multi_dot([S.T, B0, S]), -Lk])
    #Form row 2 of the Mk matrix
    Mk_row2 = np.block([-Lk.T, Dk])
    #Form the whole Mk matrix
    Mk = la.inv(np.block([[Mk_row1], [Mk_row2]]))
    return psi_k, Mk

#l-BFGS trust-region subproblem solver
def lbfgs_trust_region_subproblem(gk, psi_k, Mk, gamma_k, delta_k):
    #QR decomposition of the psi_k matrix
    Qk, Rk = la.qr(psi_k, 'reduced')
    #Compute eigen-decomposition (values and vectors) of Rk*Mk*Rk.T
    eigen_values, eigen_vectors = la.eig(la.multi_dot([Rk, Mk, Rk.T]))    
    #Sort eigen vectors & values
    idx = eigen_values.argsort()
    eigen_values_sorted = eigen_values[idx]
    eigen_vectors_sorted = eigen_vectors[:,idx]
    #Lambda_hat represents the sorted eigen-values
    lambda_hat = np.matrix(eigen_values_sorted).T
    #Vk represents the sorted eigen-vectors
    Vk = eigen_vectors_sorted    
    #Compute lambda_1
    lambda_1 = lambda_hat + gamma_k
    #Define P_parallel matrix
    P_ll = la.multi_dot([psi_k, la.inv(Rk), Vk])
    #Compute g_parallel
    g_ll = la.multi_dot([P_ll.T, gk])
    #Compute norm of P_perpendicular.T * gk
    g_NL_norm = abs(la.norm(gk)**2 - la.norm(g_ll)**2)**0.5
    #Define tau_star
    tau_star = 1
    #Compute phi using the phi(sigma) function
    sigma_k = 0
    phi = phi_bar_func(g_ll, g_NL_norm, lambda_1, sigma_k, gamma_k, delta_k)    
    #If phi >= 0 then tau_star will be equal to gamma_k
    if phi >= 0:
        tau_star = gamma_k
    #Solve for sigma_star using newton method and assign tau_star the value gamma_k + sigma_star
    else:
        lambda_min = min(lambda_1.min(), gamma_k)
        sigma_star = solve_newton_equation_to_find_sigma(g_ll, g_NL_norm, lambda_1, gamma_k, delta_k, lambda_min)
        tau_star = gamma_k + sigma_star    
    #Compute Newton's step
    first_term = la.inv(tau_star * la.inv(Mk) + la.multi_dot([psi_k.T, psi_k]))
    second_term = np.identity(N) - la.multi_dot([psi_k, first_term, psi_k.T])
    third_term = la.multi_dot([second_term, gk])
    #Return the computed Newton's step
    return (-1 / tau_star) * third_term

#Defines a function to return the phi(sigma)
def phi_bar_func(g_ll, g_NL_norm, lambda_1, sigma_k, gamma_k, delta_k):
    #Compute u
    u = sum(g_ll[i, 0]**2 / (lambda_1[i, 0] - sigma_k)**2 for i in range(len(g_ll))) + (g_NL_norm**2 / (gamma_k - sigma_k)**2)
    #Compute v(sigma_k)
    v_sigma_k = u**0.5
    #Compute phi of sigma_k
    return (1 / v_sigma_k) - (1 / delta_k)

#Defines the function to return the gradient/derivative of the phi(sigma) function
def phi_bar_prime_func(g_ll, g_NL_norm, lambda_1, sigma_k, gamma_k, delta_k):
    #Compute u
    u = sum(g_ll[i, 0]**2 / (lambda_1[i, 0] - sigma_k)**2 for i in range(len(g_ll))) + (g_NL_norm**2 / (gamma_k - sigma_k)**2)
    #Compute derivative of u
    u_prime = -sum(g_ll[i, 0]**2 / (lambda_1[i, 0] - sigma_k)**3 for i in range(len(g_ll))) - g_NL_norm**2 / (gamma_k - sigma_k)**3
    #Compute derivative of phi(sigma)
    return u**(-3/2) * u_prime

#Newton method to find sigma_star
def solve_newton_equation_to_find_sigma(g_ll, g_NL_norm, lambda_1, gamma_k, delta_k, lambda_min, tol = 1E-4):
    sigma = max( 0, -lambda_min)
    if phi_bar_func(g_ll, g_NL_norm, lambda_1, sigma, gamma_k, delta_k) < 0:
        sigma_hat = np.asscalar(max(abs(g_ll) / delta_k - lambda_1))
        sigma = max( 0, sigma_hat)
        while(abs(phi_bar_func(g_ll, g_NL_norm, lambda_1, sigma, gamma_k, delta_k) ) > tol):
            phi_bar = phi_bar_func(g_ll, g_NL_norm, lambda_1, sigma, gamma_k, delta_k)
            phi_bar_prime = phi_bar_prime_func(g_ll, g_NL_norm, lambda_1, sigma, gamma_k, delta_k)
            sigma = sigma - phi_bar / phi_bar_prime
        sigma_star = sigma
    elif lambda_min < 0:
        sigma_star = - lambda_min
    else:
        sigma_star = 0
    return sigma_star 

#Backtracking line search function only used to return search direction for the first iteration k==0
def backtracking_line_search(func, gk, wk, alpha = 1e-04, beta = 0.9):
    t = 1
    pk = -gk
    while np.asscalar(func(wk + t * pk)) > np.asscalar(func(wk) + alpha * t * np.dot(gk.T, pk)):
        if t < 0.1:
            print('WARNING! Backtracking line search did not work')
            break        
        t = t * beta
    return t * pk 

#The limited-memory BFGS trust-region method
def lbfgs_trust_region(func, quad_func, grad, w0, delta_init = 1.0, delta_max = 3.0, eta = 0.25, max_iter = 100, mem_limit = 30, tol = 1e-04):
    #Effective iterations
    eff_iter = 0
    #Initialize wk to w0
    wk = w0
    #Initialize trust-region radius
    delta_k = delta_init
    #Matrix holding last m-search steps
    S = None
    #Matrix holding last m-gradient variations based on search steps
    Y = None
    for k in range(max_iter):
        #Compute the gradient at the current point
        gk = grad(wk)
        
        #For the first iteration k==0
        if k == 0:
            #Compute initial step
            p0 = backtracking_line_search(func, grad(wk), wk)
            #Compute s0 step difference
            s0 = p0
            #Compute y0 gradient step difference
            y0 = grad(wk + p0) - grad(wk)
            #Add s0 to storage arrays
            S = s0
            #Add y0 to storage arrays
            Y = y0
            #Update weights
            wk = wk + p0
        
        #Compute gamma_k by using the Initialization method (I)
        gamma_k = initialization_method(S[:, eff_iter], Y[:, eff_iter])

        #Initialize B0
        B0 = gamma_k * np.identity(N)
        #Choose gamma_k to be the maximum between itself and 1
        gamma_k = max(gamma_k, 1)
        #Compute psi_k and Mk
        psi_k, Mk = compute_psi_and_M(B0, S, Y)
        #Compute Bk matrix
        Bk = B0 + la.multi_dot([psi_k, Mk, psi_k.T])
        #Compute newton step for the trust region subproblem
        pk = lbfgs_trust_region_subproblem(gk, psi_k, Mk, gamma_k, delta_k)
        #Compute sk step difference
        sk = pk
        #Compute yk gradient step difference
        yk = grad(wk + pk) - grad(wk)
        
        #If S[k].T * Y[k] > 0
        if la.multi_dot([sk.T, yk]) > 0:
            #Increment effective iterations
            eff_iter = eff_iter + 1
            #Add new displacement & gradient difference
            S = np.column_stack((S, sk))
            Y = np.column_stack((Y, yk))
            #Discard first items from S and Y arrays if memory limit exceeded
            if S.shape[1] > mem_limit or Y.shape[1] > mem_limit:
                S = S[:,1:]
                Y = Y[:,1:]
        
        #Compute rho_k which is the ratio of the real objective function on its quadratic approximation counterpart
        rho_k = (func(wk + pk) - func(wk)) / (quad_func(pk, gk, Bk) - quad_func(np.zeros(N), gk, Bk))
        
        #If the ratio of real function on quad approximation is greater than eta then update weights
        if rho_k > eta:
            wk = wk + pk
        
        #Calculate the Euclidean norm of pk
        norm_pk = np.asscalar(np.sqrt(np.dot(pk.T, pk)))
        
        #If ratio is close to zero or negative, => trust region must be minimized
        if rho_k < 0.25:
            delta_k = 0.25 * norm_pk
        else:
            #If ratio is close to one and pk has reached the boundary of the trust region, therefore the trust region is expanded.
            if rho_k > 0.75 and norm_pk == delta_k:
                delta_k = min(2.0 * delta_k, delta_max)        
        
        #Check whether the norm of the gradient vector vanishes or not
        print(la.norm(gk))
        if la.norm(gk) < tol:
            break

#np.column_stack
w0 = np.matrix(np.ones(N) * 5).T
lbfgs_trust_region(objective_function, quad_function, objective_gradient, w0)