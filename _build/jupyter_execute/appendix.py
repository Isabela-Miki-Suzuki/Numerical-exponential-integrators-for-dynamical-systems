#!/usr/bin/env python
# coding: utf-8

# # Appendix

# ## Code

# All the functions coded are in the following environment.

# In[1]:


from math import *
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy import linalg

stab_lim = 1000.0

def classic_euler(t0, tf, n, x0, A, g):
    '''(float, float, int, np.array, np.matrix, function) -> np.matrix'''
    h = (tf-t0)/n
    x = np.zeros((x0.size,n), dtype=np.complex_)
    x[:,0]=x0
    t = t0
    for i in range(1, n):
        x[:,i] = x[:,i-1] + h*(np.matmul(-A,x[:,i-1]) + g(x[:,i-1],t))
        t = t0 + i*h
        if np.any(x[:,i].real > stab_lim):
            x[:,i] = np.nan
    return x

def exponential_euler(t0, tf, n, x0, A, g):
    '''(float, float, int, np.array, np.matrix, function) -> np.matrix'''
    h = (tf-t0)/n
    x = np.zeros((x0.size,n), dtype=np.complex_)
    x[:,0] = x0
    t = t0
    exponential_matrix = expm(-h*A)
    hphi1 = calculate_hphi1(h, A)
    for i in range(1, n):
        x[:,i] = np.matmul(exponential_matrix, x[:,i-1]) + np.matmul(hphi1,g(x[:,i-1],t))
        t = t0 + i*h
    return x

def calculate_hphi1(h, A):
  '''(float, np.matrix) -> np.matrix'''
  dim = A.shape[0]
  hphi1 = np.matmul(np.eye(dim)-expm(-h*A), linalg.inv(A))
  return hphi1

def calculate_hphi2(h, A, hphi1):
    #IT IS NOT H2PHI2
    '''(float, np.matrix, np.matrix) -> np.matrix'''
    dim = A.shape[0]
    hphi2 = np.matmul(np.eye(dim)-hphi1/h, linalg.inv(A))
    return hphi2

def calculate_hphi3(h, A, hphi2):
    '''(float, np.matrix, np.matrix) -> np.matrix'''
    dim = A.shape[0]
    hphi3 = np.matmul(1/2*np.eye(dim)-hphi2/h, linalg.inv(A))
    return hphi3

def etd2(t0, tf, n, x0, A, g, derivate_of_g):
    '''(float, float, int, np.array, np.matrix, function, function) -> np.matrix'''
    h = (tf-t0)/n
    x = np.zeros((x0.size,n), dtype=np.complex_)
    x[:,0] = x0
    t = t0
    exponential_matrix = expm(-h*A)
    hphi1 = calculate_hphi1(h, A)
    hphi2 = calculate_hphi2(h, A, hphi1)
    for i in range(1, n):
        x[:,i] = np.matmul(exponential_matrix, x[:,i-1]) + np.matmul(hphi1,g(x[:,i-1],t)) + h*np.matmul(hphi2,derivate_of_g(x[:,i-1],t))
        t = t0 + i*h
    return x

def rk2(t0, tf, n, x0, A, g): #heun s method
    '''(float, float, int, np.array, np.matrix, function) -> np.matrix'''
    h = (tf-t0)/n
    x = np.zeros((x0.size,n), dtype=np.complex_)
    x[:,0]=x0
    t = t0
    for i in range(1, n):
        a = x[:,i-1] + h*(np.matmul(-A,x[:,i-1]) + g(x[:,i-1],t))
        f1 = np.matmul(-A,x[:,i-1]) + g(x[:,i-1],t)
        f2 = np.matmul(-A,a) + g(a,t)
        x[:,i] = x[:,i-1] + .5 * h * (f1 + f2)
        t = t0 + i*h
        if np.any(x[:,i].real > stab_lim):
            x[:,i] = np.nan
    return x

def etd2rk(t0, tf, n, x0, A, g):
    '''(float, float, int, np.array, np.matrix, function) -> np.matrix'''
    h = (tf-t0)/n
    x = np.zeros((x0.size,n), dtype=np.complex_)
    x[:,0]=x0
    t = t0
    exponential_matrix = expm(-h*A)
    hphi1 = calculate_hphi1(h, A)
    hphi2 = calculate_hphi2(h, A, hphi1)
    for i in range(1, n):
        a = np.matmul(exponential_matrix, x[:,i-1]) + np.matmul(hphi1,g(x[:,i-1],t))
        x[:,i] = np.matmul(exponential_matrix, x[:,i-1]) + np.matmul(hphi1,g(x[:,i-1],t)) + np.matmul(hphi2,g(a, t0 + i*h)-g(x[:,i-1],t))
        t = t0 + i*h
    return x

def etd2rk_midpoint_rule(t0, tf, n, x0, A, g):
    '''(float, float, int, np.array, np.matrix, function) -> np.matrix'''
    h = (tf-t0)/n
    x = np.zeros((x0.size,n), dtype=np.complex_)
    x[:,0]=x0
    t = t0
    exponential_matrix = expm(-h*A)
    exponential_matrix_2 = expm(-h/2*A)
    h_2phi1_2 = calculate_hphi1(h/2, A)
    hphi1 = calculate_hphi1(h, A)
    hphi2 = calculate_hphi2(h, A, hphi1)
    for i in range(1, n):
        b = np.matmul(exponential_matrix_2, x[:,i-1]) + np.matmul(h_2phi1_2,g(x[:,i-1],t))
        x[:,i] = np.matmul(exponential_matrix, x[:,i-1]) + np.matmul(hphi1,g(x[:,i-1],t)) + 2*np.matmul(hphi2,g(b, t + h/2)-g(x[:,i-1],t))
        t = t0 + i*h
    return x

def etd2rk_trapezoidal_naive(t0, tf, n, x0, A, g):
    '''(float, float, int, np.array, np.matrix, function) -> np.matrix'''
    h = (tf-t0)/n
    x = np.zeros((x0.size,n), dtype=np.complex_)
    x[:,0]=x0
    t = t0
    exponential_matrix = expm(-h*A)
    hphi1 = calculate_hphi1(h, A)
    for i in range(1, n):
        a = np.matmul(exponential_matrix, x[:,i-1]) + np.matmul(hphi1,g(x[:,i-1],t))
        x[:,i] = np.matmul(exponential_matrix, x[:,i-1]) + .5 * h * (np.matmul(exponential_matrix, g(x[:,i-1],t)) + g(a, t0 + i*h))
        t = t0 + i*h
    return x

def etd2rk_midpoint_rule_naive(t0, tf, n, x0, A, g):
    '''(float, float, int, np.array, np.matrix, function, np.matrix) -> np.matrix'''
    h = (tf-t0)/n
    x = np.zeros((x0.size,n), dtype=np.complex_)
    x[:,0]=x0
    t = t0
    exponential_matrix = expm(-h*A)
    exponential_matrix_2 = expm(-h/2*A)
    h_2phi1_2 = calculate_hphi1(h/2, A)
    for i in range(1, n):
        b = np.matmul(exponential_matrix_2, x[:,i-1]) + np.matmul(h_2phi1_2,g(x[:,i-1],t))
        x[:,i] = np.matmul(exponential_matrix, x[:,i-1]) + h * np.matmul(exponential_matrix_2, g(b, t+h/2))
        t = t0 + i*h
    return x

def rk4(t0, tf, n, x0, A, g):
    '''(float, float, int, np.array, np.matrix, function) -> np.matrix'''
    h = (tf-t0)/n
    x = np.zeros((x0.size,n), dtype=np.complex_)
    x[:,0]=x0
    t = t0
    for i in range(1, n):
        k1 = np.matmul(-A,x[:,i-1]) + g(x[:,i-1],t)
        x2 = x[:,i-1] + h * k1 / 2
        k2 = np.matmul(-A,x2) + g(x2,t+h/2)
        x3 = x[:,i-1] + h * k2 / 2
        k3 = np.matmul(-A,x3) + g(x3,t+h/2)
        x4 = x[:,i-1] + h * k3
        k4 = np.matmul(-A,x4) + g(x4,t0 + i*h)
        x[:,i] = x[:,i-1] + h / 6 * (k1 + 2*k2 + 2*k3 + k4)
        t = t0 + i*h
        if np.any(x[:,i].real > stab_lim):
            x[:,i] = np.nan
    return x

def etd3rk_similar(t0, tf, n, x0, A, g):
  '''(float, float, int, np.array, np.matrix, function, np.matrix) -> np.matrix'''
  h = (tf-t0)/n
  x = np.zeros((x0.size,n), dtype=np.complex_)
  x[:,0]=x0
  t = t0
  exponential_matrix = expm(-h*A)
  exponential_matrix_2 = expm(-h/2*A)
  hphi1 = calculate_hphi1(h, A)
  h_2phi1_2 = calculate_hphi1(h/2, A)
  hphi2 = calculate_hphi2(h, A, hphi1)
  h_2phi2_2 = calculate_hphi2(h/2, A, h_2phi1_2)
  hphi3 = calculate_hphi3(h, A, hphi2)
  for i in range(1, n):
    fst_term = np.matmul(exponential_matrix, x[:,i-1])
    fst_term_2 = np.matmul(exponential_matrix_2, x[:,i-1])
    a = fst_term + np.matmul(hphi1,g(x[:,i-1],t))
    a_ = fst_term_2 + np.matmul(h_2phi1_2,g(x[:,i-1],t))
    c = fst_term + np.matmul(hphi1,g(x[:,i-1],t)) + np.matmul(hphi2,g(a, t0 + i*h)-g(x[:,i-1],t))
    c_ = fst_term_2 + np.matmul(h_2phi1_2,g(x[:,i-1],t)) + np.matmul(h_2phi2_2,g(a_, t0 + i*h)-g(x[:,i-1],t))
    snd_term = np.matmul(hphi1, g(c_, t+h/2))
    trd_term = np.matmul(hphi2 - hphi1/2,g(c, t0 + i*h)-g(x[:,i-1],t))
    fth_term = 4 * np.matmul(hphi3+hphi1/8-hphi2/2, g(c, t0 + i*h)+g(x[:,i-1],t)-2*g(c_, t + h/2))
    x[:,i] = fst_term + snd_term + trd_term + fth_term
    t = t0 + i*h
  return x

def etd3rk(t0, tf, n, x0, A, g):
  '''(float, float, int, np.array, np.matrix, function, np.matrix) -> np.matrix'''
  h = (tf-t0)/n
  x = np.zeros((x0.size,n), dtype=np.complex_)
  x[:,0]=x0
  t = t0
  exponential_matrix = expm(-h*A)
  exponential_matrix_2 = expm(-h/2*A)
  hphi1 = calculate_hphi1(h, A)
  h_2phi1_2 = calculate_hphi1(h/2, A)
  hphi2 = calculate_hphi2(h, A, hphi1)
  h_2phi2_2 = calculate_hphi2(h/2, A, h_2phi1_2)
  hphi3 = calculate_hphi3(h, A, hphi2)
  for i in range(1, n):
    fst_term = np.matmul(exponential_matrix, x[:,i-1])
    fst_term_2 = np.matmul(exponential_matrix_2, x[:,i-1])
    a = fst_term_2 + np.matmul(h_2phi1_2,g(x[:,i-1],t))
    b = fst_term + np.matmul(hphi1,2*g(a,t+h/2)-g(x[:,i-1],t))
    snd_term = np.matmul(hphi1, g(a, t+h/2))
    trd_term = np.matmul(hphi2 - hphi1/2,g(b, t0 + i*h)-g(x[:,i-1],t))
    fth_term = 4 * np.matmul(hphi3+hphi1/8-hphi2/2, g(b, t0 + i*h)+g(x[:,i-1],t)-2*g(a, t + h/2))
    x[:,i] = fst_term + snd_term + trd_term + fth_term
    t = t0 + i*h
  return x

def etd3rk_naive(t0, tf, n, x0, A, g):
  '''(float, float, int, np.array, np.matrix, function, np.matrix) -> np.matrix'''
  h = (tf-t0)/n
  x = np.zeros((x0.size,n), dtype=np.complex_)
  x[:,0]=x0
  t = t0
  exponential_matrix = expm(-h*A)
  exponential_matrix_2 = expm(-h/2*A)
  hphi1 = calculate_hphi1(h, A)
  h_2phi1_2 = calculate_hphi1(h/2, A)
  for i in range(1, n):
    fst_term = np.matmul(exponential_matrix, x[:,i-1])
    fst_term_2 = np.matmul(exponential_matrix_2, x[:,i-1])
    a = fst_term + np.matmul(hphi1,g(x[:,i-1],t))
    a_ = fst_term_2 + np.matmul(h_2phi1_2,g(x[:,i-1],t))
    c = fst_term + .5 * h * (np.matmul(exponential_matrix, g(x[:,i-1],t)) + g(a, t0 + i*h))
    c_ = fst_term_2 + .25 * h * (np.matmul(exponential_matrix_2, g(x[:,i-1],t)) + g(a_, t0 + i*h))
    snd_term = np.matmul(exponential_matrix, g(x[:,i-1],t))
    trd_term = 4*np.matmul(exponential_matrix_2, g(c_,t+h/2))
    fth_term = g(c, t0 + i*h)
    x[:,i] = fst_term + h*(snd_term + trd_term + fth_term)/6
    t = t0 + i*h
  return x

def error_2(x_approx, x_exact):
    ''' (np.vector, np.vector) -> float '''
    #make sure that x_approx and x_exact have the same lenght
    v = (x_approx - x_exact)*(x_approx - x_exact).conjugate()
    #^certainly pure real
    return np.sqrt(float(np.sum(v)/x_approx.size)) #normalized

def error_sup(x_approx, x_exact):
    ''' (np.vector, np.vector) -> float '''
    #make sure that x_approx and x_exact have the same lenght
    v = abs(x_approx - x_exact)
    return np.amax(v)

def g(x, t):
    ''' (np.array, float) -> float
        (x, t) -> g(x, t)
    '''
    g = np.array([np.sin(t)])
    return g

def g_linear( x, t ):
    ''' (np.array, float) -> np.array
        (x, t) -> g(x, t)
    '''
    g = np.zeros(x.size)
    return g

def sol( t ):
    ''' (float, float) -> float
    RECEIVES the initial value and a real (t).
    APPLIES the cauchy problem solution to this initial value at this point.
    RETURNS a real value.
    '''
    lmba = 100
    sol = np.exp(-lmba*t)+(np.exp(-lmba*t)+lmba*np.sin(t)-np.cos(t))/(1+lmba*lmba)
    return sol

def sol_given_lmba(lmba, t ):
    ''' (float, float) -> float
    RECEIVES the initial value and a real (t).
    APPLIES the cauchy problem solution to this initial value at this point.
    RETURNS a real value.
    '''
    sol = np.exp(-lmba*t)+(np.exp(-lmba*t)+lmba*np.sin(t)-np.cos(t))/(1+lmba*lmba)
    return sol

def vectorize_sol_given_lmba(lmba, t0, t1, n, sol):
    '''
    (float, float, float, int, function) -> np.vector
    n is the number of steps
    '''
    x = np.zeros((sol(lmba,t0).size,n), dtype=np.complex_)
    h = (t1-t0)/n
    for i in range(n):
        x[:,i] = sol(lmba, t0+i*h)
    return x

def vectorize_sol(t0, t1, n, sol):
    '''
    (float, float, int, function) -> np.vector
    n is the number of steps
    '''
    x = np.zeros((sol(t0).size,n), dtype=np.complex_)
    h = (t1-t0)/n
    for i in range(n):
        x[:,i] = sol(t0+i*h)
    return x

def A_1D(lmba):
  '''(int) -> np.matrix'''
  return np.array([[lmba]])

def A_2D(lmba):
  '''(int) -> np.matrix'''
  return np.array([[0, -lmba],[lmba, 0]])

def errors_for_lambdas_array(n, method, t0, tf, x0, lmba0, lmbaf, Af, g, sol_given_lmba, vectorize_sol_given_lmba, error):
    '''
    This function is a variation of the errors_array function. Here, the linear
    part of the problem is varying instead of the number of steps, which is now
    fixed.
    This function will RETURN 2 arrays.
    The first one has the errors of the approximations given by the method with
    coefficient of the linear part of the ploblem
    A = Af(lmba0), Af(lmba0+1), Af(lmba0+2), ..., Af(lmbaf-1).
    The second is [lmba0, lmba0+1, lmba0+2, ..., lmbaf-1]

    RECEIVES:
    n is the number of steps. (int)
    method have arguments (t0, tf, n, x0, lmba, g) and return a
    np.vector of length n (0, 1, 2, ..., n-1), n is the number of steps. (function)
    t0 is the initial point of the approximation. (float)
    tf is the last one. (float)
    x0 is the initial value of the Cauchy problem. (np.array)
    lmba0 and lmbaf are integers as described before. (int)
    Af is a function that receives the stiffness parameter and returns the
    corresponding linear coefficient. (function)
    g is a function (float, float) -> (float). (function)
    sol is a function (float) -> (float). (function)
    vectorize_sol is a function that "transforms sol in a vector" (function)
    (float, float, int, function) -> (np.array)
    (t0, tf, n, sol) -> np.array([sol[t0], sol[t0+h], sol[t0+2h], ..., sol[tf-1]])
    error is a function (np.array, np.array) -> (float) (function)
    '''
    v = np.zeros(lmbaf-lmba0)
    domain = np.arange(lmba0, lmbaf)
    for i in range(lmbaf-lmba0):
        lmba = lmba0 + i
        m = method(t0, tf, n, x0, Af(lmba), g)
        exact = vectorize_sol_given_lmba(lmba, t0, tf, n, sol_given_lmba)
        if np.max(np.abs(m))>1000:
            v[lmba-lmba0]=np.nan
        else:
            v[lmba-lmba0] = error(m, exact)
    return v, domain

def errors_array(n0, nf, method, t0, tf, x0, A, g, sol, vectorize_sol, error):
  '''
  This function will RETURN 2 arrays.
  The first one has the errors of the approximations given by the method with
  number of steps n = n0, n0+1, n0+2, ..., nf-1.
  The second is [n0, n0+1, n0+2, ..., nf-1]

  RECEIVES:
  n0 is the first number of steps. (int)
  nf is the last one plus 1. (int)
  method have arguments (t0, tf, n, x0, A, lmba, g) and return a
  np.vector of length n (0, 1, 2, ..., n-1), n is the number of steps. (function)
  t0 is the initial point of the approximation. (float)
  tf is the last one. (float)
  x0 is the initial value of the Cauchy problem. (float)
  A is the coefficient os the linear part of the ploblem. (float)
  g is a function (int, float, float) -> (float). (function)
  sol is a function (int, float) -> (float). (function)
  vectorize_sol is a function that "transforms sol in a vector" (function)
  (float, float, int, function) -> (np.array)
  (t0, tf, n, sol) -> np.array([sol[t0], sol[t0+h], sol[t0+2h], ..., sol[tf-1]])
  error is a function (np.array, np.array) -> (float) (function)
  '''
  v = np.zeros(nf-n0)
  domain = np.arange(n0, nf)
  for n in range(n0, nf):
    m = method(t0, tf, n, x0, A, g)
    exact = vectorize_sol(t0, tf, n, sol)
    if np.max(np.abs(m))>1000:
        v[n-n0]=np.nan
    else:
        v[n-n0] = error(m, exact)
  return v, domain

def graphic_2D(domain, matrix, names, labelx, labely, title, key1, key2):
  '''
  domain is a list of np.arrays [[length n1], [legth n2], ..., [length nk]]
  k = 1, 2, ..., 5 lines. (list)
  matrix is a list of np.arrays [[length n1], [legth n2], ..., [length nk]]
  k = 1, 2, ..., 5 lines - same length that domain. (list)
  names is a list of the labels for the graphs, must have the same length that
  the number of lines in matrix. (list of Strings)
  labelx is the name of the x coordinate. (String)
  labely is the name of the y coordinate. (String)
  title is the title of the graph. (String)
  key1 is a boolean that indicates if the last graph must be black. (bool)
  key2 is a boolean that indicates if it should use the log scale. (bool)
  '''
  fig, ax = plt.subplots()

  colors = ['red', 'orange', 'brown', 'green', 'cyan', 'blue', 'pink', 'yellow', 'gold', 'maroon']
  for i in range(len(names)-1):
    ax.plot(domain[i], matrix[i], color=colors[i], label=names[i])
  if key1:
    ax.plot(domain[len(names)-1], matrix[len(names)-1], color='black', label=names[len(names)-1])
  else:
    ax.plot(domain[len(names)-1], matrix[len(names)-1], color=colors[len(names)-1], label=names[len(names)-1])
  if key2:
    plt.yscale('log')
  ax.legend()
  ax.set_xlabel(labelx)
  ax.set_ylabel(labely)
  ax.set_title(title)
  return fig, ax

def graphic_3D(domain, matrix1, matrix2, names, labelx, labely, labelz, title, key1, key2):
  '''
  domain is a list of np.arrays [[length n1], [legth n2], ..., [length nk]]
  k = 1, 2, ..., 5 lines. (list)
  matrix1 and matrix2 are lists of np.arrays [[length n1], [legth n2], ..., [length nk]]
  k = 1, 2, ..., 5 lines - same length that domain. (list)
  names is a list of the labels for the graphs, must have the same length that
  the number of lines in matrix. (list of Strings)
  labelx is the name of the x coordinate. (String)
  labely is the name of the y coordinate. (String)
  labelz is the name of the z coordinate. (String)
  title is the title of the graph. (String)
  key1 is a boolean that indicates if the last graph must be black. (bool)
  key2 is a boolean that indicates if it should use the log scale. (bool)
  '''
  fig = plt.figure()
  ax = plt.figure().add_subplot(projection='3d')

  colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
  for i in range(len(names)-1):
    ax.plot(domain[i], matrix1[i], matrix2[i], color=colors[i], label=names[i])
  if key1:
    ax.plot(domain[len(names)-1], matrix1[len(names)-1], matrix2[len(names)-1], color='black', label=names[len(names)-1])
  else:
    ax.plot(domain[len(names)-1], matrix1[len(names)-1], matrix2[len(names)-1], color=colors[len(names)-1], label=names[len(names)-1])
  if key2:
    plt.yscale('log')
  ax.legend()
  ax.set_xlabel(labelx)
  ax.set_ylabel(labely)
  ax.set_zlabel(labelz)
  ax.set_title(title)
  return fig, ax

def errors_2x(n0, k, method, t0, tf, x0, A, g, sol, vectorize_sol, error):
  '''
  This function will RETURN a np.array with the errors of the approximations given
  by the method with number of steps n = n0, 2*n0, 2**2*n0, ..., 2**(k-1)*n0.

  RECEIVES:
  n0 is the first number of steps. (int)
  k is the number of errors in the final array. (int)
  method have arguments (t0, tf, n, x0, lmba, g) and return a
  np.vector of length n (0, 1, 2, ..., n-1), n is the number of steps. (function)
  t0 is the initial point of the approximation. (float)
  tf is the last one. (float)
  x0 is the initial value of the Cauchy problem. (float)
  A is the coefficient of the linear part of the ploblem. (np.matrix)
  g is a function (float, float) -> (float). (function)
  sol is a function (float) -> (float). (function)
  vectorize_sol is a function that "transforms sol in a vector" (function)
  (float, float, int, function) -> (np.array)
  (t0, tf, n, sol) -> np.array([sol[t0], sol[t0+h], sol[t0+2h], ..., sol[tf-1]])
  error is a function (np.array, np.array) -> (float) (function)
  '''
  v = np.zeros(k)
  domain = np.zeros(k)
  for i in range(k):
    domain[i] = n0*2**i
    m = method(t0, tf, n0*2**i, x0, A, g)
    exact = vectorize_sol(t0, tf, n0*2**i, sol)
    v[i] = error(m, exact)
  return v, domain

def convergence_table(errors_2x, n0, k, t0, tf):
  '''
  RECEIVES:
  errors_2x is a array with the errors of the approximations given
  by a method with number of steps n = n0, 2*n0, 2**2*n0, ..., 2**(k-1)*n0. (np.array)
  n0 is the first number of steps. (int)
  k is the number of errors in the final array. (int)
  t0 is the initial point of the approximation. (float)
  tf is the last one. (float)
  '''
  n = n0
  print(n, (tf-t0)/n, errors_2x[0], "-", sep=" & ", end=" \\\\ \n")
  for i in range(1, k):
      n = n0 * 2 ** i
      h = (tf-t0)/n
      q = errors_2x[i-1]/errors_2x[i] #q=erro(h)/erro(h)
      r = ((tf-t0)/(n/2))/((tf-t0)/n)
      print(n, h, errors_2x[i], log(q,2)/log(r,2), sep=" & ", end=" \\\\ \n")

def convergence_table(errors_2x, n0, k, t0, tf):
  '''
  RECEIVES:
  errors_2x is a array with the errors of the approximations given
  by a method with number of steps n = n0, 2*n0, 2**2*n0, ..., 2**(k-1)*n0. (np.array)
  n0 is the first number of steps. (int)
  k is the number of errors in the final array. (int)
  t0 is the initial point of the approximation. (float)
  tf is the last one. (float)
  '''
  n = n0
  print("| n | h = $\\frac{1}{h}$ | $\\tau(0,h)$ | q = $\\frac{tau(0,h)}{tau(0, 2h)}$ |")
  print("|---|-----------------|-----------|---------------------------------|")
  print("", n, (tf-t0)/n, errors_2x[0], "-", sep=" | ", end=" | \n")
  for i in range(1, k):
      n = n0 * 2 ** i
      h = (tf-t0)/n
      q = errors_2x[i-1]/errors_2x[i] #q=erro(h)/erro(h)
      r = ((tf-t0)/(n/2))/((tf-t0)/n)
      print( "", n, h, errors_2x[i], log(q,2)/log(r,2), sep=" | ", end=" | \n")

def lmba_n_error(errors_for_lambdas_array, method, x0, Af, g, sol_given_lmba, vectorize_sol_given_lmba, error, method_name):
  lmba0 = 5
  lmbaf = 100
  n0 = 10
  nf = 128
  t0 = 0.0
  tf = 1.0
  # Create data for X, Y
  lmba_values = np.arange(lmba0, lmbaf)
  n_values = np.arange(n0, nf)
  X, Y = np.meshgrid(lmba_values, 1/n_values)
  # Create a matrix of zeros for Z
  Z = np.zeros_like(X)
  # Populate the Z matrix with data using a function
  for n in range(n0, nf):
    Z[n-n0], domain = errors_for_lambdas_array(n, method, t0, tf, x0, lmba0, lmbaf, Af, g, sol_given_lmba, vectorize_sol_given_lmba, error)
  # Create filled contour plot
  plt.contourf(X, Y, Z)
  # Add color bar for the contour plot
  plt.colorbar()
  # Add labels and title (optional)
  plt.xlabel('lambda')
  plt.ylabel('h')
  plt.title('errors for the '+method_name+' method ')
  # Show the plot
  plt.show()


# ## Convergence tables

# ### Classic Euler

# In[2]:


n0 = 128
k = 4
t0 = 0
tf = 1
x0 = np.array([1])
A = np.array([[100]])
errors_2x_vector, domain = errors_2x(n0, k, classic_euler, t0, tf, x0, A, g, sol, vectorize_sol, error_sup)
convergence_table(errors_2x_vector, n0, k, t0, tf)


# ### Exponential Euler

# In[3]:


n0 = 128
k = 4
t0 = 0
tf = 1
x0 = np.array([1])
A = np.array([[100]])
errors_2x_vector, domain = errors_2x(n0, k, exponential_euler, t0, tf, x0, A, g, sol, vectorize_sol, error_sup)
convergence_table(errors_2x_vector, n0, k, t0, tf)


# ### rk2

# In[4]:


n0 = 128
k = 4
t0 = 0
tf = 1
x0 = np.array([1])
A = np.array([[100]])

errors_2x_vector, domain = errors_2x(n0, k, rk2, t0, tf, x0, A, g, sol, vectorize_sol, error_sup)
convergence_table(errors_2x_vector, n0, k, t0, tf)


# ### etd2rk (trapezoidal)

# In[5]:


n0 = 128
k = 4
t0 = 0
tf = 1
x0 = np.array([1])
A = np.array([[100]])

errors_2x_vector, domain = errors_2x(n0, k, etd2rk, t0, tf, x0, A, g, sol, vectorize_sol, error_sup)
convergence_table(errors_2x_vector, n0, k, t0, tf)


# ### Naive version of etd2rk (trapezoidal)

# In[6]:


n0 = 128
k = 4
t0 = 0
tf = 1
x0 = np.array([1])
A = np.array([[100]])

errors_2x_vector, domain = errors_2x(n0, k, etd2rk_trapezoidal_naive, t0, tf, x0, A, g, sol, vectorize_sol, error_sup)
convergence_table(errors_2x_vector, n0, k, t0, tf)


# ### rk4

# In[7]:


n0 = 128
k = 4
t0 = 0
tf = 1
x0 = np.array([1])
A = np.array([[100]])

errors_2x_vector, domain = errors_2x(n0, k, rk4, t0, tf, x0, A, g, sol, vectorize_sol, error_sup)
convergence_table(errors_2x_vector, n0, k, t0, tf)


# ### Deduced like etd3rk

# In[8]:


n0 = 128
k = 4
t0 = 0
tf = 1
x0 = np.array([1])
A = np.array([[100]])

errors_2x_vector, domain = errors_2x(n0, k, etd3rk_similar, t0, tf, x0, A, g, sol, vectorize_sol, error_sup)
convergence_table(errors_2x_vector, n0, k, t0, tf)


# ### Naive version of etd3rk

# In[9]:


n0 = 128
k = 4
t0 = 0
tf = 1
x0 = np.array([1])
A = np.array([[100]])

errors_2x_vector, domain = errors_2x(n0, k, etd3rk_naive, t0, tf, x0, A, g, sol, vectorize_sol, error_sup)
convergence_table(errors_2x_vector, n0, k, t0, tf)


# ## Some graphics

# The following notation is used
# 
# \begin{cases}
#   u'(t) + A u(t) = g(u(t), t)\\
#   u(0) = u_0.
# \end{cases}
# 
# A Stiff problem shown in [1] is
# 
# \begin{cases}
#     u'(t) + 100 u(t) = \sin(t)\\
#     u(0) = u_0,
# \end{cases}
# 
# with solution
# 
# $$
# u(t) = u_0 \exp(-100t)+\frac{\exp(-100t)+100\sin(t)-\cos(t)}{1+100^2}.
# $$

# In[10]:


n = 128
lmba0 = 1
lmbaf = 100
t0 = 0.0
tf = 1.0
x0 = np.array([1])
lmba_1D_classic, domain = errors_for_lambdas_array(n, classic_euler, t0, tf, x0, lmba0, lmbaf, A_1D, g, sol_given_lmba, vectorize_sol_given_lmba, error_2)
lmba_1D_exponential, domain = errors_for_lambdas_array(n, exponential_euler, t0, tf, x0, lmba0, lmbaf, A_1D, g, sol_given_lmba, vectorize_sol_given_lmba, error_2)
lmba_1D_etd2rk, domain = errors_for_lambdas_array(n, etd2rk, t0, tf, x0, lmba0, lmbaf, A_1D, g, sol_given_lmba, vectorize_sol_given_lmba, error_2)
lmba_1D_etd2rk_trapezoidal_naive, domain = errors_for_lambdas_array(n, etd2rk_trapezoidal_naive, t0, tf, x0, lmba0, lmbaf, A_1D, g, sol_given_lmba, vectorize_sol_given_lmba, error_2)
lmba_1D_etd3rk_similar, domain = errors_for_lambdas_array(n, etd3rk_similar, t0, tf, x0, lmba0, lmbaf, A_1D, g, sol_given_lmba, vectorize_sol_given_lmba, error_2)
lmba_1D_etd3rk_naive, domain = errors_for_lambdas_array(n, etd3rk_naive, t0, tf, x0, lmba0, lmbaf, A_1D, g, sol_given_lmba, vectorize_sol_given_lmba, error_2)
lmba_1D_rk2, domain = errors_for_lambdas_array(n, rk2, t0, tf, x0, lmba0, lmbaf, A_1D, g, sol_given_lmba, vectorize_sol_given_lmba, error_2)
lmba_1D_rk4, domain = errors_for_lambdas_array(n, rk4, t0, tf, x0, lmba0, lmbaf, A_1D, g, sol_given_lmba, vectorize_sol_given_lmba, error_2)


# In[11]:


matrix_1D = [lmba_1D_classic, lmba_1D_exponential, lmba_1D_rk2, lmba_1D_etd2rk_trapezoidal_naive, lmba_1D_etd2rk, lmba_1D_rk4, lmba_1D_etd3rk_naive, lmba_1D_etd3rk_similar]
names = ['classic euler', 'exponential euler', 'rk2', 'etd2rk naive', 'etd2rk', 'rk4', 'etd3rk naive', "etd3rk (similar)"]
fig, ax = graphic_2D(8*[domain], matrix_1D, names, "lambda", "error", "1D problem from [1]", False, True)


# In[12]:


n0 = 10
k = 10
lmba = 100
A = lmba * np.array([[1]])
t0 = 0.0
tf = 1.0
x0 = np.array([1])
n_1D_classic, domain = errors_2x(n0, k, classic_euler, t0, tf, x0, A, g, sol, vectorize_sol, error_2)
n_1D_exponential, domain = errors_2x(n0, k, exponential_euler, t0, tf, x0, A, g, sol, vectorize_sol, error_2)
n_1D_etd2rk, domain = errors_2x(n0, k, etd2rk, t0, tf, x0, A, g, sol, vectorize_sol, error_2)
n_1D_etd2rk_trapezoidal_naive, domain = errors_2x(n0, k, etd2rk_trapezoidal_naive, t0, tf, x0, A, g, sol, vectorize_sol, error_2)
n_1D_etd3rk_similar, domain = errors_2x(n0, k, etd3rk_similar, t0, tf, x0, A, g, sol, vectorize_sol, error_2)
n_1D_etd3rk_naive, domain = errors_2x(n0, k, etd3rk_naive, t0, tf, x0, A, g, sol, vectorize_sol, error_2)
n_1D_rk2, domain = errors_2x(n0, k, rk2, t0, tf, x0, A, g, sol, vectorize_sol, error_2)
n_1D_rk4, domain = errors_2x(n0, k, rk4, t0, tf, x0, A, g, sol, vectorize_sol, error_2)


# In[13]:


matrix_2D = [n_1D_classic, n_1D_exponential, n_1D_rk2, n_1D_etd2rk_trapezoidal_naive, n_1D_etd2rk, n_1D_rk4, n_1D_etd3rk_naive, n_1D_etd3rk_similar]
names = ['classic euler', 'exponential euler', 'rk2', 'etd2rk naive', 'etd2rk', 'rk4', 'etd3rk naive', "etd3rk (similar)"]
fig_2D, ax_2D = graphic_2D(8*[1/domain], matrix_2D, names, "h", "error", "1D problem with lmba = "+str(lmba), False, True)
plt.xscale('log')


# ## Some deductions

# Here is used informations from [1], [6], [7].

# ### Exponential Euler method
# 
# For
# 
# $\begin{cases}
#     y'(t) + \lambda y(t) = g(y(t), t), t \in (t_0, T) \\
#     y(0) = y_0
# \end{cases}$
# 
# the domain is evenly discretized:
# 
# $$
#     N \in \mathbb{N}; h = \frac{T-t_0}{N}; \text{Domain: }\{t_k = t_0 + k h : k = 0, 1, ...\}.
# $$
# 
# The discretization of the ODE takes the exact solution of the Cauchy problem, given by the variation of constants formula
# 
# $$
#     y(t) = e^{-(t-t_0) \lambda}y_0 + \int_{t_0}^t [e^{-\lambda(t-\tau)} g(y(\tau), \tau)] d\tau
# $$
# 
# and, by Taylor expansion on $g$:
# 
# $\tau \in (t_k, t_{k+1})$
# 
# $$
#     g(y(\tau), \tau) = g(y(t_k), t_k) + (\tau - t_k) \frac{dg}{dt} (y(\theta_k), \theta_k)
# $$
# 
# for a $\theta_k \in (t_k, t_{k+1}),$
# 
# $$
#     y(t_{k+1}) = e^{-(t_{k+1}-t_k) \lambda}y(t_k) + \int_{t_k}^{t_{k+1}} [e^{-\lambda(t_{k+1}-\tau)} g(y(\tau), \tau)] d\tau
# $$
# 
# $$
#     = e^{-h \lambda}y(t_k) + \int_{t_k}^{t_{k+1}} \left[e^{-\lambda(t_{k+1}-\tau)} \left( g(y(t_k), t_k) + (\tau - t_k) \frac{dg}{dt} (y(\theta_k), \theta_k)\right)\right] d\tau
# $$
# 
# $$
#     = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) \int_{t_k}^{t_{k+1}} e^{-\lambda(t_{k+1}-\tau)} d\tau + \frac{dg}{dt} (y(\theta_k), \theta_k) \int_{t_k}^{t_{k+1}} (\tau - t_k) e^{-\lambda(t_{k+1}-\tau)} d\tau.
# $$
# 
# Since
# 
# $$
#     \int_{t_k}^{t_{k+1}} e^{-\lambda(t_{k+1}-\tau)} d\tau = h\phi_1(-\lambda h)= \frac{1-e^{-h \lambda}}{\lambda}
# $$
# 
# and, by the Taylor expansion of $e^{-\lambda h}$ in the point zero
# 
# $$
#     e^{-\lambda h} = 1 - \lambda h + \frac{1}{2}\lambda^2h^2 - \frac{1}{3!}\lambda^3h^3 + \dotsi + \frac{1}{n!} (-\lambda h)^n + \dotsi, n \in \mathbb{N}
# $$
# 
# $$
#      \int_{t_k}^{t_{k+1}} (\tau - t_k) e^{-\lambda(t_{k+1}-\tau)} d\tau =
#      h^2 \phi_2 (-\lambda h) =
#      h \frac{\phi_1(0) - \phi_1(-\lambda h)}{\lambda} =
#      \frac{h}{\lambda} - \frac{1-e^{-h \lambda}}{\lambda^2} = \\
#      \frac{h}{\lambda} - \frac{1-(1 - \lambda h + \frac{1}{2}\lambda^2h^2 - \frac{1}{3!}\lambda^3h^3 + \dotsi + \frac{1}{n!} (-\lambda h)^n + \dotsi)}{\lambda^2} = \\
#      \frac{h^2}{2} - \frac{h^3}{3!} \lambda + \dotsi + \frac{h^n}{n!} (-\lambda)^{n-2} + \dotsi  =  O(h^2),
# $$
# 
# $$
#     y(t_{k+1}) = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) \frac{1-e^{-h \lambda}}{\lambda} + \frac{dg}{dt} (y(\theta_k), \theta_k) O(h^2),
# $$
# 
# $$
#   y(t_{k+1}) = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) \frac{1-e^{-h \lambda}}{\lambda} + O(h^2).
# $$
# 
# That inspires the $\textbf{Exponential Euler method}$ :
# 
# $$
# y_0 = y(t_0)\\
# \textbf{for } k = 0, 1, 2, ..., N-1 :\\
#     y_{k+1} = e^{-h \lambda}y_k + g(y_k, t_k) \frac{1-e^{-h \lambda}}{\lambda}\\
#     t_{k+1} = t_k + h
# $$
# 
# with $y_k \thickapprox y(t_k)$.

# ### Exponential time differencing methods (ETD)
# 
# In the same conditions as above, it is taken a general Taylor expansion of $g$:
# 
# $\tau \in (t_k, t_{k+1}), n \in \mathbb{N}$
# 
# $$
#     g(y(\tau), \tau) = g(y(t_k), t_k) + (\tau - t_k) \frac{dg}{dt} (y(t_k), t_k) + \frac{(\tau - t_k)^2}{2!} \frac{d^2g}{dt^2} (y(t_k), t_k) + \\
#     \dotsi + \frac{(\tau - t_k)^{n-1}}{(n-1)!} \frac{d^{n-1}g}{dt^{n-1}} (y(t_k), t_k) + \frac{(\tau - t_k)^n}{n!} \frac{d^ng}{dt^n} (y(\theta_k), \theta_k)
# $$
# 
# for a $\theta_k \in (t_k, t_{k+1})$
# 
# In
# 
# $$
#     y(t_{k+1}) = e^{-h \lambda}y(t_k) + \int_{t_k}^{t_{k+1}} e^{-\lambda(t_{k+1}-\tau)} g(y(\tau), \tau) d\tau
# $$
# 
# It will now become
# 
# $$
# y(t_{k+1}) = e^{-h \lambda}y(t_k) + \int_{t_k}^{t_{k+1}} e^{-\lambda(t_{k+1}-\tau)}  g(y(t_k), t_k) +
# (\tau - t_k) \frac{dg}{dt} (y(t_k), t_k) +
# $$
# 
# $$
#  \frac{(\tau - t_k)^2}{2!} \frac{d^2g}{dt^2} (y(t_k), t_k) + \dotsi + 
# $$
# 
# $$
#  + \frac{(\tau - t_k)^{n-1}}{(n-1)!} \frac{d^{n-1}g}{dt^{n-1}} (y(t_k), t_k) + \frac{(\tau - t_k)^n}{n!} \frac{d^ng}{dt^n} (y(\theta_k), \theta_k)  d\tau,
# $$
# 
# $$
# y(t_{k+1}) = e^{-h \lambda}y(t_k) +
# g(y(t_k), t_k)\int_{t_k}^{t_{k+1}} e^{-\lambda(t_{k+1}-\tau)} d \tau +
# $$
# 
# $$
# + \frac{dg}{dt}(y(t_k), t_k)\int_{t_k}^{t_{k+1}} e^{-\lambda(t_{k+1}-\tau)} (\tau - t_k)d\tau + \frac{d^2g}{dt^2} (y(t_k), t_k)\int_{t_k}^{t_{k+1}} e^{-\lambda(t_{k+1}-\tau)} \frac{(\tau - t_k)^2}{2!}d\tau +
# $$
# 
# $$
# + \dotsi +
# $$
# 
# $$
# + \frac{d^{n-1}g}{dt^{n-1}} (y(t_k), t_k)\int_{t_k}^{t_{k+1}}  e^{-\lambda(t_{k+1}-\tau)} \frac{(\tau - t_k)^{n-1}}{(n-1)!} d\tau + \frac{d^ng}{dt^n} (y(\theta_k), \theta_k) \int_{t_k}^{t_{k+1}}  e^{-\lambda(t_{k+1}-\tau)} \frac{(\tau - t_k)^n}{n!} d\tau,
# $$
# 
# $$
# y(t_{k+1}) = e^{-h \lambda}y(t_k) +
# h\phi_1(-\lambda h) g(y(t_k), t_k) +
# h^2\phi_2(-\lambda h) \frac{dg}{dt}(y(t_k), t_k) +
# h^3\phi_3(-\lambda h)\frac{d^2g}{dt^2} (y(t_k), t_k)
# $$
# 
# $$
# + \dotsi +
# $$
# 
# $$
# + h^n\phi_n(-\lambda h) \frac{d^{n-1}g}{dt^{n-1}} (y(t_k), t_k)+
# h^{n+1}\phi_{n+1}(-\lambda h) \frac{d^ng}{dt^n} (y(\theta_k), \theta_k).
# $$
# 
# From the discussion about the exponential Euler, that is known that
# 
# $$
# h^2\phi_2(-\lambda h) = \frac{h^2}{2} - \frac{h^3}{3!} \lambda + \dotsi + \frac{h^l}{l!} (-\lambda)^{l-2} + \dotsi = \frac{1}{(-\lambda)^2} \sum\limits_{i=2}^{\infty} \frac{(-\lambda h)^i}{i!}.
# $$
# 
# Since
# 
# $$
#   \phi_{n+1}(-\lambda h) = \frac{\phi_n(-\lambda h) - \phi_n(0)}{-\lambda h} \text{ and}\\
#   \phi_n(0) = \frac{1}{n!},
# $$
# 
# $$
#   h^3 \phi_3(-\lambda h) = h^2 \frac{\phi_2(0) - \phi_2(-\lambda h)}{\lambda} = \frac{\frac{h^2}{2} - (\frac{h^2}{2} - \frac{h^3}{3!} \lambda + \dotsi + \frac{h^l}{l!} (-\lambda)^{l-2} + O(h^{l+1}))}{\lambda} = \frac{1}{(-\lambda)^3} \sum\limits_{i=3}^{\infty} \frac{(-\lambda h)^i}{i!}.
# $$
# 
# And if
# 
# $$
# h^l \phi_l(-\lambda h) = \frac{1}{(-\lambda)^l} \sum\limits_{i=l}^{\infty} \frac{(-\lambda h)^i}{i!}, \text{for a } l \in \mathbb{N},
# $$
# 
# $$
#   h^{l+1}\phi_{l+1}(-\lambda h) = h^{l+1} \frac{\phi_l(-\lambda h) - \phi_l(0)}{-\lambda h} = \frac{h^l \phi_l(0) - h^l \phi_l(-\lambda h)}{\lambda} = \frac{h^l}{l! \lambda} - \frac{1}{\lambda} \frac{1}{(-\lambda)^l} \sum\limits_{i=l}^{\infty} \frac{(-\lambda h)^i}{i!} = \frac{1}{(-\lambda)^{l+1}} \sum\limits_{i=l+1}^{\infty} \frac{(-\lambda h)^i}{i!}.
# $$
# 
# So, by induction,
# 
# $$
# h^n \phi_n(-\lambda h) = \frac{1}{(-\lambda)^n} \sum\limits_{i=n}^{\infty} \frac{(-\lambda h)^i}{i!} = O(h^n), \forall n \geq 2.
# $$
# 
# Then,
# 
# $$
# y(t_{k+1}) = e^{-h \lambda}y(t_k) +
# h\phi_1(-\lambda h) g(y(t_k), t_k) +
# h^2\phi_2(-\lambda h) \frac{dg}{dt}(y(t_k), t_k) +
# h^3\phi_3(-\lambda h)\frac{d^2g}{dt^2} (y(t_k), t_k) +
# \dotsi + \\
# h^n\phi_n(-\lambda h) \frac{d^{n-1}g}{dt^{n-1}} (y(t_k), t_k)+
# O(h^{n+1}).
# $$

# ### Exponential time differencing methods with Runge-Kutta time stepping - order 2 - Cox and Matthews - Trapezoidal rule
# 
# For the second order method, that is used the approximation
# 
# $$
#     g(y(\tau), \tau) = g(y(t_k), t_k) + (\tau - t_k) \frac{dg}{dt} (y(t_k), t_k) + O(h^2),
# $$
# 
# $\forall \tau \in (t_k, t_{k+1}).$
# 
# The first derivative is discretized with the Taylor expansion
# 
# $$
# g(y(t_{k+1}), t_{k+1}) = g(y(t_k), t_k) + h \frac{dg}{dt} (y(t_k), t_k) + O(h^2)
# $$
# 
# and the exponential Euler expression
# 
# $$
#   y(t_{k+1}) = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) \frac{1-e^{-h \lambda}}{\lambda} + O(h^2),
# $$
# 
# so that
# 
# $$
# \frac{dg}{dt} (y(t_k), t_k)  = \frac{g(a_k, t_{k+1}) - g(y(t_k), t_k)}{h} + O(h), \\
# \text{with } a_k = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) \frac{1-e^{-h\lambda}}{\lambda},
# $$
# 
# which results in the expression
# 
# $$
# g(y(\tau), \tau) = g(y(t_k), t_k) + (\tau - t_k) \frac{g(a_k, t_{k+1}) - g(y(t_k), t_k)}{h} + (\tau - t_k)O(h) \\
# \text{with } a_k = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) \frac{1-e^{-h\lambda}}{\lambda}.
# $$
# 
# Putting in the variation of constants formula
# 
# $$
# y(t) = e^{-(t-t_0) \lambda}y_0 + \int_{t_0}^t e^{-\lambda(t-\tau)} g(y(\tau), \tau) d\tau,
# $$
# 
# $$
#   y(t_{k+1}) = e^{-h \lambda}y(t_k) + \\
#   + \int_{t_k}^{t_{k+1}} e^{-\lambda(t_{k+1}-\tau)} \left[ g(y(t_k), t_k) + (\tau - t_k)  \frac{g(a_k, t_{k+1}) - g(y(t_k), t_k)}{h}  + (\tau - t_k)O(h) \right] d\tau
# $$
# 
# $$
# y(t_{k+1}) = e^{-h \lambda} y(t_k) + g(y(t_k), t_k)\int_{t_k}^{t_{k+1}} e^{-\lambda(t_{k+1}-\tau)} d \tau + \frac{g(a_k, t_{k+1}) - g(y(t_k), t_k)}{h} \int_{t_k}^{t_{k+1}} (\tau - t_k) e^{-\lambda(t_{k+1}-\tau)} d \tau + \\
# + O(h)\int_{t_k}^{t_{k+1}} (\tau - t_k) e^{-\lambda(t_{k+1}-\tau)} d \tau \\
# \text{with } a_k = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) \frac{1-e^{-h\lambda}}{\lambda}.
# $$
# 
# Then,
# 
# $$
# y(t_{k+1}) = e^{-h \lambda} y(t_k) +
#   h \phi_1 (-\lambda h) g(y(t_k), t_k) +
#   \frac{g(a_k, t_{k+1}) - g(y(t_k), t_k)}{h} h^2 \phi_2 (-\lambda h) + \\
#   + O(h)h^2 \phi_2 (-\lambda h) \\
#   y(t_{k+1}) = e^{-h \lambda} y(t_k) +
#   h \phi_1 (-\lambda h) g(y(t_k), t_k) +
#   \left[g(a_k, t_{k+1}) - g(y(t_k), t_k) \right] h \phi_2 (-\lambda h) + \\
#   + O(h^3) \\
#   \text{with } a_k = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) \frac{1-e^{-h\lambda}}{\lambda}.
# $$
# 
# Butcher tableau:
# 
# $$
# \begin{array}
# {c|cc}
# 0\\
# 1 & \phi_1(-\lambda h)\\
# \hline
# & \phi_1 (-\lambda h) - \phi_2 (-\lambda h) & \phi_2 (-\lambda h)
# \end{array}
# $$

# ### Exponential time differencing methods with Runge-Kutta time stepping - order 2 - Cox and Matthews - Midpoint rule
# 
# From the same expression:
# 
# $$
#     g(y(\tau), \tau) = g(y(t_k), t_k) + (\tau - t_k) \frac{dg}{dt} (y(t_k), t_k) + O(h^2),
# $$
# 
# $\forall \tau \in (t_k, t_{k+1}).$
# 
# The first derivative is now discretized with the Taylor expansion
# 
# $$
# g\left(y\left(t_k + \frac{h}{2}\right), t_k + \frac{h}{2} \right) = g(y(t_k), t_k) + \frac{h}{2} \frac{dg}{dt} (y(t_k), t_k) + O(h^2)
# $$
# 
# and the exponential Euler expression taken is with time step $\frac{h}{2}$
#  
# $$
#   y\left(t_k + \frac{h}{2}\right) = e^{-\frac{h \lambda}{2}}y(t_k) + g(y(t_k), t_k) \frac{h}{2} \phi_1\left( -\frac{\lambda h}{2} \right) + O(h^2),
# $$
# 
# so that
# 
# $$
# \frac{dg}{dt} (y(t_k), t_k)  = 2 \frac{g\left(b_k, t_k + \frac{h}{2} \right) - g(y(t_k), t_k)}{h} + O(h), \\
# \text{with } b_k =e^{-\frac{h \lambda}{2}}y(t_k) + g(y(t_k), t_k) \frac{h}{2} \phi_1\left( -\frac{\lambda h}{2} \right),
# $$
# 
# which results in the expression
# 
# $$
# g(y(\tau), \tau) = g(y(t_k), t_k) + 2(\tau - t_k) \frac{g\left(b_k, t_k + \frac{h}{2}\right) - g(y(t_k), t_k)}{h} + (\tau - t_k)O(h) \\
# \text{with } b_k =e^{-\frac{h \lambda}{2}}y(t_k) + g(y(t_k), t_k) \frac{h}{2} \phi_1\left( -\frac{\lambda h}{2} \right).
# $$
# 
# Putting in the variation of constants formula
# 
# $$
# y(t) = e^{-(t-t_0) \lambda}y_0 + \int_{t_0}^t e^{-\lambda(t-\tau)} g(y(\tau), \tau) d\tau,
# $$
# 
# $$
#   y(t_{k+1}) = e^{-h \lambda}y(t_k) + \\
#     + \int_{t_k}^{t_{k+1}} e^{-\lambda(t_{k+1}-\tau)} \left[ g(y(t_k), t_k) + 2(\tau - t_k) \frac{g\left(b_k, t_k + \frac{h}{2}\right) - g(y(t_k), t_k)}{h} + (\tau - t_k)O(h) \right] d\tau
# $$
# 
# $$
# y(t_{k+1}) = e^{-h \lambda} y(t_k) + g(y(t_k), t_k)\int_{t_k}^{t_{k+1}} e^{-\lambda(t_{k+1}-\tau)} d \tau + \\
#   + 2\frac{g\left(b_k, t_k + \frac{h}{2}\right) - g(y(t_k), t_k)}{h} \int_{t_k}^{t_{k+1}} (\tau - t_k) e^{-\lambda(t_{k+1}-\tau)} d \tau + O(h)\int_{t_k}^{t_{k+1}} (\tau - t_k) e^{-\lambda(t_{k+1}-\tau)} d \tau \\
#   \text{with } b_k =e^{-\frac{h \lambda}{2}}y(t_k) + g(y(t_k), t_k) \frac{h}{2} \phi_1\left( -\frac{\lambda h}{2} \right).
# $$
# 
# Then,
# 
# $$
# y(t_{k+1}) = e^{-h \lambda} y(t_k) +
#   h \phi_1 (-\lambda h) g(y(t_k), t_k) +
#   2\frac{g\left(b_k, t_k + \frac{h}{2}\right) - g(y(t_k), t_k)}{h} h^2 \phi_2 (-\lambda h) + \\
#   + O(h)h^2 \phi_2 (-\lambda h) \\
#   y(t_{k+1}) = e^{-h \lambda} y(t_k) +
#   h \phi_1 (-\lambda h) g(y(t_k), t_k) +
#   2 \left[g\left(b_k, t_k + \frac{h}{2}\right) - g(y(t_k), t_k) \right] h \phi_2 (-\lambda h) + \\
#   + O(h^3) \\
#   \text{with } b_k =e^{-\frac{h \lambda}{2}}y(t_k) + g(y(t_k), t_k) \frac{h}{2} \phi_1\left( -\frac{\lambda h}{2} \right).
# $$
# 
# Butcher tableau:
# 
# \begin{array}
# {c|cc}
# 0\\
# \frac{1}{2} & \frac{1}{2}\phi_1\left(-\frac{\lambda h}{2}\right)\\
# \hline
# & \phi_1 (-\lambda h) - 2 \phi_2 (-\lambda h) & -2 \phi_2 (-\lambda h)
# \end{array}

# ### Exponential time differencing methods with Runge-Kutta time stepping - order 2 - Classical approach - Trapezoidal rule
# 
# It is also possible to think the exponential time differencing methods with Runge-Kutta time stepping using the numerical integration, for example, for the one with second order, it starts with the trapezoidal rule (which was taken from [2]) on the variation of constants formula:
# 
# $$
# y(t_{k+1}) = e^{-h \lambda}y(t_k) + \frac{h}{2} \left[ e^{-\lambda(t_{k+1}-t_k)} g(y(t_k), t_k) + e^{-\lambda(t_{k+1}-t_{k+1})} g(y(t_{k+1}), t_{k+1}) \right] + O(h^3), \\
#     y(t_{k+1}) = e^{-h \lambda}y(t_k) + \frac{h}{2} \left[ e^{-\lambda h} g(y(t_k), t_k) + g(y(t_{k+1}), t_{k+1}) \right] +  O(h^3).
# $$
# 
# And then, from the expression seen before:
# 
# $$
# y(t_{k+1}) = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) \frac{1-e^{-h \lambda}}{\lambda} + O(h^2),
# $$
# 
# $$
#     g(y(t_{k+1}), t_{k+1}) = g(a_k, t_{k+1}) + O(h^2) \text{, with } a_k = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) \frac{1-e^{-h\lambda}}{\lambda}.
# $$
# 
# So,
# 
# $$
# y(t_{k+1}) = e^{-h \lambda}y(t_k) + \frac{h}{2} \left[ e^{-\lambda h} g(y(t_k), t_k) + g(a_k, t_{k+1}) + O(h^2) \right] +  O(h^3),
# $$
# 
# $$
# y(t_{k+1}) = e^{-h \lambda}y(t_k) + \frac{h}{2} \left[ e^{-\lambda h} g(y(t_k), t_k) + g(a_k, t_{k+1}) \right] +  O(h^3) \\
#     \text{with } a_k = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) h \phi_1 (-\lambda h).
# $$
# 
# Butcher tableau:
# 
# $$
# \begin{array}
# {c|cc}
# 0\\
# 1 & \phi_1(-\lambda h)\\
# \hline
# & \frac{1}{2} e^{-h \lambda} & \frac{1}{2}
# \end{array}
# $$

# ### Exponential time differencing methods with Runge-Kutta time stepping - order 2 - Classical approach - Midpoint rule
# 
# Besides that, using the midpoint rule, also known as rectangle rule, again taken from [2],
# 
# $$
# y(t_{k+1}) = e^{-(t_{k+1}-t_k) \lambda}y(t_k) + \int_{t_k}^{t_{k+1}} [e^{-\lambda(t_{k+1}-\tau)} g(y(\tau), \tau)] d\tau,
# $$
# 
# $$
# y(t_{k+1}) = e^{-h\lambda}y(t_k) + h e^{-\lambda\left(t_{k+1}-\frac{t_{k+1}+t_k}{2}\right)} g\left(y\left(\frac{t_{k+1}+t_k}{2}\right), \frac{t_{k+1}+t_k}{2}\right) + O(h^3),
# $$
# 
# $$
# y(t_{k+1}) = e^{-h\lambda}y(t_k) + h e^{- \frac{h\lambda}{2}} g\left(y\left(t_k + \frac{h}{2}\right), t_k+\frac{h}{2}\right) + O(h^3),
# $$
# 
# and Exponential Euler with time step $\frac{h}{2}$
# 
# $$
# y\left(t_k + \frac{h}{2}\right) = e^{-\frac{h \lambda}{2}}y(t_k) + g(y(t_k), t_k) \frac{h}{2} \phi_1\left( -\frac{\lambda h}{2} \right) + O(h^2),
# $$
# 
# results in
# 
# $$
# y(t_{k+1}) = e^{-h\lambda}y(t_k) + h e^{- \frac{h\lambda}{2}} g\left(b_k + O(h^2), t_k + \frac{h}{2}\right) + O(h^3) \\
#     \text{with } b_k =e^{-\frac{h \lambda}{2}}y(t_k) + g(y(t_k), t_k) \frac{h}{2} \phi_1\left( -\frac{\lambda h}{2} \right),
# $$
# 
# $$
# y(t_{k+1}) = e^{-h\lambda}y(t_k) + h e^{- \frac{h\lambda}{2}} g\left(b_k , t_k + \frac{h}{2}\right) + O(h^3) \\
#     \text{with } b_k =e^{-\frac{h \lambda}{2}}y(t_k) + g(y(t_k), t_k) \frac{h}{2} \phi_1\left( -\frac{\lambda h}{2} \right),
# $$
# 
# Butcher tableau:
# 
# $$
# \begin{array}
# {c|cc}
# 0\\
# \frac{1}{2} &  \frac{1}{2} \phi_1( -\frac{\lambda h}{2})\\
# \hline
# & 0 & e^{-\frac{h \lambda}{2}}
# \end{array}
# $$

# ### Third order exponential time differencing methods with Runge-Kutta time stepping (ETDRK-3)
# 
# $$
#     g(y(\tau), \tau) = g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) +
#     \left(\tau - t_{k+\frac{1}{2}}\right) \frac{dg}{dt} \left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) + \\
#     + \frac{\left(\tau - t_{k+\frac{1}{2}}\right)^2}{2!} \frac{d^2g}{dt^2} \left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) + \\
#     + \frac{\left(\tau - t_{k+\frac{1}{2}}\right)^3}{3!} \frac{d^3g}{dt^3} \left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right)
#     + O((\tau - t_k)^4),
# $$
# 
# $\forall \tau \in \mathbb{R}.$
# 
# $$
#     g(y(t_{k+1}), t_{k+1}) = g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) +
#     \left(t_{k+1} - t_{k+\frac{1}{2}}\right) \frac{dg}{dt} \left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) + \\
#     + \frac{\left(t_{k+1} - t_{k+\frac{1}{2}}\right)^2}{2!} \frac{d^2g}{dt^2} \left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) + \\
#     + \frac{\left(t_{k+1} - t_{k+\frac{1}{2}}\right)^3}{3!} \frac{d^3g}{dt^3} \left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right)
#     + O(h^4),
# $$
# 
# $$
#     g(y(t_{k+1}), t_{k+1}) = g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) +
#     \frac{h}{2} \frac{dg}{dt} \left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) + \\
#     +\frac{h^2}{8} \frac{d^2g}{dt^2} \left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right)
#     + \frac{h^3}{48} \frac{d^3g}{dt^3} \left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right)
#     + O(h^4),
# $$
# 
# and
# 
# $$
#     g(y(t_k), t_k) = g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) +
#     \left(t_k - t_{k+\frac{1}{2}}\right) \frac{dg}{dt} \left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) + \\
#     + \frac{\left(t_k - t_{k+\frac{1}{2}}\right)^2}{2!} \frac{d^2g}{dt^2} \left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) + \\
#     + \frac{\left(t_k - t_{k+\frac{1}{2}}\right)^3}{3!} \frac{d^3g}{dt^3} \left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right)
#     + O(h^4),
# $$
# 
# $$
#     g(y(t_k), t_k) = g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) -
#     \frac{h}{2} \frac{dg}{dt} \left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) + \\
#     + \frac{h^2}{8} \frac{d^2g}{dt^2} \left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right)
#     - \frac{h^3}{48} \frac{d^3g}{dt^3} \left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right)
#     + O(h^4).
# $$
# 
# Subtracting the two expressions,
# 
# $$
#   g(y(t_{k+1}), t_{k+1}) - g(y(t_k), t_k) = h \frac{dg}{dt} \left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) + O(h^3).
# $$
# 
# So,
# 
# $$
#   \frac{dg}{dt} \left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) = \frac{g(y(t_{k+1}), t_{k+1}) - g(y(t_k), t_k)}{h} + O(h^2).
# $$
# 
# And summing them
# 
# $$
#   g(y(t_{k+1}), t_{k+1}) + g(y(t_k), t_k) =
#   2 g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right)
#   + \frac{h^2}{4} \frac{d^2g}{dt^2} \left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) + O(h^4).
# $$
# 
# So,
# 
# $$
#   \frac{d^2g}{dt^2} \left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) =
#   4\frac{ g(y(t_{k+1}), t_{k+1}) + g(y(t_k), t_k) -
#   2 g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right)}{h^2}
#   + O(h^2).
# $$
# 
# This results in the expression
# 
# $$
#     g(y(\tau), \tau) = g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) +
#     \left(\tau - t_{k+\frac{1}{2}}\right)  \frac{g(y(t_{k+1}), t_{k+1}) - g(y(t_k), t_k)}{h} + \\
#     + \frac{\left(\tau - t_{k+\frac{1}{2}}\right)^2}{2!} \frac{ 4 \left[g(y(t_{k+1}), t_{k+1}) + g(y(t_k), t_k) -
#   2 g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) \right]}{h^2} + \\
#     + O((\tau - t_k)^3),
# $$
# 
# $\forall \tau \in \mathbb{R}.$
# 
# Putting in the variation of constants formula
# 
# $$
# y(t) = e^{-(t-t_0) \lambda}y_0 + \int_{t_0}^t e^{-\lambda(t-\tau)} g(y(\tau), \tau) d\tau,
# $$
# 
# $$
#   y(t_{k+1}) = e^{-h \lambda}y(t_k) + \\
#   + \int_{t_k}^{t_{k+1}} e^{-\lambda(t_{k+1}-\tau)} \left[ g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) +
#     \left(\tau - t_{k+\frac{1}{2}}\right)  \frac{g(y(t_{k+1}), t_{k+1}) - g(y(t_k), t_k)}{h} + \\
#     + \frac{\left(\tau - t_{k+\frac{1}{2}}\right)^2}{2!} \frac{ 4 \left[g(y(t_{k+1}), t_{k+1}) + g(y(t_k), t_k) -
#   2 g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) \right]}{h^2} + O((\tau - t_k)^3) \right] d\tau,
# $$
# 
# $$
#   y(t_{k+1}) = e^{-h \lambda} y(t_k) +
#   \\
#   g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right)
#   \int_{t_k}^{t_{k+1}} e^{-\lambda(t_{k+1}-\tau)} d \tau +
#   \\
#   \frac{g(y(t_{k+1}), t_{k+1}) - g(y(t_k), t_k)}{h}
#   \int_{t_k}^{t_{k+1}} \left(\tau - t_{k+\frac{1}{2}}\right) e^{-\lambda(t_{k+1}-\tau)} d \tau +
#   \\
#   + \frac{ 4 \left[g(y(t_{k+1}), t_{k+1}) + 
#   \\
#   g(y(t_k), t_k) - 2 g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) \right]}{h^2}
#   \int_{t_k}^{t_{k+1}} \frac{\left(\tau - t_{k+\frac{1}{2}}\right)^2}{2!} e^{-\lambda(t_{k+1}-\tau)} d \tau
#   \\
#   + \int_{t_k}^{t_{k+1}} O((\tau - t_k)^3) e^{-\lambda(t_{k+1}-\tau)} d \tau,
# $$
# 
# Since $\tau - t_{k+ \frac{1}{2}} = \tau - t_k - \frac{h}{2}$ and $\left(\tau - t_{k+ \frac{1}{2}} \right)^2 = (\tau - t_k)^2 + \frac{h^2}{4} - h (\tau - t_k)$,
# 
# $$
#   y(t_{k+1}) = e^{-h \lambda} y(t_k) +
#   g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right)
#   \int_{t_k}^{t_{k+1}} e^{-\lambda(t_{k+1}-\tau)} d \tau +
#   \\
#   \frac{g(y(t_{k+1}), t_{k+1}) - g(y(t_k), t_k)}{h}
#   \int_{t_k}^{t_{k+1}} \left(\tau - t_{k}\right) e^{-\lambda(t_{k+1}-\tau)} d \tau +
#   \\
#   + \frac{ 4 \left[g(y(t_{k+1}), t_{k+1}) + 
#   \\
#   g(y(t_k), t_k) - 2 g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) \right]}{h^2}
#   \int_{t_k}^{t_{k+1}} \frac{(\tau - t_k)^2}{2!} e^{-\lambda(t_{k+1}-\tau)} d \tau +
#   \\
#   - \frac{g(y(t_{k+1}), t_{k+1}) - g(y(t_k), t_k)}{h}
#   \int_{t_k}^{t_{k+1}} \frac{h}{2} e^{-\lambda(t_{k+1}-\tau)} d \tau +
#   \\
#   + \frac{ 4 \left[g(y(t_{k+1}), t_{k+1}) + g(y(t_k), t_k) - 2 g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) \right]}{h^2}
#   \int_{t_k}^{t_{k+1}} \frac{h^2}{4 \cdot 2!} e^{-\lambda(t_{k+1}-\tau)} d \tau +
#   \\
#   - \frac{ 4 \left[g(y(t_{k+1}), t_{k+1}) + g(y(t_k), t_k) - 2 g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) \right]}{h^2}
#   \int_{t_k}^{t_{k+1}} \frac{h (\tau - t_k)}{2!} e^{-\lambda(t_{k+1}-\tau)} d \tau +
#   \\
#   + \int_{t_k}^{t_{k+1}} O((\tau - t_k)^3) e^{-\lambda(t_{k+1}-\tau)} d \tau.
# $$
# 
# Then,
# 
# $$
#   y(t_{k+1}) = e^{-h \lambda} y(t_k) +
#   \\
#   g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right)
#   h \phi_1(-h \lambda) +
#   \\
#   \frac{g(y(t_{k+1}), t_{k+1}) - g(y(t_k), t_k)}{h}
#   h^2 \phi_2 (-h \lambda) +
#   \\
#   + \frac{ 4 \left[g(y(t_{k+1}), t_{k+1}) + g(y(t_k), t_k) - 2 g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) \right]}{h^2}
#   h^3 \phi_3 (-h \lambda) +
#   \\
#   - \frac{g(y(t_{k+1}), t_{k+1}) - g(y(t_k), t_k)}{h}
#   \frac{h^2 \phi_1(-h \lambda)}{2} +
#   \\
#   + \frac{ 4 \left[g(y(t_{k+1}), t_{k+1}) + g(y(t_k), t_k) - 2 g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) \right]}{h^2}
#   \frac{h^3 \phi_1(-h \lambda)}{8} +
#   \\
#   - \frac{ 4 \left[g(y(t_{k+1}), t_{k+1}) + g(y(t_k), t_k) - 2 g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) \right]}{h^2}
#   \frac{h^3 \phi_2(-h \lambda)}{2} +
#   \\
#   + O(h^4 \phi_4(-h \lambda)).
# $$
# 
# i.e.
# 
# $$
#   y(t_{k+1}) = e^{-h \lambda} y(t_k) +
#   \\
#   g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right)
#   h \phi_1(-h \lambda) + %ok
#   \\
#   \left[g(y(t_{k+1}), t_{k+1}) - g(y(t_k), t_k)\right]
#   \left( h \phi_2 (-h \lambda) - \frac{h \phi_1(-h \lambda)}{2} \right) +
#   \\
#   + 4 \left[g(y(t_{k+1}), t_{k+1}) + 
#   \\
#   g(y(t_k), t_k) - 2 g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) \right]
#   \left( h \phi_3 (-h \lambda) + \frac{h \phi_1(-h \lambda)}{8} - \frac{h \phi_2(-h \lambda)}{2} \right) + O(h^4).
# $$
# 
# Using the Cox and Mathhews's ETDRK-2 expressions to approximate $y\left(t_{k+\frac{1}{2}}\right)$ and $y(t_{k+1})$, since those are of order 2, i.e., $O(h^3)$, the expression of the method is
# 
# $$
#   y(t_{k+1}) = e^{-h \lambda} y(t_k) +
#   \\
#   g\left(c'_k, t_{k+\frac{1}{2}}\right)
#   h \phi_1(-h \lambda) + %ok
#   \\
#   \left[g(c_k, t_{k+1}) - g(y(t_k), t_k)\right]
#   \left( h \phi_2 (-h \lambda) - \frac{h \phi_1(-h \lambda)}{2} \right) +
#   \\
#   + 4 \left[g(c_k, t_{k+1}) + g(y(t_k), t_k) - 2 g\left(c'_k, t_{k+\frac{1}{2}}\right) \right]
#   \left( h \phi_3 (-h \lambda) + \frac{h \phi_1(-h \lambda)}{8} - \frac{h \phi_2(-h \lambda)}{2} \right) + O(h^4),
# $$
# 
# with
# 
# $$
#   c_k = e^{-h \lambda} y(t_k) +
#   \\
#   h \phi_1 (-\lambda h) g(y(t_k), t_k) +
#   \\
#   \left[g(a_k, t_{k+1}) - g(y(t_k), t_k) \right] h \phi_2 (-\lambda h),
#   \\
#   a_k = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) h \phi_1(-h\lambda),
#   \\
#   c'_k = e^{- \frac{h \lambda}{2}} y(t_k) +
#   \\
#   \frac{h}{2} \phi_1 \left(- \frac{\lambda h}{2} \right) g(y(t_k), t_k) +
#   \\
#   \left[g\left(a'_k, t_{k+\frac{1}{2}}\right) - g(y(t_k), t_k) \right] \frac{h}{2} \phi_2 \left(-\frac{\lambda h}{2}\right),
#   \\
#   a'_k = e^{-\frac{h \lambda}{2}}y(t_k) + g(y(t_k), t_k) \frac{h}{2} \phi_1\left(-\frac{h \lambda}{2}\right).
# $$
# 
# Here de deducing isn't exactly in Runge Kutta form, differing on the approximations for steps in minor order, so it cannot be a Butcher tableau, but doing language abuse only on the part that would form the triangle, it would be:
# 
# \begin{array}
# {c|cccc}
# 0 \\
# \frac{1}{2} & \frac{1}{2} \left( \phi_1\left(- \frac{\lambda h}{2} \right) - \phi_2\left(- \frac{\lambda h}{2} \right) \right) & \frac{1}{2}\phi_2\left(- \frac{\lambda h}{2} \right) \\
# 1 & \phi_1\left(- \lambda h \right) - \phi_2\left(- \lambda h \right) & 0 & \phi_2\left(- \lambda h \right)  \\
# \hline
# & 4 \phi_3(-h \lambda)-3\phi_2(-h\lambda)+\phi_1(-h\lambda) & -8\phi_3(-h\lambda)+4\phi_2(-h\lambda) & 4 \phi_3(-h\lambda)-\phi_2(-h\lambda) \text{   }.
# \end{array}

# ### Naive etd3rk
# 
# Here, that is taken the variation of constants formula:
# 
# $$
#     y(t_{k+1}) = e^{-h \lambda}y(t_k) + \int_{t_k}^{t_{k+1}} e^{-\lambda(t_{k+1}-\tau)} g(y(\tau), \tau) d\tau,
# $$
# 
# and applied the Simpson's rule (here was used the order of convergence from Burden) so that it will be:
# 
# $$
#     y(t_{k+1}) = e^{-h \lambda}y(t_k) + 
#     \\
#     \frac{h}{6} \left[ e^{-\lambda(t_{k+1}-t_k)} g(y(t_k), t_k) + 4 e^{-\lambda \left(t_{k+1}-t_{k + \frac{1}{2}} \right)} g\left(y\left(t_{k+\frac{1}{2}}\right), t_k + \frac{h}{2} \right) \\ + e^{-\lambda(t_{k+1}-t_{k+1})} g(y(t_{k+1}), t_{k+1}) \right] 
#     \\
#     + O(h^5), \\
#     y(t_{k+1}) = e^{-h \lambda}y(t_k) + 
#     \\
#     \frac{h}{6} \left[ e^{-\lambda h} g(y(t_k), t_k) + 4 e^{-\frac{ \lambda h}{2}} g\left(y\left(t_k + \frac{h}{2} \right), t_k + \frac{h}{2} \right) + g(y(t_{k+1}), t_{k+1}) \right] 
#     \\
#     +  O(h^5).
# $$
# 
# To approximate $y\left(t_k + \frac{h}{2} \right)$:
# 
# $$
#     y\left(t_k + \frac{h}{2} \right) = e^{- \frac{h \lambda}{2}}y(t_k) + \frac{h}{4} \left[ e^{- \frac{h \lambda}{2}} g(y(t_k), t_k) + g \left(a'_{k}, t_k + \frac{h}{2} \right) \right] +  O(h^3), \\
#     \text{with } a'_{k} = e^{- \frac{h \lambda}{2}} y(t_k) + g(y(t_k), t_k) \frac{h}{2} \phi_1 \left(-\lambda \frac{h}{2} \right),
# $$
# 
# and, for $y\left(t_{k+1} \right)$:
# 
# $$
#     y(t_{k+1}) = e^{-h \lambda}y(t_k) + \frac{h}{2} \left[ e^{-\lambda h} g(y(t_k), t_k) + g(a_k, t_{k+1}) \right] +  O(h^3), \\
#     \text{with } a_k = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) h \phi_1 (-\lambda h).
# $$
# 
# So, the expression is
# 
# $$
#   y(t_{k+1}) = e^{-h \lambda}y(t_k) + \frac{h}{6} \left[ e^{-\lambda h} g(y(t_k), t_k) + 4 e^{-\frac{ \lambda h}{2}} g\left( b'_{k}, t_k + \frac{h}{2} \right) + g(b_k, t_{k+1}) \right] +  O(h^4), \\
#   \text{with } b'_{k} = e^{- \frac{h \lambda}{2}}y(t_k) + \frac{h}{4} \left[ e^{- \frac{h \lambda}{2}} g(y(t_k), t_k) + g \left(a'_{k}, t_k + \frac{h}{2} \right) \right], \\
#   b_k = e^{-h \lambda}y(t_k) + \frac{h}{2} \left[ e^{-\lambda h} g(y(t_k), t_k) + g(a_k, t_{k+1}) \right], \\
#   a'_{k} = e^{- \frac{h \lambda}{2}} y(t_k) + g(y(t_k), t_k) \frac{h}{2} \phi_1 \left(-\lambda \frac{h}{2} \right), \\
#   a_k = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) h \phi_1 (-\lambda h).
# $$

# ## Matrix exponential
# 
# This part has information from [4].
# 
# Based on the Maclaurin series of the exponential function
# 
# $$
#     e^x = \sum_{i=0}^{\infty} \frac{x^i}{i!},
# $$
# 
# the $\textbf{exponential of a square complex matrix }A$ is defined as
# 
# $$
#     e^A \doteq \sum_{i=0}^{\infty} \frac{A^i}{i!}.
# $$
# 
# This is well defined because it has been proven that the sequence ${p_k}$ with, $\forall k \in \mathbb{N}$:
# 
# $$
#     p_k = \sum_{i=0}^{k} \frac{A^i}{i!}, \forall A \text{ as decribed above,}
# $$
# 
# is a Cauchy sequence, and therefore converge to a limit matrix which was denoted $e^A$, since the set of the square complex matrix with fixed lenght with the norm 
# 
# $$
# ||A|| = \max_{||x||=1} ||Ax||
# $$
# 
# is a Banach space.
# 
# ### Exponential of a zeros matrix
# 
# If $A =   
# \left[ {\begin{array}{ccccc}
#     0 & 0 & 0 & \dotsm & 0\\
#     0 & 0 & 0 & \dotsm & 0\\
#     0 & 0 & 0 & \dotsm & 0\\
#     \vdots & \vdots & \vdots & \ddots & \vdots\\
#     0 & 0 & 0 & \dotsm & 0\\
# \end{array} } \right] $,
# 
# $$
#     e^A \doteq \sum_{i=0}^{\infty} \frac{A^i}{i!} = I + A + \frac{A^2}{2} + \dotsm = I + 0 + 0 + \dotsm = I.
# $$
# 
# ### Exponential of a diagonal matrix
# 
# If $A =   
# \left[ {\begin{array}{ccccc}
#     \lambda_1 & 0 & 0 & \dotsm & 0\\
#     0 & \lambda_2 & 0 & \dotsm & 0\\
#     0 & 0 & \lambda_3 & \dotsm & 0\\
#     \vdots & \vdots & \vdots & \ddots & \vdots\\
#     0 & 0 & 0 & \dotsm & \lambda_{N}\\
# \end{array} } \right] 
#   = diag(\lambda_1, \lambda_2, \lambda_3, \dotsm, \lambda_N)$,
# 
# it is easy to note that
# 
# $$
#     A^2 = diag \left(\lambda_1^2, \lambda_2^2, \lambda_3^2, \dotsc, \lambda_N^2 \right)
# $$
# 
# $$
#     A^3 = diag \left(\lambda_1^3, \lambda_2^3, \lambda_3^3, \dotsc, \lambda_N^3 \right)
# $$
# 
# $$
# \vdots
# $$
# 
# $$
#     A^j = diag \left(\lambda_1^j, \lambda_2^j, \lambda_3^j, \dotsc, \lambda_N^j \right) , \forall j \in \mathbb{N}
# $$
# 
# $$
# \vdots
# $$
# 
# so
# 
# $$
#     e^A \doteq \sum_{i=0}^{\infty} \frac{A^i}{i!} = diag\left(\sum_{i=0}^{\infty} \frac{\lambda_1^i}{i!}, \sum_{i=0}^{\infty} \frac{\lambda_2^i}{i!}, \sum_{i=0}^{\infty} \frac{\lambda_3^i}{i!}, \dotsc, \sum_{i=0}^{\infty} \frac{\lambda_N^i}{i!}\right)
# $$
# 
# $$
#     = diag \left( e^{\lambda_1}, e^{\lambda_2}, e^{\lambda_3}, \dotsc, e^{\lambda_N} \right).
# $$
# 
# In the same way, if B is a diagonal by blocks matrix:
# 
# $$
# B =   
# \left[ {\begin{array}{ccccc}
#     B_1 & 0 & 0 & \dotsm & 0\\
#     0 & B_2 & 0 & \dotsm & 0\\
#     0 & 0 & B_3 & \dotsm & 0\\
#     \vdots & \vdots & \vdots & \ddots & \vdots\\
#     0 & 0 & 0 & \dotsm & B_{N}\\
# \end{array} } \right] 
#   = diag(B_1, B_2, B_3, \dotsm, B_N),
# $$
#   
# then 
#  
# $$
# e^B = diag(e^{B_1}, e^{B_2}, e^{B_3}, \dotsm, e^{B_N}).
# $$
# 
# ### Exponential of a matrix of ones above the diagonal
# 
# If $A = A_{N \times N} =   
# \left[ {\begin{array}{ccccccc}
#     0 & 1 &  &  &  &  & \\
#      & 0 & 1 &  &  &  &\\
#      &  & 0 & 1 &  &  &\\
#      &  &  & 0 & 1 &  &\\
#      &  &  &  & 0 & \ddots &  \\
#      &  &  &  &  & \ddots & 1 \\
#      &  &  &  &  &  & 0 \\
# \end{array} } \right] $,
# 
# one can calculate
# 
# $$
# A^2 = A \cdot A =  
# \left[ {\begin{array}{ccccccc}
#     0 & 1 &  &  &  &  & \\
#      & 0 & 1 &  &  &  &\\
#      &  & 0 & 1 &  &  &\\
#      &  &  & 0 & 1 &  &\\
#      &  &  &  & 0 & \ddots &  \\
#      &  &  &  &  & \ddots & 1 \\
#      &  &  &  &  &  & 0 \\
# \end{array} } \right]  \cdot 
# \left[ {\begin{array}{ccccccc}
#     0 & 1 &  &  &  &  & \\
#      & 0 & 1 &  &  &  &\\
#      &  & 0 & 1 &  &  &\\
#      &  &  & 0 & 1 &  &\\
#      &  &  &  & 0 & \ddots &  \\
#      &  &  &  &  & \ddots & 1 \\
#      &  &  &  &  &  & 0 \\
# \end{array} } \right] 
# $$
# 
# $$
#   =   \left[ {\begin{array}{ccccccc}
#     0 & 0 & 1 &  &  &  & \\
#      & 0 & 0 & 1 &  &  &\\
#      &  & 0 & 0 & 1 &  &\\
#      &  &  & 0 & 0 & \ddots &\\
#      &  &  &  & 0 & \ddots & 1 \\
#      &  &  &  &  & \ddots & 0 \\
#      &  &  &  &  &  & 0 \\
# \end{array} } \right], 
# $$
# 
# $$
#     A^3 = A \cdot A^2 = \left[ {\begin{array}{ccccccc}
#     0 & 1 &  &  &  &  & \\
#      & 0 & 1 &  &  &  &\\
#      &  & 0 & 1 &  &  &\\
#      &  &  & 0 & 1 &  &\\
#      &  &  &  & 0 & \ddots &  \\
#      &  &  &  &  & \ddots & 1 \\
#      &  &  &  &  &  & 0 \\
# \end{array} } \right] \cdot \left[ {\begin{array}{ccccccc}
#     0 & 0 & 1 &  &  &  & \\
#      & 0 & 0 & 1 &  &  &\\
#      &  & 0 & 0 & 1 &  &\\
#      &  &  & 0 & 0 & \ddots &\\
#      &  &  &  & 0 & \ddots & 1 \\
#      &  &  &  &  & \ddots & 0 \\
#      &  &  &  &  &  & 0 \\
# \end{array} } \right] 
# $$
# 
# $$
# = \left[ {\begin{array}{ccccccc}
#     0 & 0 & 0 & 1 &  &  & \\
#      & 0 & 0 & 0 & 1 &  &\\
#      &  & 0 & 0 & 0 & \ddots &\\
#      &  &  & 0 & 0 & \ddots & 1\\
#      &  &  &  & 0 & \ddots & 0 \\
#      &  &  &  &  & \ddots & 0 \\
#      &  &  &  &  &  & 0 \\
# \end{array} } \right],
# $$
# 
# $$
#     \vdots
# $$
# 
# $$
#     A^{N-2} = \left[ {\begin{array}{ccccccc}
#      &  &  &  & 0 & 1 & 0\\
#      &  &  &  &  & 0 & 1 \\
#      &  &  &  &  &  & 0 \\
#      &  &  &  &  &  &  \\
#      &  &  &  &  &  &  \\
#      &  &  &  &  &  &  \\
#      &  &  &  &  &  &  \\
# \end{array} } \right],
# $$
# 
# $$
#     A^{N-1} = \left[ {\begin{array}{ccccccc}
#      &  &  &  &  &  & 1\\
#      &  &  &  &  &  &  \\
#      &  &  &  &  &  &  \\
#      &  &  &  &  &  &  \\
#      &  &  &  &  &  &  \\
#      &  &  &  &  &  &  \\
#      &  &  &  &  &  &  \\
# \end{array} } \right],
# $$
# 
# $$
#     A^{N} = 0.
# $$
# 
# And then, with $t \in \mathbb{R}$
# 
# $$
#     e^{tA} \doteq \sum_{i=0}^{\infty} \frac{tA^i}{i!}
# $$
# 
# $$
#     = Id + tA + \frac{t^2 A^2}{2} + \frac{t^3 A^3}{6} + \dotsc + \frac{t^{N-2} A^{N-2}}{(N-2)!} + \frac{t^{N-1} A^{N-1}}{(N-1)!} + 0 + 0 + \dotsc + 0
# $$
# 
# $$
#     = \left[ {\begin{array}{ccccccc}
#     1 &  &  &  &  &  & \\
#      & 1 &  &  &  &  &\\
#      &  & 1 &  &  &  &\\
#      &  &  & 1 &  &  &\\
#      &  &  &  & 1 &  &\\
#      &  &  &  &  & \ddots &\\
#      &  &  &  &  &  & 1 \\
# \end{array} } \right] + \left[ {\begin{array}{ccccccc}
#     0 & t &  &  &  &  & \\
#      & 0 & t &  &  &  &\\
#      &  & 0 & t &  &  &\\
#      &  &  & 0 & t &  &\\
#      &  &  &  & 0 & \ddots &  \\
#      &  &  &  &  & \ddots & t \\
#      &  &  &  &  &  & 0 \\
# \end{array} } \right] + 
# $$
# 
# $$
# + \left[ {\begin{array}{ccccccc}
#     0 & 0 & \frac{t^2}{2} &  &  &  & \\
#      & 0 & 0 & \frac{t^2}{2} &  &  &\\
#      &  & 0 & 0 & \frac{t^2}{2} &  &\\
#      &  &  & 0 & 0 & \ddots &\\
#      &  &  &  & 0 & \ddots & \frac{t^2}{2} \\
#      &  &  &  &  & \ddots & 0 \\
#      &  &  &  &  &  & 0 \\
# \end{array} } \right] + \dotsc + \left[ {\begin{array}{ccccccc}
#      &  &  &  &  &  & \frac{t^{N-1}}{(N-1)!}\\
#      &  &  &  &  &  &  \\
#      &  &  &  &  &  &  \\
#      &  &  &  &  &  &  \\
#      &  &  &  &  &  &  \\
#      &  &  &  &  &  &  \\
#      &  &  &  &  &  &  \\
# \end{array} } \right]
# $$
# 
# $$
#     = \left[ {\begin{array}{ccccccc}
#     1 & t & \frac{t^2}{2} & \frac{t^3}{3!} & \frac{t^4}{4!} & \dotsc & \frac{t^{N-1}}{(N-1)!}\\
#      & 1 & t & \frac{t^2}{2} & \frac{t^3}{3!} & \ddots & \vdots \\
#      &  & 1 & t & \frac{t^2}{2} & \ddots & \frac{t^4}{4!}\\
#      &  &  & 1 & t & \ddots & \frac{t^3}{3!}\\
#      &  &  &  & 1 & \ddots & \frac{t^2}{2} \\
#      &  &  &  &  & \ddots & t \\
#      &  &  &  &  &  & 1 \\
# \end{array} } \right].
# $$
# 
# ### Exponential of a Jordan block
# 
# $\textbf{Proposition:}$ $A_1, A_2 \in \mathscr{M}_{N \times N}(\mathbb{C})$. If $A_1 \cdot A_2 = A_2 \cdot A_1$, then $e^{A_1+A_2} = e^{A_1} \cdot e^{A_2}$.
# 
# A Jordan block is of the form:
# $$
# J = \left[ {\begin{array}{ccccc}
#     \lambda_i & 1 & 0 & \dotsm & 0\\
#     0 & \lambda_i & 1 & \dotsm & 0\\
#     0 & 0 & \lambda_i & \ddots & 0\\
#     \vdots & \vdots & \vdots & \ddots & 1\\
#     0 & 0 & 0 & \dotsm & \lambda_i\\
# \end{array} } \right] 
# $$
# 
# $$
# = \left[ {\begin{array}{ccccc}
#     \lambda_i & 0 & 0 & \dotsm & 0\\
#     0 & \lambda_i & 0 & \dotsm & 0\\
#     0 & 0 & \lambda_i & \dotsm & 0\\
#     \vdots & \vdots & \vdots & \ddots & \vdots\\
#     0 & 0 & 0 & \dotsm & \lambda_i\\
# \end{array} } \right] + \left[ {\begin{array}{ccccc}
#     0 & 1 &  &  & \\
#      & 0 & 1 &  &\\
#      &  & 0 & \ddots &\\
#      &  &  & \ddots & 1\\
#      &  &  &  & 0\\
# \end{array} } \right] 
# $$
# 
# $$
#     = D + N,
# $$
# 
# and
# $$
# \left[ {\begin{array}{ccccc}
#     \lambda_i & 0 & 0 & \dotsm & 0\\
#     0 & \lambda_i & 0 & \dotsm & 0\\
#     0 & 0 & \lambda_i & \dotsm & 0\\
#     \vdots & \vdots & \vdots & \ddots & \vdots\\
#     0 & 0 & 0 & \dotsm & \lambda_i\\
# \end{array} } \right] \cdot \left[ {\begin{array}{ccccc}
#     0 & 1 &  &  & \\
#      & 0 & 1 &  &\\
#      &  & 0 & \ddots &\\
#      &  &  & \ddots & 1\\
#      &  &  &  & 0\\
# \end{array} } \right] 
# $$
# 
# $$
# = \left[ {\begin{array}{ccccc}
#     0 & \lambda_i &  &  & \\
#      & 0 & \lambda_i &  &\\
#      &  & 0 & \ddots &\\
#      &  &  & \ddots & \lambda_i\\
#      &  &  &  & 0\\
# \end{array} } \right] 
# $$
# 
# $$
# = \left[ {\begin{array}{ccccc}
#     0 & 1 &  &  & \\
#      & 0 & 1 &  &\\
#      &  & 0 & \ddots &\\
#      &  &  & \ddots & 1\\
#      &  &  &  & 0\\
# \end{array} } \right] \cdot \left[ {\begin{array}{ccccc}
#     \lambda_i & 0 & 0 & \dotsm & 0\\
#     0 & \lambda_i & 0 & \dotsm & 0\\
#     0 & 0 & \lambda_i & \dotsm & 0\\
#     \vdots & \vdots & \vdots & \ddots & \vdots\\
#     0 & 0 & 0 & \dotsm & \lambda_i\\
# \end{array} } \right],
# $$
# 
# so
# 
# $$
#     e^{tJ} = e^{tD+tN} = e^{tD} \cdot e^{tN}
# $$
# 
# $$
# = \left[ {\begin{array}{ccccc}
#     e^{t \lambda_i} & 0 & 0 & \dotsm & 0\\
#     0 & e^{t \lambda_i} & 0 & \dotsm & 0\\
#     0 & 0 & e^{t \lambda_i} & \dotsm & 0\\
#     \vdots & \vdots & \vdots & \ddots & \vdots\\
#     0 & 0 & 0 & \dotsm & e^{t \lambda_i}\\
# \end{array} } \right] \cdot \left[ {\begin{array}{ccccc}
#     1 & t & \frac{t^2}{2} & \dotsc & \frac{t^{N-1}}{(N-1)!}\\
#      & 1 & t & \ddots & \vdots\\
#      &  & 1 & \ddots & \frac{t^2}{2} \\
#      &  &  & \ddots & t \\
#      &  &  &  & 1 \\
# \end{array} } \right]
# $$
# 
# $$
# = \left[ {\begin{array}{ccccc}
#     e^{t \lambda_i} & e^{t \lambda_i}t & \frac{e^{t \lambda_i} t^2}{2} & \dotsc & \frac{e^{t \lambda_i} t^{N-1}}{(N-1)!}\\
#      & e^{t \lambda_i} & e^{t \lambda_i} t & \ddots & \vdots\\
#      &  & e^{t \lambda_i} & \ddots & \frac{e^{t \lambda_i} t^2}{2} \\
#      &  &  & \ddots & e^{t \lambda_i} t \\
#      &  &  &  & e^{t \lambda_i} \\
# \end{array} } \right], t \in \mathbb{R}.
# $$
# 
# ### Exponential of any matrix
# 
# $\textbf{Proposition: } \forall A \in \mathscr{M}_{N \times N}(\mathbb{C}), \exists M \in \mathscr{M}_{N \times N}(\mathbb{C})$ invertible, such that $A = MJM^{-1}$, with 
# 
# $$ 
# J = \left[ {\begin{array}{ccccc}
#     J_1 & 0 & 0 & \dotsm & 0\\
#     0 & J_2 & 0 & \dotsm & 0\\
#     0 & 0 & J_3 & \dotsm & 0\\
#     \vdots & \vdots & \vdots & \ddots & \vdots\\
#     0 & 0 & 0 & \dotsm & J_{N}\\
# \end{array} } \right]
# $$
# 
# and each $J_i$, $i = 1, 2, 3, \dotsc, N$ being a Jordan block, i.e., 
# 
# $$
# J_i = \left[ {\begin{array}{ccccc}
#     \lambda_i & 0 & 0 & \dotsm & 0\\
#     0 & \lambda_i & 0 & \dotsm & 0\\
#     0 & 0 & \lambda_i & \dotsm & 0\\
#     \vdots & \vdots & \vdots & \ddots & \vdots\\
#     0 & 0 & 0 & \dotsm & \lambda_i\\
# \end{array} } \right]
# $$ 
# 
# for some $\lambda_i \in \mathbb{C}$ .
# 
# Note that
# $$
#     (MJM^{-1})^k = MJM^{-1}MJM^{-1}MJM^{-1} \dotsc MJM^{-1} 
# $$
# 
# $$
#     = MJIJIJM^{-1} \dotsc MJM^{-1} = MJJJ \dotsc JM^{-1} = MJ^kM^{-1}.
# $$
# 
# Because of the formula of the series that defines the expansion, it implicates in $e^{MJM^{-1}} = M e^J M^{-1}$.
# 
# And then, using the same notation from the last proposition,
# $$
# e^{tA} = e^{tMJM^{-1}} = e^{MtJM^{-1}} = Me^{tJ}M^{-1} 
# $$
# 
# $$
#     = M \left[ { \begin{array}{ccccc}
#         e^{tJ_1} & 0 & 0 & \dotsm & 0\\
#         0 & e^{tJ_2} & 0 & \dotsm & 0\\
#         0 & 0 & e^{tJ_3} & \dotsm & 0\\
#         \vdots & \vdots & \vdots & \ddots & \vdots\\
#         0 & 0 & 0 & \dotsm & e^{tJ_{N}}\\
#     \end{array} } \right] M^{-1}, t \in \mathbb{R},
# $$
# with each block as the section above indicates.

# ## Euler method
# 
# Further detailing this explicit one-step method of
# 
# $$
#     \phi (t_{k},y_{k},h) = f(t_{k},y_{k}),
# $$
# 
# an analysis on stability, convergence and order of convergence is done.
# 
# ### Stability
# 
# For the problem
# $\begin{cases}
#     y'(t) = - \lambda y(t) \text{ ; } t \in [t_0 , T] \\
#     y(t_0)=y_0,
# \end{cases}$
# 
# with known solution
# 
# $$ y(t) = y_0e^{-\lambda (t-t_0)},$$
# 
# the method turn into:
# 
# $$
# y_0 = y(t_0)\\
# \textbf{for } k = 0, 1, 2, ..., N-1 :\\
#     y_{k+1} = y_k + h \lambda y_k \\
#     t_{k+1} = t_k + h.
# $$
# 
# Then the amplification factor is:
# $$
# (1 - h \lambda).
# $$
# 
# If
# 
# $$
# |1 - h \lambda| > 1, \text{for fixed } N,
# $$ 
# 
# it will be a divergent series 
# 
# $$
# (k \rightarrow \infty \Rightarrow y_k \rightarrow \infty),
# $$
# 
# so, since the computer has a limitant number that can represent, even if the number of steps is such that $h$ is not small enought, it might have sufficient steps to reach the maximum number represented by the machine.
# 
# However, if 
# 
# $$
#     |1 - h \lambda| < 1 \text{ and } N \text{ is fixed,}
# $$ 
# 
# it converges to zero 
# 
# $$
#     (k \rightarrow \infty \Rightarrow y_k \rightarrow 0 ).
# $$
# 
# 
# Besides that, 
# 
# $$
# |1 - h \lambda| < 1
# $$ 
# 
# is the same as 
# 
# $$
# 0 < h \lambda < 2.
# $$
# 
# So the interval of stability is $(0,2)$.
# 
# That's why the method suddenly converged, it was when $h$ got small enought to $h \lambda$ be in the interval of stability, i.e., 
# 
# $$
#     h < 2/\lambda.
# $$
# 
# It is worth mentioning here that if 
# 
# $$
# -1 < 1 - h \lambda < 0,
# $$
# 
# the error will converge oscillating since it takes positive values with even exponents and negative with odd ones.
# 
# ### Convergence
# Since
# 
# $$
# \lim_{m \to +\infty} \left(1 + \frac{p}{m} \right)^m = e^p,
# $$
# 
# and h = $\frac{T-t_0}{N}$, for $y_N$ we have
# 
# $$
# \lim_{N \to +\infty} y_N = \lim_{N \to +\infty} \left(1 - h \lambda \right)^N y_0 = \lim_{N \to +\infty} \left(1 - \frac{(T-t_0) \lambda}{N} \right)^N y_0.
# $$
# 
# It is reasonable to take $p = -(T-t_0) \lambda$ and conclude that the last point estimated by the method will converge to
# 
# $$
# y_0e^{-\lambda (T-t_0)}.
# $$
# 
# Which is precisely $y(T)$ and proves the convergence.
# 
# ### Order of convergence
# 
# Being $\tau(h, t_k)$ the local truncation error.
# 
# From
# 
# $$
#     y(t_{k+1}) = y(t_k) + h f(y(t_k),t_k) + O(h^2),
# $$
# 
# we have
# 
# $$
#     h \tau(h, t_k) \doteq \frac{y(t_{k+1}) - y(t_k)}{h} - f(t_k, y(t_k)) = O(h^2),
# $$
# 
# so
# 
# $$
#     \tau(h, t_k) = O(h).
# $$
# 
# Since for one step methods the order of convergence is the order of the local truncation error, the order is of $O(h)$, order 1.
