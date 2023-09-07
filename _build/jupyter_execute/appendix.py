#!/usr/bin/env python
# coding: utf-8

# # Appendix

# ## Code

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

