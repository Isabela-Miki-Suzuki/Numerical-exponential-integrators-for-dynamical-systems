#!/usr/bin/env python
# coding: utf-8

# # Chapter 2: Exponential methods

# In this chapter, exponential methods are introduced, with further analysis of some of them, being tested and compared to more classical equivalents.

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
    hphi1 = np.matmul(1-expm(-h*A), linalg.inv(A))
    return hphi1

def calculate_hphi2(h, A, hphi1):
    #IT IS NOT H2PHI2
    '''(float, np.matrix, np.matrix) -> np.matrix'''
    hphi2 = np.matmul(1-hphi1/h, linalg.inv(A))
    return hphi2

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

def etd2rk_cox_and_matthews(t0, tf, n, x0, A, g):
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

def etd2rk_cox_and_matthews_midpoint_rule(t0, tf, n, x0, A, g):
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

def etd2rk_trapezoidal_rule(t0, tf, n, x0, A, g):
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

def etd2rk_midpoint_rule(t0, tf, n, x0, A, g):
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

def vectorize_sol(t0, t1, n, sol):
    '''
    (float, float, int, function) -> np.vector
    n is the number of steps
    '''
    x = np.zeros(n, dtype=np.complex_)
    h = (t1-t0)/n
    for i in range(n):
        x[i] = sol(t0+i*h)
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

def g( x, t ):
    ''' (np.array, float) -> float
        (x, t) -> g(x, t)
    '''
    g = np.array([np.sin(t)])
    return g

def g_linear_deprec( x, t ):
    ''' (float, float) -> float
        (x, t) -> g(x, t)
    '''
    g = 0
    return g

def g_linear( x, t ):
    ''' (np.array, float) -> np.array
        (x, t) -> g(x, t)
    '''
    g = np.zeros(x.size)
    return g

def g_cm1 (x, t):
    ''' (np.array, float) -> np.array
        (x, t) -> g(x, t)
    '''
    lamb = .5
    c = 100
    r_2 = x[0]**2 + x[1]**2
    g = np.array([(lamb*x[1]-c*x[0])*r_2, -(lamb*x[0]+c*x[1])*r_2])
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

def sol_100_linear( t ):
    ''' (float, float) -> float
    RECEIVES the initial value and a real (t).
    APPLIES the cauchy problem solution to this initial value at this point.
    RETURNS a real value.
    '''
    sol = exp(-100*t) #u0=1
    return sol

def sol_1j_linear( t ):
    ''' (float, float) -> float
    RECEIVES the initial value and a real (t).
    APPLIES the cauchy problem solution to this initial value at this point.
    RETURNS a real value.
    '''
    return np.exp(1j*t)

def sol_non_linear_sin( t ):
    ''' (float, float) -> float
    RECEIVES the initial value and a real (t).
    APPLIES the cauchy problem solution to this initial value at this point.
    RETURNS a real value.
    '''
    sol = 2-cos(t) #u0=1
    return sol

def errors_array(n0, nf, method, t0, tf, x0, lmba, g, sol, vectorize_sol, error):
  '''
  This function will RETURN 2 arrays.
  The first one has the errors of the approximations given by the method with
  number of steps n = n0, n0+1, n0+2, ..., nf-1.
  The second is [n0, n0+1, n0+2, ..., nf-1]

  RECEIVES:
  n0 is the first number of steps. (int)
  nf is the last one plus 1. (int)
  method have arguments (t0, tf, n, x0, lmba, g) and return a
  np.vector of length n (0, 1, 2, ..., n-1), n is the number of steps. (function)
  t0 is the initial point of the approximation. (float)
  tf is the last one. (float)
  x0 is the initial value of the Cauchy problem. (float)
  lmbda is the coefficient os the linear part of the ploblem. (float)
  g is a function (float, float) -> (float). (function)
  sol is a function (float) -> (float). (function)
  vectorize_sol is a function that "transforms sol in a vector" (function)
  (float, float, int, function) -> (np.array)
  (t0, tf, n, sol) -> np.array([sol[t0], sol[t0+h], sol[t0+2h], ..., sol[tf-1]])
  error is a function (np.array, np.array) -> (float) (function)
  '''
  v = np.zeros(nf-n0)
  domain = np.zeros(nf-n0)
  for n in range(n0, nf):
    domain[n-n0] = n
    m = method(t0, tf, n, x0, lmba, g)
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

  colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
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

def errors_2x(n0, k, method, t0, tf, x0, lmba, g, sol, vectorize_sol, error):
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
  lmbda is the coefficient os the linear part of the ploblem. (float)
  g is a function (float, float) -> (float). (function)
  sol is a function (float) -> (float). (function)
  vectorize_sol is a function that "transforms sol in a vector" (function)
  (float, float, int, function) -> (np.array)
  (t0, tf, n, sol) -> np.array([sol[t0], sol[t0+h], sol[t0+2h], ..., sol[tf-1]])
  error is a function (np.array, np.array) -> (float) (function)
  '''
  v = np.zeros(k)
  for i in range(k):
    m = method(t0, tf, n0*2**i, x0, lmba, g)
    exact = vectorize_sol(t0, tf, n0*2**i, sol)
    v[i] = error(m, exact)
  return v

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
  for k in range(1, 4):
      n = n0 * 2 ** k
      h = (tf-t0)/n
      q = errors_2x[k-1]/errors_2x[k] #q=erro(h)/erro(h)
      r = ((tf-t0)/(n/2))/((tf-t0)/n)
      print(n, h, errors_2x[k], log(q,2)/log(r,2), sep=" & ", end=" \\\\ \n")


# ## Exponential Euler method

# For
# \begin{cases}
#     y'(t) + \lambda y(t) = g(y(t), t), t \in (t_0, T) \\
#     y(0) = y_0
# \end{cases}
# 
# the domain is evenly discretized:
# 
# \begin{equation*}
#     N \in \mathbb{N}; h = \frac{T-t_0}{N}; \text{Domain: }\{t_k = t_0 + k h : k = 0, 1, ...\}.
# \end{equation*}
# 
# The discretization of the ODE takes the exact solution of the Cauchy problem, given by the variation of constants formula
# \begin{equation*}
#     y(t) = e^{-(t-t_0) \lambda}y_0 + \int_{t_0}^t [e^{-\lambda(t-\tau)} g(y(\tau), \tau)] d\tau
# \end{equation*}
# 
# and, by Taylor expansion on $g$:
# 
# $\tau \in (t_k, t_{k+1})$
# \begin{equation*}
#     g(y(\tau), \tau) = g(y(t_k), t_k) + (\tau - t_k) \frac{dg}{dt} (y(\theta_k), \theta_k)
# \end{equation*}
# for a $\theta_k \in (t_k, t_{k+1}),$
# 
# \begin{equation*}
#     y(t_{k+1}) = e^{-(t_{k+1}-t_k) \lambda}y(t_k) + \int_{t_k}^{t_{k+1}} [e^{-\lambda(t_{k+1}-\tau)} g(y(\tau), \tau)] d\tau
# \end{equation*}
# 
# \begin{equation*}
#     = e^{-h \lambda}y(t_k) + \int_{t_k}^{t_{k+1}} \left[e^{-\lambda(t_{k+1}-\tau)} \left( g(y(t_k), t_k) + (\tau - t_k) \frac{dg}{dt} (y(\theta_k), \theta_k)\right)\right] d\tau
# \end{equation*}
# 
# \begin{equation*}
#     = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) \int_{t_k}^{t_{k+1}} e^{-\lambda(t_{k+1}-\tau)} d\tau + \frac{dg}{dt} (y(\theta_k), \theta_k) \int_{t_k}^{t_{k+1}} (\tau - t_k) e^{-\lambda(t_{k+1}-\tau)} d\tau.
# \end{equation*}
# 
# Since
# 
# \begin{equation*}
#     \int_{t_k}^{t_{k+1}} e^{-\lambda(t_{k+1}-\tau)} d\tau = h\phi_1(-\lambda h)= \frac{1-e^{-h \lambda}}{\lambda}
# \end{equation*}
# 
# and, by the Taylor expansion of $e^{-\lambda h}$ in the point zero
# 
# \begin{equation*}
#     e^{-\lambda h} = 1 - \lambda h + \frac{1}{2}\lambda^2h^2 - \frac{1}{3!}\lambda^3h^3 + \dotsi + \frac{1}{n!} (-\lambda h)^n + \dotsi, n \in \mathbb{N}
# \end{equation*}
# 
# \begin{gather*}
#      \int_{t_k}^{t_{k+1}} (\tau - t_k) e^{-\lambda(t_{k+1}-\tau)} d\tau =
#      h^2 \phi_2 (-\lambda h) =
#      h \frac{\phi_1(0) - \phi_1(-\lambda h)}{\lambda} =
#      \frac{h}{\lambda} - \frac{1-e^{-h \lambda}}{\lambda^2} = \\
#      \frac{h}{\lambda} - \frac{1-(1 - \lambda h + \frac{1}{2}\lambda^2h^2 - \frac{1}{3!}\lambda^3h^3 + \dotsi + \frac{1}{n!} (-\lambda h)^n + \dotsi)}{\lambda^2} = \\
#      \frac{h^2}{2} - \frac{h^3}{3!} \lambda + \dotsi + \frac{h^n}{n!} (-\lambda)^{n-2} + \dotsi  =  O(h^2),
# \end{gather*}
# 
# \begin{equation*}
#     y(t_{k+1}) = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) \frac{1-e^{-h \lambda}}{\lambda} + \frac{dg}{dt} (y(\theta_k), \theta_k) O(h^2),
# \end{equation*}
# 
# \begin{equation*}
#   y(t_{k+1}) = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) \frac{1-e^{-h \lambda}}{\lambda} + O(h^2).
# \end{equation*}
# 
# That inspires the $\textbf{Exponential Euler method}$ :
# $$
# y_0 = y(t_0)\\
# \textbf{for } k = 0, 1, 2, ..., N-1 :\\
#     y_{k+1} = e^{-h \lambda}y_k + g(y_k, t_k) \frac{1-e^{-h \lambda}}{\lambda}\\
#     t_{k+1} = t_k + h
# $$
# with $y_k \thickapprox y(t_k)$.

# In[ ]:




