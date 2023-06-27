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

def classic_euler_deprec(t0, tf, n, x0, lamb, g):
    '''(float, float, int, float, float, function) -> np.vector'''
    h = (tf-t0)/n
    x = np.zeros(n)
    x[0]=x0
    t = t0
    for i in range(1, n):
        x[i] = x[i-1] + h*(-lamb*x[i-1] + g(x[i-1],t))
        t = t0 + i*h
        if np.abs(x[i]) > stab_lim:
            x[i] = 0.0
    return x

def exponential_euler_deprec(t0, tf, n, x0, lamb, g):
    '''(float, float, int, float, function) -> np.vector'''
    h = (tf-t0)/n
    x = np.zeros(n)
    x[0]=x0
    t = t0
    phi1 = (1-np.exp(-h*lamb))/lamb
    for i in range(1, n):
        x[i] = np.exp(-h*lamb)*x[i-1] + phi1*g(x[i-1],t)
        t = t0 + i*h
    return x

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
  print("| n | h = $\\frac{1}{h}$ | $\\tau(0,h)$ | q = $\\frac{tau(0,h)}{tau(0, 2h)}$ | $log_4 ^q$|")
  print("|---|-----------------|-----------|---------------------------------|-------|")
  print("", n, (tf-t0)/n, errors_2x[0], "-", sep=" | ", end=" | \n")
  for i in range(1, k):
      n = n0 * 2 ** i
      h = (tf-t0)/n
      q = errors_2x[i-1]/errors_2x[i] #q=erro(h)/erro(h)
      r = ((tf-t0)/(n/2))/((tf-t0)/n)
      print( "", n, h, errors_2x[i], log(q,2)/log(r,2), sep=" | ", end=" | \n")


# The execution done are the following.

# In[2]:


n0 = 128
k = 4
t0 = 0
tf = 1
x0 = np.array([1])
A = np.array([[100]])
errors_2x_vector = errors_2x(n0, k, classic_euler, t0, tf, x0, A, g, sol, vectorize_sol, error_sup)
convergence_table(errors_2x_vector, n0, k, t0, tf)


# In[3]:


n0 = 128
k = 4
t0 = 0
tf = 1
x0 = np.array([1])
A = np.array([[100]])
errors_2x_vector = errors_2x(n0, k, exponential_euler, t0, tf, x0, A, g, sol, vectorize_sol, error_sup)
convergence_table(errors_2x_vector, n0, k, t0, tf)


# In[4]:


n0 = 128
k = 4
t0 = 0
tf = 1
x0 = np.array([1])
A = np.array([[100]])

errors_2x_vector = errors_2x(n0, k, etd2rk_cox_and_matthews, t0, tf, x0, A, g, sol, vectorize_sol, error_sup)
convergence_table(errors_2x_vector, n0, k, t0, tf)


# In[5]:


n0 = 128
k = 4
t0 = 0
tf = 1
x0 = np.array([1])
A = np.array([[100]])

errors_2x_vector = errors_2x(n0, k, etd2rk_cox_and_matthews_midpoint_rule, t0, tf, x0, A, g, sol, vectorize_sol, error_sup)
convergence_table(errors_2x_vector, n0, k, t0, tf)


# In[6]:


n0 = 128
k = 4
t0 = 0
tf = 1
x0 = np.array([1])
A = np.array([[100]])

errors_2x_vector = errors_2x(n0, k, etd2rk_trapezoidal_rule, t0, tf, x0, A, g, sol, vectorize_sol, error_sup)
convergence_table(errors_2x_vector, n0, k, t0, tf)


# In[7]:


n0 = 128
k = 4
t0 = 0
tf = 1
x0 = np.array([1])
A = np.array([[100]])

errors_2x_vector = errors_2x(n0, k, etd2rk_midpoint_rule, t0, tf, x0, A, g, sol, vectorize_sol, error_sup)
convergence_table(errors_2x_vector, n0, k, t0, tf)


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
# It is also possible to think the exponential time differencing methods with Runge-Kutta time stepping using the numerical integration, for example, for the one with second order, it starts with the trapezoidal rule (which was taken from [8]) on the variation of constants formula:
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
# Besides that, using the midpoint rule, also known as rectangle rule, again taken from [8],
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
