#!/usr/bin/env python
# coding: utf-8

# # Exponential methods

# In[1]:


from basecode import *


# In this chapter, exponential methods are introduced, with further analysis of some of them, being tested and compared to more classical equivalents.
# 
# The whole basecode is in the appendix.
# 
# The stiff problem used in all the convergence tables is the following one taken from [1]:
# 
# $$
#     u'(t) + 100 u(t) = \sin(t)\\
#     u(0) = 1.
# $$
# 
# Solution: 
# 
# $$
# u(t) = \exp(-100t)+\frac{\exp(-100t)+100\sin(t)-\cos(t)}{1+100^2}.
# $$

# In this whole section informations from [1], [6], [7] are used.

# ## Exponential Euler method

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

# Expression:
# 
# $$
#   y(t_{k+1}) = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) \frac{1-e^{-h \lambda}}{\lambda} + O(h^2).
# $$

# Table of convergence:

# In[2]:


n0 = 128
k = 4
t0 = 0
tf = 1
x0 = np.array([1])
A = np.array([[100]])
errors_2x_vector, domain = errors_2x(n0, k, exponential_euler, t0, tf, x0, A, g, sol, vectorize_sol, error_sup)
convergence_table(errors_2x_vector, n0, k, t0, tf)


# 
# | n | h = $\frac{1}{h}$ | $\tau(0,h)$ | q = $\frac{tau(0,h)}{tau(0, 2h)}$ |
# |---|-----------------|-----------|---------------------------------|
#  | 128 | 0.0078125 | 4.398075514689716e-05 | - | 
#  | 256 | 0.00390625 | 2.074422525626487e-05 | 1.0841625981445133 | 
#  | 512 | 0.001953125 | 1.0056221183126109e-05 | 1.0446214904461004 | 
#  | 1024 | 0.0009765625 | 4.948885884282876e-06 | 1.0229126060177947 | 

# The table proved the order of conergence given by the deduction, and, comparing to the one of the classic Euler method:

# In[3]:


n0 = 128
k = 4
t0 = 0
tf = 1
x0 = np.array([1])
A = np.array([[100]])
errors_2x_vector, domain = errors_2x(n0, k, classic_euler, t0, tf, x0, A, g, sol, vectorize_sol, error_sup)
convergence_table(errors_2x_vector, n0, k, t0, tf)


# | n | h = $\frac{1}{h}$ | $\tau(0,h)$ | q = $\frac{tau(0,h)}{tau(0, 2h)}$ |
# |---|-----------------|-----------|---------------------------------|
#  | 128 | 0.0078125 | 0.2391072699739873 | - | 
#  | 256 | 0.00390625 | 0.08650412059872986 | 1.466817233501749 | 
#  | 512 | 0.001953125 | 0.039214210532948934 | 1.1413923006132296 | 
#  | 1024 | 0.0009765625 | 0.018739566082401515 | 1.0652890085799935 | 

# the exponential one has much better approximations since the beginning, proving the efficiency of the exponential method.

# ## Exponential time differencing methods (ETD)

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

# Expression:
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

# It is possible to note that the exponential euler is essentially the exponential time differencing method of order 1.
# 
# In the same way as Taylor methods, the problem here is that at the expense of a higher order of convergence, ends up requiring the evaluation and implementation of multiple derivatives that may not even be easy to calculate. It can be avoided using Runge-Kutta methods, the next to be analyzed.

# ## Exponential time differencing methods with Runge-Kutta time stepping
# 
# Here, the exponential Runge-Kutta were compared to methods from deductions following the classic runge-kutta approach in the constant variation formula.
# 
# All convergence tables prove the deduced order.
# 
# However, it is remarkable how much better the exponential methods are in relation to the new ones presented here, showing that we cannot be naive and apply the integral approximations expecting an exponential Runge-Kutta performance, the treatment must be exact for the linear part, as is done in all exponential methods.
# 
# ### Exponential - Trapezoidal rule
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
# 
# Expression:
# 
# $$
#   y(t_{k+1}) = e^{-h \lambda} y(t_k) +
#   h \phi_1 (-\lambda h) g(y(t_k), t_k) +
#   \left[g(a_k, t_{k+1}) - g(y(t_k), t_k) \right] h \phi_2 (-\lambda h) + \\
#   + O(h^3) \\
#   \text{with } a_k = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) \frac{1-e^{-h\lambda}}{\lambda}.
# $$

# In[4]:


n0 = 128
k = 4
t0 = 0
tf = 1
x0 = np.array([1])
A = np.array([[100]])

errors_2x_vector, domain = errors_2x(n0, k, etd2rk, t0, tf, x0, A, g, sol, vectorize_sol, error_sup)
convergence_table(errors_2x_vector, n0, k, t0, tf)


# Convergence table:
# 
# | n | h = $\frac{1}{h}$ | $\tau(0,h)$ | q = $\frac{tau(0,h)}{tau(0, 2h)}$ |
# |---|-----------------|-----------|---------------------------------|
#  | 128 | 0.0078125 | 4.186569175362864e-08 | - | 
#  | 256 | 0.00390625 | 1.0575183428604418e-08 | 1.985085775819591 | 
#  | 512 | 0.001953125 | 2.652380943352073e-09 | 1.9953227875115886 | 
#  | 1024 | 0.0009765625 | 6.638462730912398e-10 | 1.9983668943519293 |

# ### Naive deduction - Trapezoidal rule
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
# 
# Expression:
# 
# $$
# y(t_{k+1}) = e^{-h \lambda}y(t_k) + \frac{h}{2} \left[ e^{-\lambda h} g(y(t_k), t_k) + g(a_k, t_{k+1}) \right] +  O(h^3) \\
#     \text{with } a_k = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) h \phi_1 (-\lambda h).
# $$

# In[5]:


n0 = 128
k = 4
t0 = 0
tf = 1
x0 = np.array([1])
A = np.array([[100]])

errors_2x_vector, domain = errors_2x(n0, k, etd2rk_trapezoidal_naive, t0, tf, x0, A, g, sol, vectorize_sol, error_sup)
convergence_table(errors_2x_vector, n0, k, t0, tf)


# Convergence table:
# 
# | n | h = $\frac{1}{h}$ | $\tau(0,h)$ | q = $\frac{tau(0,h)}{tau(0, 2h)}$ |
# |---|-----------------|-----------|---------------------------------|
#  | 128 | 0.0078125 | 0.0004242643044311458 | - | 
#  | 256 | 0.00390625 | 0.00010714498082271644 | 1.9853990333325726 | 
#  | 512 | 0.001953125 | 2.6871031228085582e-05 | 1.9954406751889993 | 
#  | 1024 | 0.0009765625 | 6.725136514989377e-06 | 1.9984162299862431 | 

# Notably worse than the real exponential one.

# To compare, this is the convergence table of the RK2, in which we can see that the errors are much bigger:

# In[6]:


n0 = 128
k = 4
t0 = 0
tf = 1
x0 = np.array([1])
A = np.array([[100]])

errors_2x_vector, domain = errors_2x(n0, k, rk2, t0, tf, x0, A, g, sol, vectorize_sol, error_sup)
convergence_table(errors_2x_vector, n0, k, t0, tf)


# | n | h = $\frac{1}{h}$ | $\tau(0,h)$ | q = $\frac{tau(0,h)}{tau(0, 2h)}$ |
# |---|-----------------|-----------|---------------------------------|
#  | 128 | 0.0078125 | 0.06606851127601271 | - | 
#  | 256 | 0.00390625 | 0.01256096444797522 | 2.395015596211044 | 
#  | 512 | 0.001953125 | 0.0027104154026279526 | 2.212361357172686 | 
#  | 1024 | 0.0009765625 | 0.0006264383048139033 | 2.1132696413977325 | 

#  ### Exponential - Third order
# 
#  ### Third order exponential time differencing methods with Runge-Kutta time stepping (ETDRK-3)
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
#   g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right)
#   h \phi_1(-h \lambda) +
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
#   g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right)
#   h \phi_1(-h \lambda) + %ok
#   \left[g(y(t_{k+1}), t_{k+1}) - g(y(t_k), t_k)\right]
#   \left( h \phi_2 (-h \lambda) - \frac{h \phi_1(-h \lambda)}{2} \right) +
#   \\
#   + 4 \left[g(y(t_{k+1}), t_{k+1}) + g(y(t_k), t_k) - 2 g\left(y\left(t_{k+\frac{1}{2}}\right), t_{k+\frac{1}{2}}\right) \right]
#   \left( h \phi_3 (-h \lambda) + \frac{h \phi_1(-h \lambda)}{8} - \frac{h \phi_2(-h \lambda)}{2} \right) + O(h^4).
# $$
# 
# Using the Cox and Mathhews's ETDRK-2 expressions to approximate $y\left(t_{k+\frac{1}{2}}\right)$ and $y(t_{k+1})$, since those are of order 2, i.e., $O(h^3)$, the expression of the method is:
# 
# $$
#   y(t_{k+1}) = e^{-h \lambda} y(t_k) +
#   g\left(c'_k, t_{k+\frac{1}{2}}\right)
#   h \phi_1(-h \lambda) + %ok
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
#   h \phi_1 (-\lambda h) g(y(t_k), t_k) +
#   \left[g(a_k, t_{k+1}) - g(y(t_k), t_k) \right] h \phi_2 (-\lambda h),
#   \\
#   a_k = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) h \phi_1(-h\lambda),
#   \\
#   c'_k = e^{- \frac{h \lambda}{2}} y(t_k) +
#   \frac{h}{2} \phi_1 \left(- \frac{\lambda h}{2} \right) g(y(t_k), t_k) +
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

# In[7]:


n0 = 128
k = 4
t0 = 0
tf = 1
x0 = np.array([1])
A = np.array([[100]])

errors_2x_vector, domain = errors_2x(n0, k, etd3rk_similar, t0, tf, x0, A, g, sol, vectorize_sol, error_sup)
convergence_table(errors_2x_vector, n0, k, t0, tf)


# Convergence table:
# 
# | n | h = $\frac{1}{h}$ | $\tau(0,h)$ | q = $\frac{tau(0,h)}{tau(0, 2h)}$ |
# |---|-----------------|-----------|---------------------------------|
#  | 128 | 0.0078125 | 5.0853024048669315e-12 | - | 
#  | 256 | 0.00390625 | 3.212833644961055e-13 | 3.9844153810116354 | 
#  | 512 | 0.001953125 | 2.0132983821752326e-14 | 3.996213373698299 | 
#  | 1024 | 0.0009765625 | 1.2602766052971504e-15 | 3.997748687591092 | 
# 
#  Better than what expected, giving order 4.

# ### Naive deduction - Third order
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

# In[8]:


n0 = 128
k = 4
t0 = 0
tf = 1
x0 = np.array([1])
A = np.array([[100]])

errors_2x_vector, domain = errors_2x(n0, k, etd3rk_naive, t0, tf, x0, A, g, sol, vectorize_sol, error_sup)
convergence_table(errors_2x_vector, n0, k, t0, tf)


# Convergence table:
# 
# | n | h = $\frac{1}{h}$ | $\tau(0,h)$ | q = $\frac{tau(0,h)}{tau(0, 2h)}$ |
# |---|-----------------|-----------|---------------------------------|
#  | 128 | 0.0078125 | 1.083876968009309e-06 | - | 
#  | 256 | 0.00390625 | 6.883813637344194e-08 | 3.9768491535433466 | 
#  | 512 | 0.001953125 | 4.322307012305515e-09 | 3.9933345852265947 | 
#  | 1024 | 0.0009765625 | 2.705360744453822e-10 | 3.9979086629155343 | 
# 
#  Also with order 4, better than what expected. But notably worse than the real exponential one.

# To compare, this is the convergence table of the RK4, in which we can see that the errors are much bigger:

# In[9]:


n0 = 128
k = 4
t0 = 0
tf = 1
x0 = np.array([1])
A = np.array([[100]])

errors_2x_vector, domain = errors_2x(n0, k, rk4, t0, tf, x0, A, g, sol, vectorize_sol, error_sup)
convergence_table(errors_2x_vector, n0, k, t0, tf)


# | n | h = $\frac{1}{h}$ | $\tau(0,h)$ | q = $\frac{tau(0,h)}{tau(0, 2h)}$ |
# |---|-----------------|-----------|---------------------------------|
#  | 128 | 0.0078125 | 0.002141816843239275 | - | 
#  | 256 | 0.00390625 | 9.770249694801558e-05 | 4.4542958704375835 | 
#  | 512 | 0.001953125 | 5.250705130854794e-06 | 4.21781234893684 | 
#  | 1024 | 0.0009765625 | 3.024340525237257e-07 | 4.1178186851779905 | 

# ## Graphics
# 
# Next, it is shown graphics from the same problem, but first showing the error as the linear part, $100$, changes from $0$ to $100$ and next with $\lambda = 100$ but changing the time step.

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


# Here it is notable that as the lambda increases, and so does the stiffness, the exponential methods deal really well, even dropping the error, since the exponential part is precisely solved, so, as it gains more relevance, the method perfoms better. Meanwhile, the other methods (classic and naive) start to decline, dealing badly with the stiffness. Just as predicted.

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


# Here is visually clear the orders already confirmed by the convergence tables.
