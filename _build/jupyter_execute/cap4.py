#!/usr/bin/env python
# coding: utf-8

# # Exponential methods

# In this chapter, exponential methods are introduced, with further analysis of some of them, being tested and compared to more classical equivalents.

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

# ## Exponential time differencing methods (ETD)
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

# It is possible to note that the exponential euler is essentially the exponential time differencing method of order 1.
# 
# In the same way as Taylor methods, the problem here is that at the expense of a higher order of convergence, ends up requiring the evaluation and implementation of multiple derivatives that may not even be easy to calculate. It can be avoided using Runge-Kutta methods.

# ## Exponential time differencing methods with Runge-Kutta time stepping
# 
# For the Runge-Kutte methods, that is used approximations of the derivatives that converges together with the whole expression as the time step decreases.
