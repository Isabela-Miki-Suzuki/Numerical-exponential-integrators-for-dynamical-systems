#!/usr/bin/env python
# coding: utf-8

# # Exponential methods

# In this chapter, exponential methods are introduced, with further analysis of some of them, being tested and compared to more classical equivalents.
# 
# All the codes that created the convergence and deduction tables are in the appendix.
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

# ## Exponential Euler method

# Expression:
# 
# $$
#   y(t_{k+1}) = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) \frac{1-e^{-h \lambda}}{\lambda} + O(h^2).
# $$
# 
# Table of convergence:
# 
# | n | h = $\frac{1}{h}$ | $\tau(0,h)$ | q = $\frac{tau(0,h)}{tau(0, 2h)}$ | $log_4 ^q$|
# |---|-----------------|-----------|---------------------------------|-------|
#  | 128 | 0.0078125 | 4.398075514689716e-05 | - | 
#  | 256 | 0.00390625 | 2.074422525626487e-05 | 1.0841625981445133 | 
#  | 512 | 0.001953125 | 1.0056221183126109e-05 | 1.0446214904461004 | 
#  | 1024 | 0.0009765625 | 4.948885884282876e-06 | 1.0229126060177947 | 

# The table proved the order of conergence given by the deduction, and, comparing to the one of the classic Euler method:

# | n | h = $\frac{1}{h}$ | $\tau(0,h)$ | q = $\frac{tau(0,h)}{tau(0, 2h)}$ | $log_4 ^q$|
# |---|-----------------|-----------|---------------------------------|-------|
#  | 128 | 0.0078125 | 0.2391072699739873 | - | 
#  | 256 | 0.00390625 | 0.08650412059872986 | 1.466817233501749 | 
#  | 512 | 0.001953125 | 0.039214210532948934 | 1.1413923006132296 | 
#  | 1024 | 0.0009765625 | 0.018739566082401515 | 1.0652890085799935 | 

# the exponential one has much better approximations since the beginning, proving the efficiency of the exponential method.

# ## Exponential time differencing methods (ETD)

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
# Expression:
# 
# $$
#   y(t_{k+1}) = e^{-h \lambda} y(t_k) +
#   h \phi_1 (-\lambda h) g(y(t_k), t_k) +
#   \left[g(a_k, t_{k+1}) - g(y(t_k), t_k) \right] h \phi_2 (-\lambda h) + \\
#   + O(h^3) \\
#   \text{with } a_k = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) \frac{1-e^{-h\lambda}}{\lambda}.
# $$
# 
# Convergence table:
# 
# | n | h = $\frac{1}{h}$ | $\tau(0,h)$ | q = $\frac{tau(0,h)}{tau(0, 2h)}$ | $log_4 ^q$|
# |---|-----------------|-----------|---------------------------------|-------|
#  | 128 | 0.0078125 | 4.186569175362864e-08 | - | 
#  | 256 | 0.00390625 | 1.0575183428604418e-08 | 1.985085775819591 | 
#  | 512 | 0.001953125 | 2.652380943352073e-09 | 1.9953227875115886 | 
#  | 1024 | 0.0009765625 | 6.638462730912398e-10 | 1.9983668943519293 |
# 
# ### Exponential - Midpoint rule
# 
# Expression:
# 
# $$
# y(t_{k+1}) = e^{-h \lambda} y(t_k) + g(y(t_k), t_k)\int_{t_k}^{t_{k+1}} e^{-\lambda(t_{k+1}-\tau)} d \tau + \\
#   + 2\frac{g\left(b_k, t_k + \frac{h}{2}\right) - g(y(t_k), t_k)}{h} \int_{t_k}^{t_{k+1}} (\tau - t_k) e^{-\lambda(t_{k+1}-\tau)} d \tau + O(h)\int_{t_k}^{t_{k+1}} (\tau - t_k) e^{-\lambda(t_{k+1}-\tau)} d \tau \\
#   \text{with } b_k =e^{-\frac{h \lambda}{2}}y(t_k) + g(y(t_k), t_k) \frac{h}{2} \phi_1\left( -\frac{\lambda h}{2} \right).
# $$
# 
# Convergence table:
# 
# | n | h = $\frac{1}{h}$ | $\tau(0,h)$ | q = $\frac{tau(0,h)}{tau(0, 2h)}$ | $log_4 ^q$|
# |---|-----------------|-----------|---------------------------------|-------|
#  | 128 | 0.0078125 | 2.9740964063024178e-08 | - | 
#  | 256 | 0.00390625 | 6.3603379351490075e-09 | 2.225276088173374 | 
#  | 512 | 0.001953125 | 1.4582129219398166e-09 | 2.1249020291594443 | 
#  | 1024 | 0.0009765625 | 3.4828753076032726e-10 | 2.065850662914468 | 
# 
# ### New deduction - Trapezoidal rule
# 
# Expression:
# 
# $$
# y(t_{k+1}) = e^{-h \lambda}y(t_k) + \frac{h}{2} \left[ e^{-\lambda h} g(y(t_k), t_k) + g(a_k, t_{k+1}) \right] +  O(h^3) \\
#     \text{with } a_k = e^{-h \lambda}y(t_k) + g(y(t_k), t_k) h \phi_1 (-\lambda h).
# $$
# 
# Convergence table:
# 
# | n | h = $\frac{1}{h}$ | $\tau(0,h)$ | q = $\frac{tau(0,h)}{tau(0, 2h)}$ | $log_4 ^q$|
# |---|-----------------|-----------|---------------------------------|-------|
#  | 128 | 0.0078125 | 0.0004242643044311458 | - | 
#  | 256 | 0.00390625 | 0.00010714498082271644 | 1.9853990333325726 | 
#  | 512 | 0.001953125 | 2.6871031228085582e-05 | 1.9954406751889993 | 
#  | 1024 | 0.0009765625 | 6.725136514989377e-06 | 1.9984162299862431 | 
# 
# ### New deduction - Midpoint rule
# 
# Expression:
# 
# $$
# y(t_{k+1}) = e^{-h\lambda}y(t_k) + h e^{- \frac{h\lambda}{2}} g\left(b_k , t_k + \frac{h}{2}\right) + O(h^3) \\
#     \text{with } b_k =e^{-\frac{h \lambda}{2}}y(t_k) + g(y(t_k), t_k) \frac{h}{2} \phi_1\left( -\frac{\lambda h}{2} \right),
# $$
# 
# Convergence table:
# 
# | n | h = $\frac{1}{h}$ | $\tau(0,h)$ | q = $\frac{tau(0,h)}{tau(0, 2h)}$ | $log_4 ^q$|
# |---|-----------------|-----------|---------------------------------|-------|
#  | 128 | 0.0078125 | 0.00021050633676356068 | - | 
#  | 256 | 0.00390625 | 5.346923320679979e-05 | 1.977082770096472 | 
#  | 512 | 0.001953125 | 1.34290321535252e-05 | 1.9933536556922617 | 
#  | 1024 | 0.0009765625 | 3.362162453383888e-06 | 1.
