#!/usr/bin/env python
# coding: utf-8

# # 3.2 Chapter 2: Classical methods
# 
# In order to show that the exponential methods improve in dealing with Stiff problems, that is necessary to know how the previows methods deal with them, so a review on the theory of the classical methods is made in this chapter. In particular there will be focus on the one step methods. All the information is from [3].

# ## One step methods for ODE 

# In order to find a approximation for the solution of the problem
# $\begin{cases}
# y'(t) = f(t, y(t)), t \in [t_0,T] \\
# y(t_0)=y_0 \text{,}
# \end{cases}$

# they are of the form:
# $$
# y_{k+1} = y_{k} + h \phi (t_{k},y_{k},t_{k+1},y_{k+1},h) \text{,}
# $$

# with $$k = 0, 1, ..., n-1;$$
# $$
# N \in \mathbb{N}; h = \frac{T-t_0}{N}; \\
# \{t_i = t_0 + ih : i = 0, 1, ..., N\}; \\ 
# y_n \thickapprox y(t_n) .
# $$

# To analyse the method, there is a model problem
# $\begin{cases}
#     y'(t) = - \lambda y(t) \text{ ; } t \in[t_0,T]\\ 
#     y(t_0)=y_0,\\
# \end{cases}$
# 
# whose solution is $y(t) = y_0 e^{-\lambda (t-t_0)}$
# with $\lambda > 0.$

# If that is possible to manipulate the method so that, for this problem, can be written as
# $$
# y_{k+1} = \zeta(\lambda,h) y_k,
# $$
# then $$\zeta(\lambda,h)$$ is called $\textbf{amplification factor}$ of the method.

# By induction, it gives
# $$
# y_{k+1} = \zeta(\lambda, h)^{k+1} y_0.
# $$

# It is well known that this expression only converges as k goes to infinity if 
# $$
# |\zeta(\lambda, h)| < 1
# $$
# and then converges to zero.

# When it occurs, i.e., $k \rightarrow \infty \Rightarrow y_k \rightarrow 0$ such as the exact solution $y(t) = y_0 e^{-\lambda (t-t_0)}$, it is said that there is $\textbf{stability}$.

# The inequation gives a interval for which values of $\lambda h$, $|\zeta(\lambda, h)|<1$, called $\textbf{interval of stability}$.

# And if the interval of stability contains all the points $z$ such that $Re(z) < 0$, the method is said $\textbf{A-stable}$.

# The reason for taking this specific problem is that it models the behaviour of the difference between the approximation and the solution on a small neighbourhood of any Cauchy problem:

# Taking
# $\begin{cases}
#     y'(t) = f(y(t), t), t \in (t_0, T) \\
#     y(t_0) = y_0 \in \mathbb{K}
# \end{cases}$

# and a approximation $z$ of the solution $y$, doing
# $$
# \sigma(t) = z(t) - y(t) \Rightarrow
# $$

# $$
# \dot{\sigma}(t) = \dot{z}(t) - \dot{y}(t) = f(z(t), t) - f(y(t), t) \Rightarrow
# $$

# $$
# \dot{\sigma}(t) + \dot{y}(t) = \dot{z}(t) = f(z(t), t) = f(y(t)+\sigma(t), t)
# $$
# 
# $$
#  = f(y(t), t) + \sigma(t)\frac{\partial f}{\partial y} + O(\sigma^2(t)),
# $$

# so
# \begin{cases}
#     \dot{\sigma}(t) \approx \sigma(t) \frac{\partial f}{\partial y} (y(t), t) \\
#     \sigma(t_k) = \sigma_k.
# \end{cases}

# Other important definitions are:
# 
# $\textbf{Local truncation error:}$ Is the difference between the exact expression and its numerical approximation in a certain point and with a certain domain discretization. If the domain is equally spaced by $h$ is often denoted by $\tau(h,t_0)$ being $t_0$ the point.

# $\textbf{Order of the local truncation error:}$ the local truncation error (which depends on the $h$ spacing of the discretized domain) $\tau(h)$ has order $n \in \mathbb{N}$ if $\tau(h) = O(h^n) $, i.e., if there is constant $M \in \mathbb{R}$ and $h_0 \in \mathbb{R}$ such that $\tau(h) \leq M h^n$, $\forall h \leq h_0$.

# $\textbf{Global error:}$ Is the difference between the approximation given by the method for the solution of the problem on a certain point and the exact one (unlike the local truncation error, here we take the solution we got, not the expression used to find the approximation).

# $\textbf{Consistency:}$ The method is said consistent if $\lim _{h \to 0} \frac{1}{h}\tau(h,x_0) = 0$.
# 
# $\textbf{Obs.:}$ For consistency, we usually only analyse for the linear part of the Cauchy problem, since this is the part that most influences in the consistency.
# 
# $\textbf{Order of consistency:}$ is the smallest order (varying the points at which the local error is calculated) of the local truncation error.

# $\textbf{Convergence:}$ A numerical method is convergent if, and only if, for any well-posed Cauchy problem and for every $t \in (t_0, T)$,
# $$\lim_{h \to 0} e_k = 0$$
# with $t - t_0 = kh$ fixed and $e_k$ denoting the global error on $t_k$ (following the past notation).

# $\textbf{Theorem:}$ A one-step explicit method given by
# $$
# y_0 = y(t_0) \\
# y_{k+1} = y_{k} + h \phi (t_{k},y_{k},h)
# $$
# such that $\phi$ is Lipschitzian in y, continuous in their arguments, and consistent for any well-posed Cauchy problem is convergent. Besides that, the convergence order is greater or equal to the consistency order.
# 
# $\textit{Prove:}$ [3] pág 29-31.

# ## Examples

# Euler method: 
# 
# $$
#     \phi (t_{k},y_{k},h) = f(t_{k},y_{k})
# $$
# 
# Modified Euler method: 
# 
# $$
#     \phi (t_{k},y_{k},h) = \frac{1}{2} \left[ f(t_{k},y_{k}) + f(t_{k+1},y_{k} + h f(t_{k},y_{k})) \right]
# $$
# 
# Midpoint method: 
# 
# $$
#     \phi (t_{k},y_{k},h) = f(t_{k} + \frac{h}{2},y_{k} + \frac{h}{2} f(t_{k},y_{k}))
# $$
# 
# Classic Runge-Kutta (RK 4-4): 
# 
# $$
#     \phi (t_{k},y_{k},h) = \frac{1}{6} \left( \kappa_1 + 2 \kappa_2 + 2 \kappa_3 + \kappa_4 \right), \text{with }\\
#     \kappa_1 = f(t_{k},y_{k})\\
#     \kappa_2 = f(t_{k} + \frac{h}{2},y_{k} + \frac{h}{2} \kappa_1)\\
#     \kappa_3 = f(t_{k} + \frac{h}{2},y_{k} + \frac{h}{2} \kappa_2)\\
#     \kappa_4 = f(t_{k} + h, y_{k} + h \kappa_3)
# $$

# ## Euler method

# Further detailing this explicit one-step method of
# 
# $$
#     \phi (t_{k},y_{k},h) = f(t_{k},y_{k}),
# $$
# 
# an analysis on stability, convergence and order of convergence is done.

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
# $$
#     |1 - h \lambda| < 1 \text{ and } N \text{ is fixed,}
# $$ 
# 
# it converges to zero 
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

# ### Convergence
# Since
# $$
# \lim_{m \to +\infty} \left(1 + \frac{p}{m} \right)^m = e^p,
# $$
# and h = $\frac{T-t_0}{N}$, for $y_N$ we have
# $$
# \lim_{N \to +\infty} y_N = \lim_{N \to +\infty} \left(1 - h \lambda \right)^N y_0 = \lim_{N \to +\infty} \left(1 - \frac{(T-t_0) \lambda}{N} \right)^N y_0.
# $$
# It is reasonable to take $p = -(T-t_0) \lambda$ and conclude that the last point estimated by the method will converge to
# $$
# y_0e^{-\lambda (T-t_0)}.
# $$
# Which is precisely $y(T)$ and proves the convergence.

# ### Order of convergence
# 
# Being $\tau(h, t_k)$ the local truncation error.
# 
# From
# $$
#     y(t_{k+1}) = y(t_k) + h f(y(t_k),t_k) + O(h^2),
# $$
# 
# we have
# $$
#     h \tau(h, t_k) \doteq \frac{y(t_{k+1}) - y(t_k)}{h} - f(t_k, y(t_k)) = O(h^2),
# $$
# 
# so
# $$
#     \tau(h, t_k) = O(h).
# $$
# 
# Since for one step methods the order of convergence is the order of the local truncation error, the order is of $O(h)$, order 1.
