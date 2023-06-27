#!/usr/bin/env python
# coding: utf-8

# # Motivation - Stiffness
# 
# The reason for studying exponential methods is that those are good with $\textbf{stiff differential equations}$ in terms of precision and how small the time step is required to be to achieve good accuracy. 

# ## Cauchy problem
# 
# A $\textbf{Cauchy problem}$ is a ordinary differential equation (ODE) with initial conditions. Being its standard scalar form:
# 
# $\begin{cases}
#     y'(t) = f(y(t), t), t \in (t_0, T) \\
#     y(t_0) = y_0 \in \mathbb{K} \text{,}
# \end{cases}$
# 
# with $\mathbb{K}$ a field, $f$ function with image in $\mathbb{K}$ and $t_0, T \in \mathbb{R}$.
# 
# Sometimes, it is convenient to separate the linear part of $f$ as indicated below:
# 
# $$\begin{equation*}
#     f(y(t), t) = g(y(t), t) - \lambda y(t) \text{,}
# \end{equation*}$$
# 
# with $\lambda \in \mathbb{K}$ or $\mathscr{M}_{N \times N}(\mathbb{K})$.
# 
# So the system is:
# 
# $\begin{cases}
#     y'(t) + \lambda y(t) = g(y(t), t), t \in (t_0, T) \\
#     y(0) = y_0 
#     \text{.}
# \end{cases}$
# 
# In this project, the stiff ones were those addressed.
# 
# Notation as in [1].

# ## Stiffness
# 
# The error of the approximation given by a method trying to estimate the solution of a Cauchy problem is always given by a term multiplied by a higher derivative of the exact solution, because of the Taylor expansion with Lagrange form of the remainder. In that way, if that is enough information about this derivative, the error can be estimated. 
# 
# If the norm of the derivative increases with the time, but the exact solution doesn't, that is possible that the error dominates the approximation and the precision is lost. Those problems are called $\textbf{stiff equations}$.
# 
# Between them, there are the $\textbf{stiff differential equations}$, that have exact solution given by the sum of a $\textit{transient solution}$ with a $\textit{steady state solution}$.
# 
# The $\textbf{transient solution}$ is of the form:

# $$\begin{align*}
# e^{-ct} \text{, with c >>1, }
# \end{align*}$$

# which is known to go to zero really fast as t increases. But its $n$th derivative

# $$\begin{align*}
# \mp c^{n}e^{-ct}
# \end{align*}$$

# doesn't go as quickly and may increase in magnitude.

# The $\textbf{steady state solution}$, however, as its name implies, have small changes as time passes, with higher derivative being almost constant zero.

# In a system of ODE's, these characteristics are most common in problems in which the solution of the initial value problem is of the form

# $$\begin{align*}
# e^{A}
# \end{align*}$$

# being $A$ a matrix such that $\lambda_{min}$ and $\lambda_{max}$ are the eigenvalue with minimum and maximum value in modulus and $\lambda_{min} << \lambda_{max}$. On the bigger magnitude eigenvalue direction, the behaviour is very similar to the transient solution, having drastic changes over time and on the smaller one, comparing to that, changes almost nothing as times passes, like the steady state solution.

# Work around these problems and being able to accurately approximate these so contrasting parts of the solutions requires more robust methods than the more classic and common one-step methods addressed at the beginning of the study of numerical methods for Cauchy problems. For the systems, it is also required that that is a precise way to calculate the exponential of a matrix. 
# 
# In this project, we studied the $\textbf{exponential methods}$, their capabilities to deal with these problems and the comparision with other simpler methods.

# Definition from [2].
