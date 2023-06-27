#!/usr/bin/env python
# coding: utf-8

# # Important concepts for the study of exponential methods

# In this chapter, a review on $\phi$ functions is done because of its need when applying exponential methods in systems of ODE with initial value. Besides that, the format of the treated problem is shown.
# 
# It is worth remembering the importance of the matrix exponential for the linear problems treated here, which is in the appendix.

# ## Linear problem

# The linear problem is, following with the used notation:
# 
# $$
# \begin{cases}
#     y'(t) + \lambda y(t) = g(y(t), t), t \in (t_0, T) \\
#     y(t_0) = y_0 
#     \text{,}
# \end{cases}
# $$
# 
# the one with $g \equiv 0.$

# So, generaly, it is of the form:
# 
# $$
# \begin{cases}
#     y'(t) = A y(t), t \in (t_0, T) \\
#     y(t_0) = y_0 
#     \text{,}
# \end{cases}
# $$
# 
# with $A \in \mathscr{M}_{N \times N}(\mathbb{C}), N \in \mathbb{N}$  (remembering that a matrix $1 \times 1$ is simply a number).

# Because $A y(t)$ is a $C^1$ function in $y$, continuous in $t$ and $t \in (t_0, T)$, a limited interval, by the existence and uniqueness theorem, there is a single solution of the problem.

# Since 
# 
# $$
#     \frac{d}{dt}y_0e^{A(t-t_0)} \doteq \lim_{h\to0} \frac{y_0e^{A(t-t_0+h)}-y_0e^{A(t-t_0)}}{h}
# $$
# 
# $$
#     = y_0e^{(t-t_0)A}\lim_{h\to0} \frac{e^{Ah}-I}{h} 
# $$
# 
# $$
#     = y_0e^{(t-t_0)A}\lim_{h\to0} \frac{Ae^{Ah}}{1} 
# $$
# 
# $$
#     = y_0e^{(t-t_0)A} \frac{Ae^{A0}}{1}
# $$
# 
# $$
#     = y_0e^{(t-t_0)A} A I = A y_0e^{(t-t_0)A} 
# $$

# using L'Hôpital's rule on the second equality and noting that $A(t-t_0+h) = A(t-t_0)+Ah$ and $A(t-t_0) \cdot Ah = (t-t_0)hAA = Ah \cdot A(t-t_0)$, so it was possible to apply the last proposition and make $e^{A(t-t_0+h)} = e^{A(t-t_0)} \cdot e^{Ah}$,

# taking
# 
# $$
#     y(t) = y_0e^{A(t-t_0)},
# $$

# $$
#     y'(t) = A y_0 e^{(t-t_0)A} = A y(t) \text{ and } y(t_0) = y_0 e^{(t_0-t_0)A} = y_0 I = y_0.
# $$

# So, the solution for the general linear problem is $y(t)=y_0 e^{A(t-t_0)}$.

# All information about matrix exponential is from [4].

# ## General problem

# Returning to the general case
# 
# $\begin{cases}
#     y'(t) + \lambda y(t) = g(y(t), t), t \in (t_0, T) \\
#     y(t_0) = y_0 
#     \text{,}
# \end{cases}$
# 
# there is the variation of constants formula:
# 
# $$
#     y(t) = e^{-t \lambda}y_0 + \int_{t_0}^t e^{-\lambda(t-\tau)} g(y(\tau), \tau) d\tau.
# $$
# 
# This well known implicit function, gives a solution of the problem. 
# 
# If the integral part can be solved, there is a explicit solution, and if the problem satisfies the hypotesis of the Piccard problem, being Lipschitz in $t$, this is the only solution.
# 
# This formula is the basis of all the exponential methods.

# ## $\phi$ functions
# 
# Before introducing exponential methods, it is useful to present the $\phi$ functions.
# 
# They are $\mathbb{C} \rightarrow \mathbb{C}$ functions defined as:
# 
# $$
#   \phi_0 (z) = e^z;
# $$
# 
# $$
#   \phi_n (z) = \int_{0}^{1} e^{(1-\tau)z} \frac{\tau^{n-1}}{(n-1)!} \,d\tau, n \geq 1.
# $$

# By integration by parts,
# 
# $$
#   \phi_{n+1} (z) = \int_{0}^{1} e^{(1-\tau)z} \frac{\tau^n}{n!} \,d\tau \\
#   = - \frac{e^{(1-1)z}}{z} \frac{1^n}{n!} + \frac{e^{(1-0)z}}{z} \frac{0^n}{l!} - \int_{0}^{1} -\frac{e^{(1-\tau)z}}{z} \frac{\tau^{n-1}}{(n-1)!} \,d\tau \\
#   = - \frac{1}{n!z} + \frac{1}{z}\int_{0}^{1} e^{(1-\tau)z} \frac{\tau^{n-1}}{(n-1)!} \,d\tau.
# $$

# Since
# 
# $$
#   \phi_n(0) = \int_{0}^{1} e^0 \frac{\tau^{n-1}}{(n-1)!} \,d\tau = \int_{0}^{1} \frac{\tau^{n-1}}{(n-1)!} \,d\tau = \frac{1^n}{n!} - 0 = \frac{1}{n!},
# $$

# $$
#   \phi_{n+1}(z) = \frac{\phi_n(z) - \phi_n(0)}{z}, \textbf{the recursive characterization}.
# $$

# By the properties of integral [5], if $h \in \mathbb{R}^*, t_k \in \mathbb{R}, t_k+h = t_{k+1},$
# 
# $$
#   \phi_n (z) = \int_{0}^{1} e^{(1-\tau)z} \frac{\tau^{n-1}}{(n-1)!} \,d\tau \\
#   = \frac{1}{h}\int_{0}^{h} e^{\frac{(h-\tau)z}{h}} \frac{\tau^{n-1}}{h^{n-1}(n-1)!} \,d\tau \\
#   = \frac{1}{h}\int_{t_k}^{t_k + h} e^{\frac{(h-\tau+t_k)z}{h}} \frac{(\tau - t_k)^{n-1}}{h^{n-1}(n-1)!} \,d\tau,
# $$

# $$
#   \phi_n (z) = \frac{1}{h^l}\int_{t_k}^{t_{k+1}} e^{\frac{1}{h}(t_{k+1}-\tau)z} \frac{(\tau - t_k)^{n-1}}{(n-1)!} \,d\tau.
# $$

# Information from [1].
