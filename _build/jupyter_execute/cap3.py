#!/usr/bin/env python
# coding: utf-8

# # Important concepts for the study of exponential methods

# In this chapter, a review on matrix exponential is done, essencial for the linear problems treated here, followed by a section of $\phi$ functions, because of its need when applying exponential methods in systems of ODE with initial value. Besides that, the format of the treated problem is shown.

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
