# 3.1 Stiff equation

## 3.1.1 Cauchy problem

A $\textbf{Cauchy problem}$ is a ordinary differential equation (ODE) with initial conditions. Being its standard scalar form:

\begin{cases}
    y'(t) = f(y(t), t), t \in (t_0, T) \\
    y(t_0) = y_0 \in \mathbb{K}
\end{cases}

with $\mathbb{K}$ a field, $f$ function and $t_0, T \in \mathbb{R}$.

Sometimes, it is convenient to separate the linear part of $f$ as indicated below:

\begin{equation*}
    f(y(t), t) = g(y(t), t) - \lambda y(t)
\end{equation*}

with $\lambda \in \mathbb{R}$.

So the system is:

\begin{cases}
    y'(t) + \lambda y(t) = g(y(t), t), t \in (t_0, T) \\
    y(0) = y_0
\end{cases}

In this project, the stiff ones were those addressed.

## 3.1.2 Stiffness

In the field of numerical methods, $\textbf{stiffness}$ is a concept that for a problem and a numerical method, assigns the property of this problem being solved unstablely by the method.

Stiff scalar Cauchy problems have solution that varies slowly in one region and in another varies rapidly.