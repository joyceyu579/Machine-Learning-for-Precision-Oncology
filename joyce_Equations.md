# IC50 

\begin{align*}
Fifty &= \frac{[\text{Top - Bottom}]}{2}
\end{align*}

<br>

\begin{equation}    
Y = \text{Bottom} + \frac{\text{Top} - \text{Bottom}}{1 + 10^{\left((\text{IC50} - X) \cdot \text{HillSlope} + \log\left(\frac{\text{Top} - \text{Bottom}}{\text{Fifty} - \text{Bottom}} - 1\right)\right)}}
\end{equation}


# Mean normalized growth rate inhibition

\begin{equation}
    \text{Normalized GR Inhibition} = 2^{\frac{\log_2\left(\frac{x(c)}{x_0}\right)}{\log_2\left(\frac{x_{\text{ctrl}}}{x_0}\right)}} - 1
\end{equation}


# logP coefficient

\begin{align*}
\log P &= \log_{10}(\text{Partition Coefficient}) \\\\
P &= \frac{[\text{concentration in organic phase}]}{[\text{concentration in aqueous phase}]}
\end{align*}

# Linear equation
\begin{equation}
y = w_1 x_1 + w_2 x_2 + w_3 x_3 + \cdots + w_n x_n + b
\end{equation}


# ReLu function
\begin{equation}
f(x) = max(0,x)
\end{equation}


# Vanilla gradient descent
\begin{equation}
W_{t+1} = W_t - \alpha \nabla W_t
\end{equation}

# Velocity in the idea of momentum
\begin{equation}
V_{t+1} = \beta W_t + (1-\beta)\nabla W_t
\end{equation}

# Momentum applied to gradient descent

\begin{equation}
W_{t+1} = W_t - \alpha \nabla V_t+1
\end{equation}

# RMSprop

<u> velocity with mean squared error of gradient: 

\begin{equation}
V_{t+1} = \beta W_t + (1-\beta)\nabla W_t^2
\end{equation}

<br>
<u> modified gradient descent: 
\begin{equation}
W_{t+1} = W_t - \alpha \frac{{\nabla W_t+1}}{\sqrt{V_{t+1} + \epsilon}}
\end{equation}

# Adam optimizer
<u>Moments:
\begin{equation}
\text{moment1}_t = \beta W_t + (1-\beta)\nabla W_t
\end{equation}

\begin{equation}
\text{moment2}_t = \beta W_t + (1-\beta)\nabla W_t^2
\end{equation}

<u>Estimated moments:

\begin{equation}
\hat{moment1_t} = \frac{{V_t}}{(1-\beta_1^t)}
\end{equation}

\begin{equation}
\hat{moment2_t} = \frac{{V_t}}{(1-\beta_2^t)}
\end{equation}

<br>

<u> modified gradient descent: 
\begin{equation}
W_{t+1} = W_t - \alpha \frac{{\hat{moment1_t}}}{\sqrt{\hat{moment2_t} + \epsilon}}
\end{equation}


