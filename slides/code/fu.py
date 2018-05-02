def f(u, S0, K, r, sigma, T):
    m      = (r - .5*sigma*sigma)*T
    s      = sigma*sqrt(T)
    f_u    = exp(-r*T) * 
             np.maximum(S0*exp(scnorm.ppf(u, m, s))-K,0)
    return f_u    

u      = rand(1000000)
f_u    = f(u,S0,K,r,sigma,T)

print mean(f_u)