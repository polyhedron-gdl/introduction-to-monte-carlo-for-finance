
require(binhf)

file_name = paste(getwd(),'data', 'ts_sp_mib.csv', sep="/")
ts <- read.csv(file_name, sep=';')
ts$Date = as.Date(ts$Date)

attach(ts)

shift_price    <- shift(Price,1)
shift_price[1] <- 0
yield <- log(shift_price/Price)
yield[1] <- 0

hist(yield, prob=TRUE, ylim=c(0,100), breaks=20)
curve(dnorm(x, mean(yield), sd(yield)), add=TRUE, col="darkblue", lwd=2)

wiener = function( n, tt ) {
  e = rnorm( n, 0, 1 )
  x = c(0,cumsum( e )) / sqrt(n)
  y = x[ 1+floor( n * tt ) ]
  return( list( x = x, y = y ) )
}

time_step  <- 1000
nsim       <- 100
t          <- seq(0,1,1/time_step)
delta_t    <- t[2]-t[1]

m          <- mean(yield)
s          <- sd(yield)

# volatility
sigma   <- s * sqrt(250) * sqrt(delta_t)
# drift 
drift   <- (m + .5*sigma*sigma)*delta_t

S0      <- Price[length(Price)]

paths <- wiener(time_step, t )$y
S1    <- S0*exp(drift + sigma * paths)

lower = 0.99*min(S1)
upper = 1.01*max(S1)

plot( S1, type="l",ylim = c(lower,upper), xlab = "Time", ylab="" )

for(i in 1:nsim){
    paths <- wiener(time_step, t )$y
    S     <- S0*exp(drift + sigma * paths)
    lines( S , col=sample(rainbow(100)))
}

# Adding confidence level curve

# 90-th percentile
p <- qnorm(0.9)
y1=S0*exp(drift+p*sigma*sqrt(t))
y2=S0*exp(drift-p*sigma*sqrt(t))
lines(y1, lwd=2,col='blue')
lines(y2, lwd=2,col='blue')

# 99-th percentile
p <- qnorm(0.99)
y1=S0*exp(drift+p*sigma*sqrt(t))
y2=S0*exp(drift-p*sigma*sqrt(t))
lines(y1, lwd=2,col='red')
lines(y2, lwd=2,col='red')

# 99.99-th percentile
p <- qnorm(0.9999)
y1=S0*exp(drift+p*sigma*sqrt(t))
y2=S0*exp(drift-p*sigma*sqrt(t))
lines(y1, lwd=2,col='green')
lines(y2, lwd=2,col='green')

