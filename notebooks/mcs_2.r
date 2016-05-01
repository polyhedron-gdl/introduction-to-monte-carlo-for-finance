
n <- 10000
x <- runif(n)
y <- runif(n)
inside <- x^2 + y^2 <= 1
pi <- 4*sum(inside) / n

plot(x,y,
     col=ifelse(inside,'blue','red'), cex=0.5, pch='.',
     main=sprintf("Bootstrap approximation of pi\nusing %s random samples, 
                   pi = %1.5f",n,pi)
    )
