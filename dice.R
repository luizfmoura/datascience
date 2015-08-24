#simula 2 dados lançados de forma totalmente aleatória
roll <- function(seq = 1:6){
  dice <- sample(seq,2,TRUE)
  sum(dice)
}


#probabilidade aumentada para o numero 6
roll_6 <- function(seq = 1:6){
  dice <- sample(seq,2,replace = TRUE, prob = c(1/8,1/8,1/8,1/8,1/8,3/8))
  sum(dice)
}

y <- replicate(10000,roll())
x <- replicate(10000,roll_6())
#qplot(y, binwidth = 1)
qplot(x, binwidth = 1)