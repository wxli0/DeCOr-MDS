#######################
# - Quality control
# - Norùalize to Rank or Quantile-transformed data
#######################

setwd(".")
library(MASS)

## Quality Control
# Filter out if more than 10 non-zero fields, in Counts
v13lqphylotypecounts_rs <- read.csv("./data/v13lqphylotypecounts_rs.csv", header=FALSE)
v13lqphylotypecounts_rs_b <- v13lqphylotypecounts_rs[rowSums(v13lqphylotypecounts_rs!=0)>=10,]
c(nrow(v13lqphylotypecounts_rs_b),ncol(v13lqphylotypecounts_rs_b))
v13lqphylotypecounts_rs_c <- v13lqphylotypecounts_rs_b[,colSums(v13lqphylotypecounts_rs_b!=0)>=10]
c(nrow(v13lqphylotypecounts_rs_c),ncol(v13lqphylotypecounts_rs_c)) #2255,425
# # write to file
# write.table(v13lqphylotypecounts_rs_c,file="v13lqphylotypecounts_rs_c.csv",row.names=FALSE,col.names=FALSE,sep=",")

# Subset pheno accordingly
v13lqphylotypePheno_rs <- read.csv("./data/v13lqphylotypePheno_rs.csv")
c(nrow(v13lqphylotypePheno_rs),ncol(v13lqphylotypePheno_rs))
v13lqphylotypePheno_rs_c <- v13lqphylotypePheno_rs[rowSums(v13lqphylotypecounts_rs!=0)>=10,]
c(nrow(v13lqphylotypePheno_rs_c),ncol(v13lqphylotypePheno_rs_c)) #2255,24
# # write pheno subset to file
# write.csv(v13lqphylotypePheno_rs_c,file="v13lqphylotypePheno_QIHP.csv",row.names=FALSE)

  nSamples <- nrow(v13lqphylotypecounts_rs_c)
  nMeta    <- ncol(v13lqphylotypecounts_rs_c)

######
# Turn Counts into Ranks
######

# Counts into ranks, by sample
v13lqphylotypeQuantRank <- c()

for (i in 1:nSamples) {
  a<- c(as.matrix(v13lqphylotypecounts_rs_c[i,]))
  # Ranks
    tmp <- rank(a)
    tmp <- tmp/max(tmp)
  # Aggregate to resulting normalized matrix
    v13lqphylotypeQuantRank <- rbind(v13lqphylotypeQuantRank, tmp)
}
# normalized matrix, Rank
c(nrow(v13lqphylotypeQuantRank),ncol(v13lqphylotypeQuantRank)) # 2255  425
write.table(v13lqphylotypeQuantRank,file="v13lqphylotypeQuantRank.csv",row.names=FALSE,col.names=FALSE,sep=",")

######
# Turn Counts into Z quantiles
######

# Counts into ranks, by sample
v13lqphylotypeQuantZ <- c()

for (i in 1:nSamples) {
  a<- c(as.matrix(v13lqphylotypecounts_rs_c[i,v13lqphylotypecounts_rs_c[i,]!=0]))
  # Estimate distribution parameters
    fittpZ<-fitdistr(x=a, densfun = 'normal')
    ###estimateZ <- c(estimateZ,fittpZ$estimate)
    ###estimateZ.sd <- c(estimateZ.sd,fittpZ$sd)
  # Quantiles
    tmpZ <- pnorm(c(as.matrix(v13lqphylotypecounts_rs_c[i,])), fittpZ$estimate)
    tmpZ[tmpZ==min(tmpZ)] <- 0.
  # Aggregate to resulting normalized matrix
    v13lqphylotypeQuantZ <- rbind(v13lqphylotypeQuantZ, tmpZ)
}
# normalized matrix, Rank
c(nrow(v13lqphylotypeQuantZ),ncol(v13lqphylotypeQuantZ)) # 2255  425
write.table(v13lqphylotypeQuantZ,file="v13lqphylotypeQuantZ.csv",row.names=FALSE,col.names=FALSE,sep=",")


######
# Counts into NB or E Quantiles
####
v13lqphylotypeQuantNB <- c()
v13lqphylotypeQuantE <- c()
estimates <- estimates.sd <- c()
estimateE <- estimateE.sd <- estimateEc <- estimateEc.sd <- estimateEs <- estimateEs.sd <- c()
for (i in 1:2255)
{
  a<- c(as.matrix(v13lqphylotypecounts_rs_c[i,v13lqphylotypecounts_rs_c[i,]!=0]))
  fitNB<-fitdistr(x=a, densfun = 'negative binomial')
  fittpE<-fitdistr(x=a, densfun = 'exponential')
  estimates <- rbind(estimates,fitNB$estimate)
  estimates.sd <- rbind(estimates.sd,fitNB$sd)
  estimateE <- c(estimateE,fittpE$estimate)
  estimateE.sd <- c(estimateE.sd,fittpE$sd)
  tmp <- pnbinom(c(as.matrix(v13lqphylotypecounts_rs_c[i,])),
                                   size = fitNB$estimate[1],
                                   mu=fitNB$estimate[2])
  tmpE <- pexp(c(as.matrix(v13lqphylotypecounts_rs_c[i,])),
                                   rate = fittpE$estimate)
  tmp[tmp==min(tmp)] <- 0.
  tmpE[tmpE==min(tmpE)] <- 0.
  v13lqphylotypeQuantNB <- rbind(v13lqphylotypeQuantNB, tmp)
  v13lqphylotypeQuantE <- rbind(v13lqphylotypeQuantE, tmpE)
}

# normalized matrix, NB
c(nrow(v13lqphylotypeQuantNB),ncol(v13lqphylotypeQuantNB))
write.table(v13lqphylotypeQuantNB,file="v13lqphylotypeQuantNB.csv",row.names=FALSE,col.names=FALSE,sep=",")
# normalized matrix, E
c(nrow(v13lqphylotypeQuantE),ncol(v13lqphylotypeQuantE)) #2255,425
write.table(v13lqphylotypeQuantE,file="v13lqphylotypeQuantE.csv",row.names=FALSE,col.names=FALSE,sep=",")



