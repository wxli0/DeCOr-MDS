### Compute AICs for the eigenvectors computed from test_hmp.py (current dimension is 11)

library(MASS)
library(nnet)
# library(HelpersMG)

# nSimplices + MDS
Xe_df = data.frame(read.csv("outputs/hmp_Xe_dim11.txt", sep=" ", header=FALSE))
hmp_df = data.frame(read.csv("./data/hmp_v13lqphylotypePheno_QIHP.csv"))
# print(head(Xe_df))
# print(head(hmp_df))

phenos = list("THROAT", "EARS", "STOOL", "NOSE", "ELBOWS", "MOUTH", "VAGINA")

for (pheno in phenos) {
    pheno_index = grep(pheno, colnames(hmp_df))
    pheno_ind = hmp_df[, pheno_index]
    # print(paste(pheno, length(pheno_ind)))
    glm_model = glm(pheno_ind ~  V1+V2+V3+V4+V5+V6+V7+V8+V9+V10, data=Xe_df, family='binomial')
    glm_model = stepAIC(glm_model, direction = "backward", trace=0)
    glm_aic = extractAIC(glm_model, show.option=TRUE)[2]
    print(paste("GLM model AIC for", pheno, "is:", glm_aic))

    mn_model = multinom(pheno_ind ~ V1+V2+V3+V4+V5+V6+V7+V8+V9+V10, data=Xe_df, trace=0)
    mn_model = stepAIC(mn_model, direction = "backward", trace=0)
    mn_aic = extractAIC(mn_model, show.option=TRUE)[2]
    print(paste("MN model AIC for", pheno, "is:", mn_aic))
}

# benchmark with other methods