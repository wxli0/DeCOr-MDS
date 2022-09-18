### Compute AICs for the eigenvectors computed from test_hmp.py (current dimension is 11)

library(MASS)
library(nnet)
# library(HelpersMG)

hmp_df = data.frame(read.csv("./data/hmp_v13lqphylotypePheno_QIHP.csv"))
# print(head(Xe_df))
# print(head(hmp_df))

phenos = list("THROAT", "EARS", "STOOL", "NOSE", "ELBOWS", "MOUTH", "VAGINA")
methods = list("QE_nSimplices_cMDS", "NB_nSimplices_cMDS", "NB_cMDS", "QE_cMDS")

for (method in methods) {
    Xe_df = data.frame(read.csv(paste0("./outputs/hmp_", method, "_coord.txt"), sep=" ", header=FALSE))

    for (pheno in phenos) {
        pheno_index = grep(pheno, colnames(hmp_df))
        pheno_ind = hmp_df[, pheno_index]
        # print(paste(pheno, length(pheno_ind)))
        glm_model = glm(pheno_ind ~  V1+V2+V3+V4+V5+V6+V7+V8+V9+V10, data=Xe_df, family='binomial')
        glm_model = stepAIC(glm_model, direction = "backward", trace=0)
        glm_aic = extractAIC(glm_model, show.option=TRUE)[2]
        print(paste(method, "GLM model AIC for", pheno, "is:", glm_aic))
    }
    
    phenoSingleVector <- THROAT+STOOL*2+MOUTH*3+EARS*4+NOSE*5+ELBOWS*6+VAGINA*7
    phenoSV <- phenoSingleVector
    phenoSV[phenoSingleVector==1] <- "THROAT"
    phenoSV[phenoSingleVector==2] <- "STOOL"
    phenoSV[phenoSingleVector==3] <- "MOUTH"
    phenoSV[phenoSingleVector==4] <- "EARS"
    phenoSV[phenoSingleVector==5] <- "NOSE"
    phenoSV[phenoSingleVector==6] <- "ELBOWS"
    phenoSV[phenoSingleVector==7] <- "VAGINA"
    phenoSV <- as.factor(phenoSV)

    AIC.SITE <- AIC(stepAIC(multinom(phenoSV ~ V1+V2+V3+V4+V5+V6+V7+V8+V9+V10,data = data0)))    
}

# benchmark with other methods