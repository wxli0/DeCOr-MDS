### Compute AICs for the eigenvectors computed from test_hmp.py (current dimension is 11)

library(MASS)
library(nnet)
# library(HelpersMG)

data_dir = "../data"
output_dir = "../outputs"
hmp_df = data.frame(read.csv(file.path(data_dir, "hmp_v13lqphylotypePheno_target_rs_c.csv")))


# phenos = list("THROAT", "STOOL", "MOUTH", "EARS", "NOSE", "ELBOWS", "VAGINA")
target_phenos = list("STOOL", "VAGINA")
methods = list("QE_nSimplices_target_cMDS", "NB_nSimplices_target_cMDS", "QE_target_MDS_cMDS", "NB_target_MDS_cMDS")


for (method in methods) {
    Xe_df = data.frame(read.csv(file.path(output_dir, paste0("hmp_", method, "_axes.txt")), sep=" ", header=FALSE))

    for (pheno in target_phenos) {
        pheno_index = grep(pheno, colnames(hmp_df))
        pheno_ind = hmp_df[, pheno_index]
        glm_model = glm(pheno_ind ~  V1+V2+V3+V4+V5+V6+V7+V8+V9+V10, data=Xe_df, family='binomial')
        glm_aic = stepAIC(glm_model, direction = "backward", trace=0)
        glm_aic = extractAIC(glm_aic, show.option=TRUE)[2]
        print(paste(method, "GLM model AIC for", pheno, "is:", glm_aic))
    }
    
    phenoSingleVector <- hmp_df[["STOOL"]]*2+hmp_df[["VAGINA"]]*7

    phenoSV <- phenoSingleVector
    phenoSV[phenoSingleVector==2] <- "STOOL"
    phenoSV[phenoSingleVector==7] <- "VAGINA"
    phenoSV <- as.factor(phenoSV)

    mn_model = multinom(phenoSV ~ V1+V2+V3+V4+V5+V6+V7+V8+V9+V10, data = Xe_df)
    mn_aic <- stepAIC(mn_model, direction = "backward", trace=0)
    mn_aic = extractAIC(mn_aic, show.option=TRUE)[2]
    print(paste(method, "MN model AIC for is:", mn_aic)) 
}

# benchmark with other methods