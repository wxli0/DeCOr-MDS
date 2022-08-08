library(MASS)
# library(HelpersMG)

Xe_df = data.frame(read.csv("outputs/hmp_Xe_dim11.txt", sep=" ", header=FALSE))
hmp_df = data.frame(read.csv("./data/hmp_v13lqphylotypePheno_QIHP.csv"))
# print(head(Xe_df))
# print(head(hmp_df))

phenos = list("THROAT", "EARS", "STOOL", "NOSE", "ELBOWS", "MOUTH", "VAGINA")

for (pheno in phenos) {
    pheno_index = grep(pheno, colnames(hmp_df))
    pheno_ind = hmp_df[, pheno_index]
    model = glm(pheno_ind ~  V1+V2+V3+V4+V5+V6+V7+V8+V9+V10, data= Xe_df, family='binomial')
    model = stepAIC(model, direction = "backward", trace=0)
    aic = extractAIC(model, show.option=TRUE)[2]
    print(paste("AIC for", pheno, "is:", aic))

}