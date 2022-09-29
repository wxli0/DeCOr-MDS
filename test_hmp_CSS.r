library(metagenomeSeq)

# TODO: missing v13lqphylotypecounts_rs_c.taxa, v13lqphylotypecounts_rs_c.clin
#
## CSS-MDS, hmp
#
## a- Data and formatting
hmp <- load_meta("./data/hmp_v13lqphylotypecounts_rs_c.csv")
hmp$counts[1:5,1:5]
taxa = read.delim("./data/hmp_v13lqphylotypecounts_rs_c.taxa",stringsAsFactors = F)[, 2] 
otu =  read.delim("./data/hmp_v13lqphylotypecounts_rs_c.taxa",stringsAsFactors = F)[, 1]
# clin = load_phenoData("v13lqphylotypecounts_rs_c.clin",tran = TRUE)
#
# phenotypeData = as(clin, "AnnotatedDataFrame")
OTUdata =       as(data.frame(otu,row.names = 1:length(otu)), "AnnotatedDataFrame") #OTUdata = as(lung$taxa, "AnnotatedDataFrame")
varLabels(OTUdata) = "taxa"
# obj = newMRexperiment(hmp$counts,phenoData=phenotypeData,featureData=OTUdata)
obj = newMRexperiment(hmp$counts, featureData=OTUdata)


## b- Preparatory work
p= cumNormStat(obj) #data(lungData) ; p = cumNormStat(lungData)

# Calculate the scaling factors by running cumNorm
obj = cumNorm(obj, p=p) # lungData = cumNorm(lungData, p = p)

# CSS-normalized matrix
mat = MRcounts(obj, norm=TRUE, log=TRUE)

#
## MDS (euclidean distance)
#
csse <- cmdscale(dist(t(mat), method = "euclidean"),k = 20)


