#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Changes wrt. v0_9b.py: 
#   - anaCounts: . calculate wcCounts, Manhattan or Bray distances'matrix
#                . get matrix as similarity if wished, double center wherever necessary,
#                . find eigenvectors, eigenvalues and return
#   - wcCounts_similDist: calculate wcCounts distance or similarity matrix
#   - Manhattan_similDist: calculate Manhattan distance or similarity matrix
#   - Bray_similDist: calculate Bray-Curtis distance or similarity matrix
#
# Changes wrt. v0_9b.py: 
#   - WCS_covariance:	- made tolerant to missing values
#   - added COVariance: covariance matrix tolerant to missing values
#
#
# Contents:
#   PCAstd
#   PCAeigenstrat
#   PCAmcdeigstr
#   UCS_covariance
#   WCS_covariance
#   WEDS_covariance
#   PCAmcdREL
#   PCAmcd
#   PCAsph
#   ACParRedux
#   COVariance

# For all PCA functions:
# in data, columns should be variables (individuals) and rows should be observations (SNPs)
# dim of data is Lxn with L >> n

from scipy.spatial.distance import pdist, squareform
import warnings
import numpy as np
import sys
execfile("/data/clegrand/Travail/diss/py/Lib/robust_covariance_adapted_verbose_v0.5.py")
execfile("/data/clegrand/Travail/diss/py/Lib/ACPar_v5.py")
execfile("/data/clegrand/Travail/diss/py/Lib/cMDS.py")

def PCAstd(data1,mafCtrl=False,correl=False,extend=False):
    """
    Purpose: 
    --------
    Perform standard PCA
    
    In:
    ----------
    data1           Genotypes (0/1/2) Lxn matrix:  L rows of observations (SNPs)  x   n columns of variables (Individuals)
    mafCtrl=False   Default: take coding 0/1/2 as is.
    mafCtrl=True    Ensure that minor allele is coded 1. Recode 0/1/2 into 2/1/0 otherwise.
    correl=False    Default: Use covariance matrix.
    correl=True     Use correlation matrix instead of covariance matrix.
    extend=False    Default: Only the first 4 eigenvectors/values are returned, eigenvalues are normalized wrt sum of all eigenvalues.
    extend=True     location, scores and the whole matrice/vector of eigenvectors/values are returned.
   """
    from scipy import linalg as la
    import copy
    # MAF control : minor allele should be coded 1, not 0., so that 0/1/2 means homozygote common / heterozygote / homozygote rare
    if mafCtrl:
        p= np.mean(data1, axis=1) / 2.   # frequency per snp
        data=copy.deepcopy(data1)
        # recode 2/1/0 if frequency is higher than 0.5
        for i in range(np.shape(data1)[0]):
            if p[i] > 0.5: data[i,:]=2.-data1[i,:]
    else:
        data=data1
    # center data
    location = np.mean(data, axis=0) # mean of each column
    datac = data-location
    # covariance matrix
    C = np.cov(datac.T)
    if correl: C = np.corrcoef(datac.T)
    #print 'covariance/correlation matrix extract:' ; print C[:15,:15]
    # calculate eigenvectors & eigenvalues of the covariance matrix
    evals, evecs = la.eig(C)
    # sort by eigenvalue in decreasing order
    idx = np.argsort(abs(evals))[::-1]
    evecst = evecs[:,idx]
    evalst= evals[idx]
    # customize results and return
    if extend:
        #scores = np.dot(datac,evecst)
        return evalst,evecst,location,0 #scores
    else:
        # select the first 4 eigenvectors and -values, normalize eigenvalues
        xeigenvectors = evecst[:,:4]
        PC1to4unNormalized = evalst[:4]
        PC1to4 = evalst[:4]/sum(evalst)
        return PC1to4,xeigenvectors,0,0

def PCAeigenstrat(data,mafCtrl=False,extend=False,simpl=False,
                  diploid_biallelic=True,mediane=False):
    """
    Purpose: 
    --------
    Perform EIGENSTRAT method (standard PCA using SNPs variance to normalize)
    
    In:
    ----------
    data            Genotypes (0/1/2) Lxn matrix:  L rows of observations (SNPs)  x   n columns of variables (Individuals)
    #correl=False    Default: Use covariance matrix
    #correl=True     Use correlation matrix instead of covariance matrix
    extend=False    Default: Only the first 4 eigenvectors/values are returned, eigenvalues are normalized wrt sum of all eigenvalues
    extend=True     location, scores and the whole matrice/vector of eigenvectors/values are returned
   """
    from scipy import linalg as la
    
    # matrice de covariance Eigenstrat **** CODE ****
    xtx=COVariance(data,mafCtrl=mafCtrl,extend=extend,simpl=simpl,
                  diploid_biallelic=diploid_biallelic,mediane=mediane)
    #xtx=(np.dot(datac,datac.transpose()))/(np.shape(data2)[0])    # A VERIFIER : au final nSamples x nSamples comme codé ici ?
    # # matrice de covariance Eigenstrat ***  PAPER ***
    # locInd=np.mean(datac,axis=0)
    # datacc=datac-locInd
    # xcov=(np.dot(datacc,datacc.transpose()))/(np.shape(data2)[0]) # A VERIFIER : au final nSamples x nSamples comme codé ici ?
    
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # EIGENSTRAT CODE:
    evals, evecs = la.eig(xtx)
    # EIGENSTRAT PAPER (results seem very similar to above):
    #evals, evecs = la.eig(xcov)
    
    # sort by eigenvalue in decreasing order
    idx = np.argsort(abs(evals))[::-1]
    evecst = evecs[:,idx]
    evalst= evals[idx]
    # customize results and return
    if extend:
        #scores = np.dot(datac.transpose(),evecst)
        return evalst,evecst,0,0 #scores
    else:
        # select the first 4 eigenvectors and -values, normalize eigenvalues
        xeigenvectors = evecst[:,:4]
        PC1to4unNormalized = evalst[:4]
        PC1to4 = evalst[:4]/sum(evalst)
        return PC1to4,xeigenvectors,0,0

def PCAmcdeigstr(data,extend=False,mafCtrl=False,support_fraction=None,
                 diploid_biallelic=True,mode='SNPs'):
    """
    Purpose: 
    --------
    Prepare and launch Minimum Covariance Determinant procedure to find the covariance matrix,
    based on the eigenstrat-normalized data matrix [G']: g'_ij = (g_ij - mu_i) / sqrt(p(1-p)) ,
    and compute eigenv's on the covariance matrix.
    
    In:
    ----------
    data1           Genotypes (0/1/2) Lxn matrix:  L rows of observations (SNPs)  x   n columns of variables (Individuals)
                    expected :  - less variables (N columns) than realisations (M rows)
                                - matrix rank is N - otherwise incorporation of jitter
                                - diff with testjitter.py "if matRank < M" in the latter, prob. error in testjitter.py
    extend=False    Default: Only the first 4 eigenvectors/values are returned, eigenvalues are normalized wrt sum of all eigenvalues
    extend=True     location, scores and the whole matrice/vector of eigenvectors/values are returned
    mafCtrl=False   Default: take coding 0/1/2 as is.
    mafCtrl=True    Ensure that minor allele is coded 1. Recode 0/1/2 into 2/1/0 otherwise.
    support_fraction parameter passed to <<fast_mcd>>
                    Default:None
                    Otherwise : float, 0 < support_fraction < 1
                                The proportion of points to be included in the support of the raw
                                MCD estimate. Default is None, which implies that the minimum
                                value of support_fraction will be used within the algorithm:
                                `[n_sample + n_features + 1] / 2`.
    mode            subset of 'SNPs' or 'Individuals'
    
    """
    from scipy import linalg as la
    from copy  import deepcopy
    
    location = np.nanmean(data, axis=1) # mean of each row, ignoring NaNs
    
    # frequency per SNP
    p= location / 2.
    fr= (1.+location*np.shape(data)[1]) / (2.+2.*np.shape(data)[1])
    
    # MAF control
    if mafCtrl:
        data2=deepcopy(data)
        p2=deepcopy(p)
        for i in range(np.shape(data)[0]):
            if p[i] > 0.5:
                data2[i,:]=2.-data[i,:]
        # recalc center of data and frequency per SNP
        location2 = np.nanmean(data2, axis=1) # mean of each row
        p2= location2 / 2.
        fr2= (1.+location2*np.shape(data2)[1]) / (2.+2.*np.shape(data2)[1])
    else:
        data2=data ; p2=p ; location2=location ; fr2=fr
    
    # normalized genotypes matrix
    if diploid_biallelic:
        datac = (data2.transpose()-location2)/np.sqrt(fr2*(1-fr2))
    else:
        mu=np.nanmean(data2, axis=1, keepdims=True) # mean of each row, ignoring NaNs
        if mediane:
            nrow=np.shape(data2)[0]
            mu=np.median(data2[np.isfinite(data2)], axis=1).reshape((nrow,1)) #use option keepdims=True for numpy 1.9 or newer
        ncol=np.shape(data2)[1]
        quotient=0.
        for allel in range(5):
            #posterior estimate of unobserved allele frequency
            fa=(1+np.sum(data2==float(allel),axis=1,keepdims=True))/float(1+ncol)
            quotient=quotient+(fa*(1-fa))
        quotient=np.sqrt(quotient)
        # Transformation
        datac=((data2-mu)/quotient).T
    
    datac = datac.transpose()
    
    (n,m)=np.shape(datac)
    if n<m:
        datac=datac.T
        (n,m)=np.shape(datac)
    # Rank and jitter : NOT DOABLE WHEN MISSING VALUES IN DATA
        
    # MCD
    #print "np.shape(datac)= ",np.shape(datac)
    #print "support_fraction= ",support_fraction
    if mode=='individuals':
        datac=datac.T
    
    (location, C, support, dist) = fastmcdnan(datac,support_fraction)    #,support_fraction=0.9)
    
    if mode=='individuals':
        datac=datac.T
        location=location.T
    
    X_centered = datac - location
        
    #print 'covariance matrix excerpt (mcd+eigstr):'
    #print C[:15,:15]

    # Calculate eigenvectors & eigenvalues of the covariance matrix
    evals, evecs = la.eig(C)
    # sort by eigenvalue in decreasing order
    idx = np.argsort(abs(evals))[::-1]
    evecst = evecs[:,idx]
    evalst= evals[idx]
    if extend:
        # calculate data coordinates in eigenvectors base
        #scores = np.dot(X_centered,evecst)
        return evalst,evecst,0,0
    else:
        # select the first 4 eigenvectors and -values, normalize eigenvalues
        xeigenvectors = evecst[:,:4]
        PC1to4unNormalized = evalst[:4]
        PC1to4 = evalst[:4]/sum(evalst)
        return PC1to4,xeigenvectors,0,0

def UCS_covariance(X, assume_centered=False):
    """Computes the UCS 'covariance' estimator

    Parameters
    ----------
    X : 2D ndarray, shape (n_samples, n_features) 
                 --- here (  nSNPs,     nInds   ) because so many SNPs ans so few Inds. ---
        Data from which to compute the 'covariance' estimate

    assume_centered : Boolean
        Not used in WCS computation
        Kept to maintain similar interface

    Returns
    -------
    UCS matrix : 2D ndarray, shape (n_features, n_features) 

    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))
        warnings.warn("Only one sample available. "
                      "You may want to reshape your data array")

    #Initialisations
    L,n=np.shape(X)
    covariance=np.array(n*[n*[float('NaN')]])
    #MAFs
    location = np.mean(X, axis=1) # mean of each row
    p= (1+location*n) / (2+2*n)
    for i in range(L):
        if p[i] > 0.5:
            X[i,:]=2.-X[i,:]
    location2 = np.mean(X, axis=1) # mean of each row
    p2= (1.+location2*n) / (2.+2.*n)

    """
    UCS - Unweighted Corrected Similarity:
    Ref: Oliehoek et al 2006 equation (5), page 486 + S_xy of equation (1), page 485
    
    S_xy= 1/4 * [Iac+Iad+Ibc+Ibd] where a, b are the alleles for genotype of individual x,
                                       c, d are the alleles for genotype of individual y,
                                       Iac is 1 if alleles a and c are equal, 0 otherwise.
    
    S_xy= 1 if both alleles are 0 (common), for individuals x and y,
          0.5 if both alleles are 0 for one individual, and if one allele is 1 (mutant) for the other individual,
          0 if both alleles are 0 for one individual, and both alleles are 1 (mutant) for the other individual,
          0.5 if one allele is 0 and the other 1, for both individuals,
          0.5 if one allele is 0 and the other 1, for one individual, and both alleles are 1, for the other individual,
          1 if both alleles are 1 (mutant), for both individuals.
          
    """
    coeff=np.array([[1*p2/p2,    1/2.*p2/p2, 0.*p2/p2  ],
                    [1/2.*p2/p2, 1/2.*p2/p2, 1/2.*p2/p2],
                    [0.*p2/p2,   1/2.*p2/p2, 1*p2/p2   ]])
    data_0=(X==0) ; data_1=(X==1) ; data_2=(X==2)
    for i in range(n):
        g_i=np.vstack((data_0[:,i],data_1[:,i],data_2[:,i]))
        sum_S_xy_perSample= np.sum(np.dot(coeff[:,0]*g_i,data_0[:,i:]) \
                                  +np.dot(coeff[:,1]*g_i,data_1[:,i:]) \
                                  +np.dot(coeff[:,2]*g_i,data_2[:,i:]),axis=0)
        covariance[i,i:]=covariance[i:,i]=4./L*sum_S_xy_perSample-2.   

    return covariance


def WCS_covariance(X, assume_centered=False,biallelic=True,dist=False, 
    gamma=None,Lmediane=False,dblCenter=False):
    """
    Computes the WCS 'covariance' estimator 	(dist=False)
          or the WCS distance matrix 		(dist=True)
    Tolerant to missing values

    Parameters
    ----------
    X : 2D ndarray, shape (n_samples, n_features) 
                 --- here (  nSNPs,     nInds   ) because so many SNPs ans so few Inds. ---
        Data from which to compute the 'covariance' estimate

    assume_centered : Boolean
        Not used in WCS computation
        Kept to maintain similar interface
        
    gamma : if not None, gamma is used as threshold for Huber l2 -> l1 function

    Returns
    -------
    WCS matrix : 2D ndarray, shape (n_features, n_features) 

    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))
        warnings.warn("Only one sample available. "
                      "You may want to reshape your data array")

    #Initialisations
    L,n=np.shape(X)
    covariance=np.array(n*[n*[float('NaN')]])
    #MAFs
    location = np.nanmean(X, axis=1) # mean of each row, ignoring NaNs
    p= (1+location*n) / (2+2*n)
    for i in range(L):
        if p[i] > 0.5:
            X[i,:]=2.-X[i,:]
    location2 = np.nanmean(X, axis=1) # mean of each row
    p2= (1.+location2*n) / (2.+2.*n)

    """
    Ref: Oliehoek et al 2006 equations (6) anf (7), page 486 + S_xy of equation (1), page 485
    
    Equation (6): r_xy= 2/W *sum_on_l(w_l*(S_xy,l - sl) / (1-sl))
    where r_xy is the relatedness between individuals x and y, W is the sum of w_l's over all loci,
          w_l is the weight for one locus, determined after equation (7),
          S_xy is the similarity for individuals x and y at locus l,
          and sl=1/nl = 1/(nb of alleles)
    
    Equation (7): 1/wl= sum_on_alleles_i(p_i^2)*(1-sum_on_alleles_i(p_i^2)) / (1-1/nl)^2
    where p_i is the allele i frequency. p_1 is p2, p_0 is 1-p2
    
    In biallelic case, equations (6) and (7) simplify to:  r_xy= 2/W * sum_on_l(w_l*S_xy,l/(1-1/nl)) - 2./(nl-1)
                                        w_l=  (1-1/nl)^2 / (sum_on_alleles_i(p_i^2)*(1-sum_on_alleles_i(p_i^2)))
    """
        
    if biallelic:
        w_l=1/(4*(p2*p2+(1-p2)*(1-p2))*(1-(p2*p2+(1-p2)*(1-p2))))
        """
        Missing values: Sum is on l from 1 to L wherever geno is not missing
        => coeff_and_wl, data_0 etc and g_i remain unchanged.
        => sum_w_l_S_xy_perSample is correct since g_i elements are false when missing,
           meaning that nothing is added in this case.
        => W has to be particularized to pairs of individuals, since wl should 
           be 0 when geno is missing for one of the individuals compared.
        """
        #W sum of weights per individual (previously: W=sum(w_l))
        nonmissingwl= np.array(n*[w_l]).T * (1.-np.isnan(X))
        W=np.dot((1.-np.isnan(X)).T,nonmissingwl)
        #combined similarity coefficient and wl weights per locus
        coeff_and_wl=np.array([[w_l,    w_l/2., 0.*w_l],
                               [w_l/2., w_l/2., w_l/2.],
                               [0.*w_l, w_l/2., w_l   ]])
        data_0=(X==0) ; data_1=(X==1) ; data_2=(X==2)
        for i in range(n):
            # gi: logicals data_0, data_1 and data_2 restricted to individual i
            g_i=np.vstack((data_0[:,i],data_1[:,i],data_2[:,i])) #here column i of data_0 becomes row i of g_i
            # similarity for individual i crossed with itsel and individuals i+1 to n
            sum_w_l_S_xy_perSample= np.sum(np.dot(coeff_and_wl[:,0]*g_i,data_0[:,i:]) \
                                          +np.dot(coeff_and_wl[:,1]*g_i,data_1[:,i:]) \
                                          +np.dot(coeff_and_wl[:,2]*g_i,data_2[:,i:]),axis=0)
            # covariance with weight W depending on individual i and j in [i,n]
            covariance[i,i:]=covariance[i:,i]= 4./W[i,i:] * sum_w_l_S_xy_perSample - 2.
        # Scale to [0;1] interval
        covariance=covariance/2.
        if dist:
            covariance=1.-covariance
    else:
        data=X
        # Frequences
        # Missing values: use nanmx, count of non nans elements per row
        sl=1./(1.+np.nanmax(data,axis=1,keepdims=True))
        nnonans=np.sum(1.-np.isnan(data),axis=1,keepdims=True)
        f0=np.sum(data==0,axis=1,keepdims=True)/nnonans
        f1=np.sum(data==1,axis=1,keepdims=True)/nnonans
        f2=np.sum(data==2,axis=1,keepdims=True)/nnonans
        f3=np.sum(data==3,axis=1,keepdims=True)/nnonans
        f4=np.sum(data==4,axis=1,keepdims=True)/nnonans
        sum_on_alleles_i=f0**2+f1**2+f2**2+f3**2+f4**2
        # Weights
        w_l=(1.-sl)**2 / (sum_on_alleles_i*(1-sum_on_alleles_i))
        #W sum of weights per individual (previously: W=sum(w_l))
        nonmissingwl= np.array(n*[w_l.flatten()]).T * (1.-np.isnan(data))
        W=np.dot((1.-np.isnan(X)).T,nonmissingwl)
        idx=np.logical_not(np.isnan(data))
        # Elements of WCS matrix
        for i in range(n):
            for j in range(i,n):
                #normalization *0.5 (unchanged in spite of nl being !=2) so that 1-C has values between 0 and 1
                ijdx=idx[:,i]*idx[:,j] #common non-nan elements for individual pair (i,j)
                Lij=sum(ijdx)
                S_xyl=(data[ijdx,i]==data[ijdx,j]).reshape((Lij,1)) #Similarity element-wise
                # WCS covariance, missing values => excluded via ijdx AND W paricularized to pair (i,j)
                covariance[i,j]=covariance[j,i]= 0.5*(2./W[i,j] *np.sum((w_l[ijdx]*(S_xyl-sl[ijdx])/(1.-sl[ijdx])) ,axis=0) )
        if dist:
            covariance=1.-covariance
    
    center=np.median(covariance)
    if gamma is not None:
        if gamma=='mad':
            G=3.*mad(covariance)
        else:
            G=gamma
        print "gamma cutoff l2-l1 is : ",G
        for i in range(n):
            for j in range(i,n):
                x= covariance[i,j]-center
                if abs(x) > G:
                    covariance[i,j]= covariance[j,i]= center+np.sign(x)*np.sqrt(G*(2.*abs(x)-G))
    
    if dblCenter:
        if Lmediane:
            median1=np.median(covariance,axis=1)
            covariance= covariance-median1[:, np.newaxis]
            median0=np.median(covariance,axis=0)
            covariance= covariance-median0[np.newaxis,:]
        else:
            mean1=np.mean(covariance,axis=1)
            covariance= covariance-mean1[:, np.newaxis]
            mean0=np.mean(covariance,axis=0)
            covariance= covariance-mean0[np.newaxis,:]
    
    return covariance

def WEDS_covariance(X, assume_centered=False):
    """Computes the WEDS 'covariance' estimator

    Parameters
    ----------
    X : 2D ndarray, shape (n_samples, n_features) 
                 --- here (  nSNPs,     nInds   ) because so many SNPs ans so few Inds. ---
        Data from which to compute the 'covariance' estimate

    assume_centered : Boolean
        Not used in WCS computation
        Kept to maintain similar interface

    Returns
    -------
    WEDS matrix : 2D ndarray, shape (n_features, n_features) 

    """

    X = np.asarray(X)
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))
        warnings.warn("Only one sample available. "
                      "You may want to reshape your data array")

    #Initialisations
    L,n=np.shape(X)
    covariance=np.array(n*[n*[float('NaN')]])
    #MAFs
    location = np.mean(X, axis=1) # mean of each row
    p= (1+location*n) / (2+2*n)
    for i in range(L):
        if p[i] > 0.5:
            X[i,:]=2.-X[i,:]
    location2 = np.mean(X, axis=1) # mean of each row
    p2= (1.+location2*n) / (2.+2.*n)

    """
    WEDS - Weighted Equal Drift Similarity:
    Ref: Oliehoek et al 2006 equation (8), page 486 + S_xy of equation (1), page 485
    
    Equation (8): sl= (sum_on_alleles_i(p^2+(1-p)^2) - Smin) / (1-Smin)
    where Smin is the minimum expected similarity min(sum_on_alleles_i(p^2+(1-p)^2))
    
    Equations (6) and (7) are used to compute relatedness rxy and weights wl,
    replacing sl=1/nl=1/2 by the sl defined in equation (8).
    
    Finally, r_xy can be expressed as follows:
    
    rxy= 2./W * [sum_on_l(w_l/(1-sl) *S_xy,l) - sum_on_l(w_l*sl/(1-sl))]
    
    where wl=   (1-sl)^2/((p^2+(1.-p)^2)*(1-(p^2+(1.-p)^2)))
          W=    sum_on_l(w_l)
          sl=   (p^2+(1-p)^2) - Smin) / (1-Smin)
          Smin= min_over_loci(p^2+(1-p)^2)
          p=    minor allele frequency
    
    """
    pre_sl=p2*p2+(1-p2)*(1-p2)
    Smin=min(pre_sl)
    sl=(pre_sl-Smin)/(1-Smin)
    w_l=(1-sl)*(1-sl)/((p2*p2+(1-p2)*(1-p2))*(1-(p2*p2+(1-p2)*(1-p2))))
    W=sum(w_l)
    coeff_w_l_sl=np.array([[w_l/(1-sl),      w_l/(2.*(1-sl)), 0.*w_l         ],
                           [w_l/(2.*(1-sl)), w_l/(2.*(1-sl)), w_l/(2.*(1-sl))],
                           [0.*w_l,          w_l/(2.*(1-sl)), w_l/(1-sl)     ]])
    data_0=(X==0) ; data_1=(X==1) ; data_2=(X==2)
    for i in range(n):
        g_i=np.vstack((data_0[:,i],data_1[:,i],data_2[:,i]))
        sum_w_l_sl_S_xy_perSample= np.sum(np.dot(coeff_w_l_sl[:,0]*g_i,data_0[:,i:]) \
                                      +np.dot(coeff_w_l_sl[:,1]*g_i,data_1[:,i:]) \
                                      +np.dot(coeff_w_l_sl[:,2]*g_i,data_2[:,i:]),axis=0)
        covariance[i,i:]=covariance[i:,i]= 2./W*(sum_w_l_sl_S_xy_perSample - np.sum(w_l*sl/(1-sl)))

    return covariance


def PCAmcdREL(data1,extend=False,mafCtrl=False,support_fraction=None,
    cov_method=WCS_covariance,gamma=None,Lmediane=False,dblCenter=False):
    """
    Purpose: 
    --------
    Prepare and launch Minimum Covariance Determinant procedure using a relatedness matrix instead
    of a covariance matrix, compute eigenv's on this relatedness matrix.
    Relatedness can be UCS, WCS or WEDS
    
    In:
    ----------
    data1           Genotypes (0/1/2) Lxn matrix:  L rows of observations (SNPs)  x   n columns of variables (Individuals)
                    expected :  - less variables (N columns) than realisations (M rows)
                                - matrix rank is N - otherwise incorporation of jitter
                                - diff with testjitter.py "if matRank < M" in the latter, prob. error in testjitter.py
    extend=False    Default: Only the first 4 eigenvectors/values are returned, eigenvalues are normalized wrt sum of all eigenvalues
    extend=True     location, scores and the whole matrice/vector of eigenvectors/values are returned
    mafCtrl=False   Default: take coding 0/1/2 as is.
    mafCtrl=True    Ensure that minor allele is coded 1. Recode 0/1/2 into 2/1/0 otherwise.
    support_fraction parameter passed to <<fast_mcd>>
                    Default:None
                    Otherwise : float, 0 < support_fraction < 1
                                The proportion of points to be included in the support of the raw
                                MCD estimate. Default is None, which implies that the minimum
                                value of support_fraction will be used within the algorithm:
                                `[n_sample + n_features + 1] / 2`.
    cov_method=UCS_covariance  Default: covariance matrix replaced by relatedness, UCS matrix
    cov_method=WCS_covariance  Covariance matrix replaced by relatedness, WCS matrix
                               WCS_covariance: ONLY METHOD TOLERANT TO MISSINGNESS IN DATA - others would have
                               to be adapted
    cov_method=WEDS_covariance covariance matrix replaced by relatedness, WEDS matrix
    gamma           Cutoff l2 -> l1 for WCS_covariance, thus defining Huber WCS covariance (or distance) matrix
    Lmediane        Center using median when True, for WCS_covariance method
    dblCenter       Centering by rows and columns when True, in WCS_covariance
    """
    from scipy import linalg as la
    from copy  import deepcopy

    if mafCtrl:
        p= np.nanmean(data1, axis=1) / 2.   # frequency per snp
        data=deepcopy(data1)
        # recode 2/1/0 if frequency is higher than 0.5
        for i in range(np.shape(data1)[0]):
            if p[i] > 0.5: data[i,:]=2.-data1[i,:]
    else:
        data=deepcopy(data1)

    """ Check matrix rank : not doable with missingness ; or try and compute wcs mat, check det of it.
    M,N=np.shape(data)
    #if M<N: data=data.T ; M2=N ; N=M ; M=M2   # Transpose to smallest dim
    dataori=deepcopy(data)
    matRank= np.linalg.matrix_rank(np.float32(data))
    #print 'matRank = ',matRank
    # Jitter always to avoid pb on subsets #older:# Jitter to obtain rank=N ; further Idea : SVD-reduce dimensionality when rank is less than the size of the cov matrix
    isJitter='TRUE'
    if isJitter: # if matRank < N:
        #print "jitter introduced"
        # setting
        mag=0.1
        coefMat = np.random.normal(0, mag, np.shape(data))       
        data=data*(1+coefMat)
        matRank2= np.linalg.matrix_rank(np.float32(data))
        #print 'matRank2 = ',matRank2
    """
    
    if cov_method == WCS_covariance:
        argsWCS= gamma,Lmediane,dblCenter
    else:
        argsWCS= None
    
    # MCD
    (location, C, support, dist) = fastmcdnan(data,support_fraction,cov_computation_method=cov_method,argsWCS=argsWCS)    
    #,support_fraction=0.9)
    #print support[:10]
    """
    # Return to original data, if jitter has been applied
    if isJitter:
        # if matRank < N:
        print dataori[:10,:10]
        datanj= dataori[support]
        location= np.mean(datanj,axis=0)
        C= cov_method(datanj) # use support given by MCD to compute C on core data
        precision = la.pinvh(C)         ; X_centered = data - location
        dist = (np.dot(X_centered, precision) * X_centered).sum(1)
    """
    X_centered = data - location
        
    #print 'covariance matrix excerpt:'
    #print C[:15,:15]

    # calculate eigenvectors & eigenvalues of the covariance matrix
    evals, evecs = la.eig(C)
    # sort by eigenvalue in decreasing order
    idx = np.argsort(abs(evals))[::-1]
    evecst = evecs[:,idx]
    evalst= evals[idx]
    if extend:
        # calculate data coordinates in eigenvectors base
        #scores = np.dot(X_centered,evecst)
        return evalst,evecst,location,0 #scores
    else:
        # select the first 4 eigenvectors and -values, normalize eigenvalues
        xeigenvectors = evecst[:,:4]
        PC1to4unNormalized = evalst[:4]
        PC1to4 = evalst[:4]/sum(evalst)
        return PC1to4,xeigenvectors,0,0

def PCAmcd(data1,extend=False,mafCtrl=False,support_fraction=None):
    """
    Purpose: 
    --------
    Prepare and launch Minimum Covariance Determinant procedure to find the covariance matrix,
    compute eigenv's on this covariance matrix
    
    In:
    ----------
    data1           Genotypes (0/1/2) Lxn matrix:  L rows of observations (SNPs)  x   n columns of variables (Individuals)
                    expected :  - less variables (N columns) than realisations (M rows)
                                - matrix rank is N - otherwise incorporation of jitter
                                - diff with testjitter.py "if matRank < M" in the latter, prob. error in testjitter.py
    extend=False    Default: Only the first 4 eigenvectors/values are returned, eigenvalues are normalized wrt sum of all eigenvalues
    extend=True     location, scores and the whole matrice/vector of eigenvectors/values are returned
    mafCtrl=False   Default: take coding 0/1/2 as is.
    mafCtrl=True    Ensure that minor allele is coded 1. Recode 0/1/2 into 2/1/0 otherwise.
    support_fraction parameter passed to <<fast_mcd>>
                    Default:None
                    Otherwise : float, 0 < support_fraction < 1
                                The proportion of points to be included in the support of the raw
                                MCD estimate. Default is None, which implies that the minimum
                                value of support_fraction will be used within the algorithm:
                                `[n_sample + n_features + 1] / 2`.
    
    """
    from scipy import linalg as la
    from copy  import deepcopy
    if mafCtrl:
        p= np.mean(data1, axis=1) / 2.   # frequency per snp
        data=deepcopy(data1)
        # recode 2/1/0 if frequency is higher than 0.5
        for i in range(np.shape(data1)[0]):
            if p[i] > 0.5: data[i,:]=2.-data1[i,:]
    else:
        data=data1

    # Check matrix rank
    M,N=np.shape(data)
    #if M<N: data=data.T ; M2=N ; N=M ; M=M2   # Transpose to smallest dim
    dataori=deepcopy(data)
    matRank= np.linalg.matrix_rank(data.astype('float32'))
    #print 'matRank = ',matRank
    # Jitter always to avoid pb on subsets #older:# Jitter to obtain rank=N ; further Idea : SVD-reduce dimensionality when rank is less than the size of the cov matrix
    isJitter='TRUE'
    if isJitter: # if matRank < N:
        #print "jitter introduced"
        # setting
        mag=0.1
        coefMat = np.random.normal(0, mag, np.shape(data))       
        data=data*(1+coefMat)
        matRank2= np.linalg.matrix_rank(data.astype('float32'))
        #print 'matRank2 = ',matRank2
    # MCD
    (location, C, support, dist) = fast_mcd(data,support_fraction)    #,support_fraction=0.9)
    # Return to original data, if jitter has been applied
    if isJitter: # if matRank < N:
        data= dataori
        location= data[support].mean(0) ; C= np.cov(data[support].T) # use support given by MCD to compute C on core data
        precision = la.pinvh(C)         ; X_centered = data - location
        dist = (np.dot(X_centered, precision) * X_centered).sum(1)
    else:
        X_centered = data - location
        
    #print 'covariance matrix excerpt:'
    #print C[:15,:15]

    # calculate eigenvectors & eigenvalues of the covariance matrix
    evals, evecs = la.eig(C)
    # sort by eigenvalue in decreasing order
    idx = np.argsort(abs(evals))[::-1]
    evecst = evecs[:,idx]
    evalst= evals[idx]
    if extend:
        # calculate data coordinates in eigenvectors base
        #scores = np.dot(X_centered,evecst)
        return evalst,evecst,location,0 #scores
    else:
        # select the first 4 eigenvectors and -values, normalize eigenvalues
        xeigenvectors = evecst[:,:4]
        PC1to4unNormalized = evalst[:4]
        PC1to4 = evalst[:4]/sum(evalst)
        return PC1to4,xeigenvectors,0,0

def PCAsph(data1,mode,extend=False,mafCtrl=False):
    from scipy import linalg as la
    from copy import deepcopy
    # MAF control : minor allele should be coded 1, not 0., so that 0/1/2 means homozygote common / heterozygote / homozygote rare
    if mafCtrl:
        p= np.mean(data1, axis=1) / 2.   # frequency per snp
        data=deepcopy(data1)
        # recode 2/1/0 if frequency is higher than 0.5
        for i in range(np.shape(data1)[0]):
            if p[i] > 0.5: data[i,:]=2.-data1[i,:]
    else:
        data=data1

    # location and covariance matrix
    if mode=="rows":
        location = np.mean(data, axis=0)          # column-wise mean
        datac = data-location
        sq = (np.sum(datac*datac,axis=1))**0.5  # root of vertical sum of squared elements
        datacn = datac/sq[:,None]               # row-wise division
        C = np.cov(datacn.T)
    elif mode=="columns":
        location = np.mean(data, axis=1)          # row-wise mean
        datac = data-location[:,None]             # [:,None] reshapes as column vector
        sq = (np.sum(datac*datac,axis=0))**0.5  # root of vertical sum of squared elements
        datacn = datac/sq                       # column-wise division
        C = np.cov(datacn)
    #print 'covariance matrix extract:'
    #print C[:15,:15]
    # eigenvectors & eigenvalues of covariance matrix
    evals, evecs = la.eig(C)
    # sort by eigenvalue in decreasing order
    idx = np.argsort(abs(evals))[::-1]
    evecst = evecs[:,idx]
    evalst= evals[idx]
    # customize results and return
    if extend:
        # calculate data coordinates in eigenvectors base
        #scores = np.dot(datacn,evecst)
        return evalst,evecst,location,0 #scores
    else:
        # select the first 4 eigenvectors and -values, normalize eigenvalues
        xeigenvectors = evecst[:,:4]
        PC1to4unNormalized = evalst[:4]
        PC1to4 = evalst[:4]/sum(evalst)
        return PC1to4,xeigenvectors,0,0

def COVariance(data,mafCtrl=False,extend=False,simpl=False,
                  diploid_biallelic=True,mediane=False,
                  dist=True,gamma=None):
    """
    Purpose: 
    --------
    Calculate covariance matrix as in EIGENSTRAT, tolerate nans
    
    In:
    ----------
    data            Genotypes (0/1/2) Lxn matrix:  L rows of observations (SNPs)  x   n columns of variables (Individuals)
    extend=False    Default: Only the first 4 eigenvectors/values are returned, eigenvalues are normalized wrt sum of all eigenvalues
    extend=True     location, scores and the whole matrice/vector of eigenvectors/values are returned
    gamma : if not None, gamma is used as threshold for Huber l2 -> l1 function
    """
    import copy
    # center of data
    location = np.nanmean(data, axis=1) # mean of each row, ignoring NaNs
    
    # frequency per SNP
    p= location / 2.
    fr= (1.+location*np.shape(data)[1]) / (2.+2.*np.shape(data)[1])
    if simpl: fr= location / 2.
    
    # MAF control
    if mafCtrl:
        data2=copy.deepcopy(data)
        p2=copy.deepcopy(p)
        for i in range(np.shape(data)[0]):
            if p[i] > 0.5:
                data2[i,:]=2.-data[i,:]
        # recalc center of data and frequency per SNP
        location2 = np.nanmean(data2, axis=1) # mean of each row, ignoring NaNs
        p2= location2 / 2.
        fr2= (1.+location2*np.shape(data2)[1]) / (2.+2.*np.shape(data2)[1])
        if simpl: fr2= location2 / 2.
    else:
        data2=data ; p2=p ; location2=location ; fr2=fr
    
    # normalized genotypes matrix
    if diploid_biallelic:
        datac = (data2.transpose()-location2)/np.sqrt(fr2*(1-fr2))
    else:
        mu=np.nanmean(data2, axis=1, keepdims=True) # mean of each row, ignoring NaNs
        if mediane:
            nrow=np.shape(data2)[0]
            mu=np.median(data2[np.isfinite(data2)], axis=1).reshape((nrow,1)) #use option keepdims=True for numpy 1.9 or newer
        ncol=np.shape(data2)[1]
        quotient=0.
        for allel in range(5):
            #posterior estimate of unobserved allele frequency
            fa=(1+np.sum(data2==float(allel),axis=1,keepdims=True))/float(1+ncol)
            quotient=quotient+(fa*(1-fa))
        quotient=np.sqrt(quotient)
        # Transformation
        datac=((data2-mu)/quotient).T

    (n,m)=np.shape(datac.T)
    if n<m:
        datac=datac.T
        (n,m)=np.shape(datac.T)
    
    # matrice de covariance Eigenstrat **** CODE ****
    #xtx=(np.dot(datac,datac.transpose()))/(np.shape(data2)[0])    # A VERIFIER : au final nSamples x nSamples comme codé ici ?
    # xtx with missing values taken into account in the following:

    # Samples' pairwise counts of non-missing genotypes
    idx=1.*np.logical_not(np.isnan(datac)) #index of non-nan elements
    ijdxM=np.dot(idx,idx.T) #counts for non-nan elements for each individual pair (i,j)

    # Replace nans with zeros for the np.dot to work out correctly
    matIndex=np.where(np.isnan(datac)) 
    datac[matIndex]=0.

    # Covariance matrix ; ijdxM allow to recalibrate due to missing values
    xtx=(np.dot(datac,datac.T)) /ijdxM
    
    covariance=xtx
    
    if dist:
        covariance= np.max(covariance) - covariance

    center=np.median(covariance)
    if gamma is not None:
        if gamma=='mad':
            G=3.*mad(covariance)
            print "gamma cutoff is: ",G
        else:
            G=gamma
        for i in range(m):
            for j in range(i,m):
                x= covariance[i,j]-center
                if abs(x) > G:
                    covariance[i,j]= covariance[j,i]= center+np.sign(x)*np.sqrt(G*(2.*abs(x)-G))
    
    
    return covariance
    
##################
#   - acpCounts: . call anaCounts to calculate wcCounts, Manhattan or Bray distances'matrix
#                . calculate acp or MDS and return
def acpCounts(data,
          acpOrMds='acp',
          metric='wcCounts',
          extend=False,
          dist=False,
          dblCenter=False,
          Lmediane=True,
          Lmad=True,
          wcCo_Lsigned=False):
    """
    Purpose: 
    --------
    Perform PCA or MdS based on similarity or distance matrices
    
    In:
    ----------
    data            Genotypes (0/1/2) Lxn matrix:  L rows of observations (SNPs)  x   n columns of variables (Individuals)
    acpOrMds        Can be 'acp' or 'mds'
    metric          Can be . wcCounts: weighted corrected counts similarity or distance
                           . Manhattan
                           . Bray-Curtis
    extend          If TRUE, more output is returned (full ma, location, etc.)
    dblCenter       If TRUE the simil. or dist matrix is centered along columns and rows
    Lmediane        If TRUE, median is used instead of mean
    Lmad            If TRUE, median absoulte deviation is used instead of stand deviation 
    wcCo_Lsigned    If TRUE, used signed pairwise differences, otherwise use absolute difference
    """
    from scipy import linalg as la
    
    # similarity matrix, or distance matrix
    
    C=anaCounts(data,
          method=metric,
          extend=extend,
          dist=dist,
          dblCenter=dblCenter,
          Lmediane=Lmediane,
          Lmad=Lmad,
          wcCo_Lsigned=wcCo_Lsigned)
    
    # acp
    if acpOrMds=='acp':
        # calculate eigenvectors & eigenvalues of the similarity matrix
        evals, evecs = la.eig(C)
        # sort by eigenvalue in decreasing order
        idx = np.argsort(abs(evals))[::-1]
        evecst = evecs[:,idx]
        evalst= evals[idx]    
        # New coordinates 
        idxPos,=np.where(evalst>0)
        Xf=np.dot(evecst[:,idxPos],np.diag(evalst[idxPos]**0.5))
      
    # mds (classical)
    if acpOrMds=='mds':
        if (Lmediane | Lmad):
            evalst, evecst, Xf=cMDS(C, alreadyCentered=True)
        else:
            evalst, evecst, Xf=cMDS(C)
    
    # customize results and return
    if extend:
        # return all eigenvectors and -values
        scores = 0.
        return evalst,evecst,Xf,scores
    else:
        # select the first 4 eigenvectors and -values, normalize eigenvalues
        xeigenvectors = evecst[:,:4]
        PC1to4unNormalized = evalst[:4]
        PC1to4 = evalst[:4]/sum(evalst)
        return PC1to4,xeigenvectors,Xf,0

##################
#   - anaCounts: . calculate wcCounts, Manhattan or Bray distances'matrix
#                . get matrix as similarity if wished, double center wherever necessary,
#                . find eigenvectors, eigenvalues and return
def anaCounts(data,
          method='wcCounts',
          extend=False,
          dist=False,
          dblCenter=False,
          Lmediane=True,
          Lmad=True,
          wcCo_Lsigned=False):
    """
    Purpose: 
    --------
    Perform PCA or MdS based on similarity or distance matrices
    
    In:
    ----------
    data            Counts Lxn matrix:  L rows of observations (phylotypes)  x   n columns of variables (Individuals)
    method          Can be . wcCounts: weighted corrected counts similarity or distance
                           . Manhattan
                           . Bray-Curtis
    extend          If TRUE, more output is returned (full ma, location, etc.)
    dist            If TRUE, a distance matrix is returned, otherwise a similarity matrix
    dblCenter       If TRUE the simil. or dist matrix is centered along columns and rows
    Lmediane        If TRUE, median is used instead of mean
    Lmad            If TRUE, median absoulte deviation is used instead of stand deviation 
    wcCo_Lsigned    If TRUE, used signed pairwise differences, otherwise use absolute difference
    """
    import numpy as np
    from scipy import linalg as la
    import copy
    import sys
    
    if method=='wcCounts':
        C=wcCounts_similDist(data,
          dist=dist,
          Lmediane=Lmediane,
          Lmad=Lmad,
          Lsigned=wcCo_Lsigned,
          form='square')
        
    elif method=='Manhattan':
        # data (P,n) must be transposed to (n,P), 
        # in order to obtain distances between the n individuals
        C=squareform(pdist(data.T,metric='cityblock'))
        if not dist:
            C= 1- C/np.max(C)
        
    elif method=='Bray-Curtis':
        # data (P,n) must also be transposed to (n,P)
        C=squareform(pdist(data.T,metric='braycurtis'))
        if not dist:
            C= 1- C/np.max(C)
        
    else:
        sys.exit("Distance not recognized, it should be : wcCounts, Manhattan or Bray-Curtis")
        
    # Double centering, if dblCenter is True
    if dblCenter:
        if Lmediane:
            median1=np.median(C,axis=1)
            C= C-median1[:, np.newaxis]
            median0=np.median(C,axis=0)
            C= C-median0[np.newaxis,:]
        else:
            mean1=np.mean(C,axis=1)
            C= C-mean1[:, np.newaxis]
            mean0=np.mean(C,axis=0)
            C= C-mean0[np.newaxis,:]
    
    return C
 

#   - wcCounts_similDist: calculate wcCounts distance or similarity matrix
def mad(data, axis=None, K=1.4826):
    return K*np.median(np.abs(data - np.median(data, axis)), axis)

def wcCounts_similDist(X,
          assume_centered=False,
          dist=False,
          Lmediane=False,
          Lmad=False,
          form='square',
          simpl=False,
          sdTrunc=None,
          weightCorrD=False
          ):
    """
    Computes the weighted corrected counts' similarity	(dist=False)
           		or distance (dist=True) matrix.
    Tolerant to missing values

    Parameters
    ----------
    X :               2D ndarray, shape (n_samples, n_features)
                      Data from which to compute similarity or distance matrix
    assume_centered : Boolean
                      Not used in WCS computation
                      Kept to maintain similar interface
    dist :            distance matrix if true, similairyt matrix otherwise
    Lmediane :        use mediane if true, mean otherwise
    Lmad :            use median absolute deviation if true, standard deviation otherwise
    Lsigned :         use signed pairwise differences if true, absolute difference if false
    form :            square matrix ('square') or distance matrix ('dist')
    Returns
    -------
    WCS matrix : 2D ndarray, shape (n_features, n_features) 

    """
        
    X = np.asarray(X)
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))
        warnings.warn("Only one sample available. "
                      "You may want to reshape your data array")

    # Initialisations
    # P: nb of phylotypes ; n: nb of samples
    P,n=np.shape(X)
    similDist=np.array(n*[n*[float('NaN')]])

    """
    Weighted Corrected Agreement between Counts:
    a_xy= 1/W *sum_on_p(w_p*(S_xy,p - sp) / (max(abs)-sp))
    where x, y are samples for which we calculate agreement a, 
            based on similarity per phylogeny S_xy,p, 
          p is a phylogeny 
          W=sum_on_p(w_p)
          w_p is the weight per phylogeny, w_p=(1-sp)^2/Scale(pairwise S_xy,p)
          s_p is the expected pairwise similarity among all S_xy,p, 
               for all (x,y) pairs, at phylogeny p.
               Here, s_p=median(S_xy,p) among (x,y) pairs.
          NB: luckily, for hmp data (restricted to v.introitus), S_xy,p ~ N(s_p,sigma)
          S_xy,p is the pairwise absolute difference between individual x and y,
                        or signed pairwise difference if Lsigned is True.
    
    Inspired from: Oliehoek et al 2006 equations (6) anf (7), page 486 + S_xy of equation (1), page 485
        Equation (6): r_xy= 2/W *sum_on_l(w_l*(S_xy,l - sl) / (1-sl))
        where r_xy is the relatedness between individuals x and y, W is the sum of w_l's over all loci,
              w_l is the weight for one locus, determined after equation (7),
              S_xy is the similarity for individuals x and y at locus l,
              and sl=1/nl = 1/(nb of alleles)
        Equation (7): 1/wl= sum_on_alleles_i(p_i^2)*(1-sum_on_alleles_i(p_i^2)) / (1-1/nl)^2
        where p_i is the allele i frequency. p_1 is p2, p_0 is 1-p2
        
        In biallelic case, equations (6) and (7) simplify to:  r_xy= 2/W * sum_on_l(w_l*S_xy,l/(1-1/nl)) - 2./(nl-1)
                                            w_l=  (1-1/nl)^2 / (sum_on_alleles_i(p_i^2)*(1-sum_on_alleles_i(p_i^2)))  
    """
    # Initialisations
    data=X  #(P,n) P:nb of phylogenies, n: nb of samples
    W=0.
    a_xy=np.array(0.)
    
    if not (simpl or weightCorrD):
        distDesSxy=True
    else:
        distDesSxy=False
    
    # Relatedness between samples for each phylogeny
    if simpl : 
        for p in range(P):
            # pairwise differences for phylogeny p
            S_xyp=[]
            for i in range((n-1)):
                S_xyp.extend(abs(X[p,i]-X[p,(i+1):]))
            S_xyp=np.array(S_xyp)
            # weight
            w_p=   1 / np.var(S_xyp)       
            # Weighted agreement for current phylogeny, cumulated.
            a_xy= a_xy + w_p*S_xyp
            # Weight, cumulated
            W=W+w_p
        # Pairwise agreement matrix
        similDist= 1/W * a_xy
            
    elif distDesSxy:
        for p in range(P):
            # pairwise differences for phylogeny p
            S_xyp=[]
            for i in range((n-1)):
                # absolute distance for phylogeny p
                S_xyp.extend(abs(X[p,i]-X[p,(i+1):]))
            S_xyp=np.array(S_xyp)
            # center, scale and weighted corrections (weights) for current phylogeny
            if Lmediane:      s_p=   np.median(S_xyp)
            else:             s_p=   np.mean(S_xyp)
            if Lmad:          w_p=   1. / mad(S_xyp)**2
            else:             w_p=   1. / np.std(S_xyp)**2
            # Weighted agreement for current phylogeny, cumulated.
            a_xy= a_xy + w_p*(S_xyp-s_p)
            # Weight, cumulated
            W=W+w_p
        # Pairwise agreement matrix
        similDist= 1/W * a_xy
    
    # transform in similarity matrix or square matrix according to selected parameters
    if not dist:       similDist= 1-similDist/np.max(similDist) # assuming centering and scaling are done elsewhere
    #if dist:       similDist2= -similDist2 # assuming centering and scaling are done elsewhere
    if form=='square': similDist= squareform(similDist)
    #if form=='square': similDist2= squareform(similDist2)
    
    return similDist

def ManhNanDist(X):
    """
    Returns Manhattan Distance between data rows, ignoring Nans, 
    taking mean and rescaling by p,
    to make distances with and without nan comparable.

    """
    (n,p) = np.shape(X)
    ManhDist=np.array(n*(n-1)/2*[float('NaN')])
    index=0
    for i in range(n-1):    # row i,i+1 ... n-1
        dijs=np.nanmean(abs(X[i,:]-X[(i+1):,:]),axis=1)
        ManhDist[index:(index+n-i-1)]=dijs
        index=index+n-i-1
    
    return p*ManhDist

def trimmed_std(data, percentile):
    data = np.array(data)
    data.sort()
    percentile = percentile / 2.
    low = int(percentile * len(data))
    high = int((1. - percentile) * len(data))
    return data[low:high].std(ddof=0)

def trimmed_count_std(data, percentile):
    data = np.array(data)
    data.sort()
    high = sum(data==0.) + int((1. - percentile) * sum(data!=0.))
    return data[:high].std(ddof=0)


