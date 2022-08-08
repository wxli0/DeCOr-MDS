#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt
import os
import random as alea
import sys
from scipy.linalg import solve,pinv,pinv2
from scipy.spatial.distance import pdist, squareform
#

""" nSimplex Volume
    Cayley-Menger formula     
"""

def nSimplexVolume(indices,squareDistMat,denominateurexact=True):
    n = np.size(indices) - 1
    restrictedD = squareDistMat[:,indices][indices,:]
    CMmat = np.vstack(((n+1)*[1.] , restrictedD))
    CMmat = np.hstack((np.array([[0.]+(n+1)*[1.]]).T , CMmat))
    # NB : missing (-1)**(n+1)  ; but unnecessary here since abs() is taken afterwards
    if denominateurexact:
        denominateur = float(2**n * (math.factorial(n))**2)
    else:
        # if calculation of n*Vn/Vn-1 then n*sqrt(denominateurn/denominateurn-1)
        # simplify to 1/sqrt(2), to take into account in CALLING function.
        denominateur = 1.
    VnSquare=np.linalg.det(CMmat**2) # or numpy.linalg.slogdet
    return np.sqrt(abs(VnSquare/denominateur))


""" Convert cm to inches     
"""

def cm2inch(value):
    return value/2.54


""" """
""" Find outlyingness for a specific dimension N                                                                        """
"""                                                                                     """
"""   Boucle sur chaque point, bootstrap sur le reste pour n fixé                         

      Calcul de h_i = median( n * V_n / V_{n-1} ) , 
                avec Vn incluant i, 
                V_{n-1} ne l'incluant pas.                                                """

"""   Calcul de h_i / median(delta_.,i)                                                   """

"""   Calcul de h_signif = median( h_i / median(delta_i,.) )_90%, 
                sur les x% de h_i les moins élevés.
                Est-ce > cutoff ? Si oui, ajouter cette dimension supplémentaire.         """

"""   Calculer et collecter : delta^corr_i² = delta_i² - h_i² ,                           """
"""             dès que : h_i / median(delta_i,.) > cutoff ,                          """
"""             mais :    median( h_i / median(delta_i,.) ) < cutoff.                 

      Calculer l'estimée de negentropie Jn pour l'ensemble des h (trimmed)
      (See FastICA publications, Aapo Hyvärinen, Helsinki University of Technology)
                J = [ E G(u) - E G(v) ]**2
                avec G(u) = 1/alpha log cosh(alpha*u), ou
                     G(u) = -exp(-u**2/2),
                avec v = N(0,1) et
                u = h / stddev(h), où h fait partie de l'ensemble trimmed des h.
                
                Remarque 1 : 
                Le signe, manquant dans h et donc dans u, n'est pas nécessaire au calcul de G.
                Remarque 2 :
                Il n'est pas nécessaire de centrer, car h correspond à la distance
                à l'origine, où l'origine est le volume V_{n-1}. 
                Origine = V_{n-1} = "base" du simplexe de dimension n.

"""
def nSimplwhichh(N,data,trim,n,seed=1,figpath=os.getcwd(), verbose=False):
    alea.seed(seed)
    h=N*[float('NaN')]
    hs=N*[float('NaN')]
    Jhn=[]
    
    Lfig=False
    if Lfig: 
        fig, axes = plt.subplots(nrows=int(np.ceil(25/5.)), ncols=5, figsize=(cm2inch(50), cm2inch(30)))
        axe = axes.ravel()
        j = 0
    
    #""" Calcul de h_i pour tout i """
    for i in range(N):
        #for i in alea.sample(range(N),25):
        if verbose:
            print ("i=",i)
        
        hcollection=[]
        B=1000
        countVzero = 0
        for b in range(B):
            indices  = alea.sample( [x for x in range(N) if x != i] , (n-1)+1 )
            Vn       = nSimplexVolume( [i]+indices , data, denominateurexact=False)
            Vnm1     = nSimplexVolume( indices, data, denominateurexact=False)
            if Vnm1!=0:
                #hcourant = n * Vn / Vnm1
                hcourant =  Vn / Vnm1 / np.sqrt(2.)
                hcollection.append( hcourant )
            else:
                countVzero += 1
        
        B = B - countVzero
        
        #print "hcollection=",hcollection
        if Lfig:
            axe[j].hist(hcollection) ; j += 1
        
        h[i] = np.median(hcollection)
        
        hcollectionorder = [hcollection[b] for b in np.argsort(hcollection)]
        hcollectiontrim  = hcollectionorder[ int(B*(1-trim)/2) : int(B - B*(1-trim)/2) ] 
        hs[i] = np.std(hcollectiontrim)
        
        #""" Calculer l'estimée de kurtosis robuste pour hcollection (trimmed) """
        #Cf. J. J. A. Moors, “A quantile alternative for kurtosis”, The Statistician, 37, pp. 25-32, 1988.
        E = np.percentile(hcollection,[25.,50.,75.])
        #print "i,E : ",i,E
        kurtoct = ((E[2]-E[0])/E[1] - 1.23)
        #E = np.percentile(hcollection,[12.5,25.,37.5,50.,62.5,75.,87.5])
        #kurtoct = abs(((E[6]-E[4])+(E[2]-E[0]))/(E[5]-E[1]) - 1.23)
        
        #""" Calculer l'estimée de négentropie Jn pour hcollection (trimmed) """
        #seed=696969 ; alea.seed(seed)
        alpha = 1.0 #  "constant in range [1, 2]"  (fastICA R package help pages)
        # estimée, N(0,1)
        v=np.array([ alea.gauss(mu=0.,sigma=1.) for b in range(100000) ])
        EG_lch_v = np.mean( 1/alpha * np.log( np.cosh(alpha * v) ) )
        EG_exp_v = np.mean( -np.exp( -v**2 / 2 ) )                  #[EG_lch_v,EG_exp_v]
        # estimée, h
        EG_lch_u = np.mean( 1/alpha * np.log( np.cosh(alpha * (hcollectiontrim / np.std(hcollectiontrim)) ) ) )
        EG_exp_u = np.mean( -np.exp( -(hcollectiontrim / np.std(hcollectiontrim))**2 / 2 ) )   #[EG_lch_u,EG_exp_u]
        ## estimée, N(0,1)
        #v=np.array([ alea.gauss(mu=0.,sigma=1.) for b in range(100000) ])
        #EG_lch_v = np.mean( 1/alpha * np.log( np.cosh(alpha * v) ) )
        #EG_exp_v = np.mean( -np.exp( -v**2 / 2 ) )                  #[EG_lch_v,EG_exp_v]
        ## estimée, h
        #EG_lch_u = np.mean( 1/alpha * np.log( np.cosh(alpha * (hcollectiontrim / np.std(hcollectiontrim)) ) ) )
        #EG_exp_u = np.mean( -np.exp( -(hcollectiontrim / np.std(hcollectiontrim))**2 / 2 ) )   #[EG_lch_u,EG_exp_u]
        # Négentropie : différence entre h et N(0,1)
        #print (EG_lch_u - EG_lch_v)**2 , (EG_exp_u - EG_exp_v)**2 , kurtoct
        Jhn.append([ (EG_lch_u - EG_lch_v)**2 , (EG_exp_u - EG_exp_v)**2 , kurtoct ])
    
    if Lfig:
        fig.subplots_adjust(left=0.03, right=0.97)
        fig.suptitle("n="+str(n))
        #plt.show()
        plt.savefig(figpath+"/hcollection_n"+str(n)+".png") ; plt.close("all")
    
    
    return h,hs,Jhn

""" 
    remove out - n - ness 
"""
def nSimpldimred(data, hn, cutoff=0.1, mode="autre"):
    """
    Inputs :
      data    Square distance matrix
      hn      Component outlying from dimension n, output of h_hs_Jhn
                hn is used to calculate decisive parameter honmediandist, 
                which serves to flag a point as an outlier
      cutoff  Threshold defining outlying points :
                a point is flagged outlier if honmediandist[point] > cutoff
      mode    "commun"                 : if outlyingness is likely to be shared between 2 points
              !="commun", e.g. "autre" : if outliers are isolated 
                                         (can be interpreted as per-point noise).
    Outputs :
      cdata   Square distance matrix devoid of outlyingness out of dimension n.
    """
    # Decisive parameter h / median(dist)
    h=[1.0*x for x in hn]   #1.0*hn #1.0* : to avoid replacing hn with potentially changing values of h
    honmediandist = (h / np.median(data,axis=0))**2      # ? : *N/(N-1) to remove bias due to the 0 ?
    #print "    med honmediandist",np.median(honmediandist)
    #print "    std              ",np.std(honmediandist)
    #print "    max              ",np.max(honmediandist)
    
    #""" Calculer et collecter : delta^corr_i ² = delta_i ² - h_i ² """
    cdata = 1.0*data
    #
    ## outliers
    #
    list_outliers = list(np.where( honmediandist > cutoff )[0])
    #print "outliers : "+str(list_outliers)
    #print "len(list_outliers)="+str(len(list_outliers))
    #
    ## Reduce dimension
    #
    N=np.shape(data)[0]
    for i in list_outliers:
        for j in [x for x in range(N) if x!=i]:
            if mode=="commun":
                hij = (h[i]+h[j])/2
                cdata[i,j] = cdata[j,i] = np.sqrt(np.max((0.,(data[i,j]**2 - hij**2))))
            else: #outlier(s) isolés
                cdata[i,j] = cdata[j,i] = np.sqrt(np.max((0.,(cdata[i,j]**2 - h[i]**2))))
                # Limitation : ce calcul ne tient pas compte d'éventuels clusters d'outliers.
    return cdata, [len(list_outliers), np.median(honmediandist), np.std(honmediandist), np.max(honmediandist)]




""" 
    Classical multidimensional scaling
""" 

def cMDS(D,alreadyCentered=False):
    """
    D: distance / dissimilarity matrix, square and symetric, diagonal 0
    """
    (p,p2)=np.shape(D)
    if p != p2:
        sys.exit("D must be symetric...")
        
    # Double centering
    if not alreadyCentered:
        J=np.eye(p)-np.ones((p,p))/p
        Dcc=-0.5*np.dot(J,np.dot(D**2,J))
    else: 
        # allows robust centering with mediane and mad
        # outside this routine
        Dcc=D
    
    # Eigenvectors
    evals, evecs = np.linalg.eigh(Dcc)
    
    # Sort by eigenvalue in decreasing order
    idx = np.argsort(abs(evals))[::-1]
    #idx = np.argsort(evals)[::-1]
    evecst = evecs[:,idx]
    evalst= evals[idx]    
    
    # Undelying coordinates 
    idxPos,=np.where(evalst>0)
    Xe=np.dot(evecst[:,idxPos],np.diag(evalst[idxPos]**0.5))
    
    return evalst[idxPos], evecst[:,idxPos], Xe
    

""" 
    Find out which dimension is most relevant for data
    (there is room for improvement regarding detection of the dimension) 
"""    
def nSimplwhichdim(data,cutoff,trim,ngmetric="negentropy",racine=13378451,n0=2,nf=5):
    
    """ Initialisations """
    N=np.shape(data)[0]
    dico_h_hs_Jhn     = {}
    dico_hsignif    = {}
    dico_outliers   = {}
    dico_negentropy = {}
    dico_cdata      = {}
    alea.seed(racine)
    stop=False
    
    """ Boucle sur n """
    for n in range(n0,nf):
        
        if not stop:
            
            h,hs,Jhn = nSimplwhichh(N,data,trim,n,seed=n+1) 
            print ("n, neg(log.cosh), neg(exp), rkurtosis : ", np.mean(np.array(Jhn),0))
            dico_h_hs_Jhn[n] = h,hs,Jhn
            
            horder = np.argsort(h)
            htrim  = [h[i] for i in horder][ int(N*(1-trim)/2) : int(N - N*(1-trim)/2) ]
            
            #""" Calcul de h_i / median(delta_.,i), pour détection d'outliers                                                   """
            honmediandist = (h / np.median(data,axis=0))**2      # ? : *N/(N-1) to remove bias due to the 0 ?
            
            #""" Calcul de h_signif = median( h_i / median(delta_i,.) )_trim% """
            #honmediandisttrim = [honmediandist[i] for i in np.argsort(h)][:int(N*trim)]
            #hsignif = np.median(honmediandisttrim)
            #print(hsignif)
            #plt.figure() ; plt.hist(h);plt.title("n="+str(n))  #plt.show()
            
            dataorder = np.argsort(squareform(data))
            datatrim  = [squareform(data)[i] for i in dataorder][:int(N*(N-1)/2*trim)]
            
            #hsignif = np.sum(np.array(htrim)**2) / np.sum(np.array(datatrim)**2)
            nm1 = max(n0,n-1)
            #Jh_{n}
            JhnOrd = np.argsort(np.array(Jhn)[:,0])
            Jhntrim = [Jhn[i] for i in JhnOrd][ int(N*(1-trim)/2) : int(N - N*(1-trim)/2) ]
            #Jh_{n-1}
            Jhnm1 = dico_h_hs_Jhn[nm1][2]
            Jhnm1Ord = np.argsort(np.array(Jhnm1)[:,0])
            Jhnm1trim = [Jhnm1[i] for i in Jhnm1Ord][ int(N*(1-trim)/2) : int(N - N*(1-trim)/2) ]
            # decision variable hsignif
            if ngmetric=="negentropy":
                hsignif = np.mean(np.array(Jhn)[:,0]) / np.mean(np.array(dico_h_hs_Jhn[(nm1)][2])[:,0])
            elif ngmetric=="negentropyexp":
                hsignif = np.mean(np.array(Jhn)[:,1]) / np.mean(np.array(dico_h_hs_Jhn[(nm1)][2])[:,1])
            elif ngmetric=="rkurtosis":
                hsignif = np.mean(np.array(Jhn)[:,2]) / np.mean(np.array(dico_h_hs_Jhn[(nm1)][2])[:,2])
            
            #hsignif = np.mean(np.array(Jhn)[:,1]) / np.mean(np.array(dico_h_hs_Jhn[(nm1)][2])[:,1])
            #hsignif = np.mean(np.array(Jhn)[:,0]) / np.mean(np.array(dico_h_hs_Jhn[(nm1)][2])[:,0])
            print ("hsignif (negentropy(logcosh), neg(exp) , rkurtosis) = "+str(np.mean(np.array(Jhn)[:,0]) / np.mean(np.array(dico_h_hs_Jhn[(nm1)][2])[:,0]))+","+str(np.mean(np.array(Jhn)[:,1]) / np.mean(np.array(dico_h_hs_Jhn[(nm1)][2])[:,1]))+","+str(np.mean(np.array(Jhn)[:,2]) / np.mean(np.array(dico_h_hs_Jhn[(nm1)][2])[:,2])))
            ##hsignif = np.mean(np.array(Jhn)[:,1]) / np.mean(np.array(dico_h_hs_Jhn[(nm1)][2])[:,1])
            ##hsignif = np.median(np.array(Jhn)[:,0]) / np.median(np.array(dico_h_hs_Jhn[(nm1)][2])[:,0])
            ##hsignif = np.mean(Jhntrim) / np.mean(Jhnm1trim)
            dico_hsignif[n]=(hsignif,np.max(h/np.median(data,axis=0)))
            
            #""" Calculer et collecter : delta^corr_i ² = delta_i ² - h_i ² """
            cdata = 1.0*data
            #cutoff negentropy : 0.5
            
            dico_outliers[n] = list(np.where( honmediandist >cutoff)[0])
            print ("Il y a "+str(len(dico_outliers[n]))+" outliers : "+str(dico_outliers[n]))
            
            if hsignif >= cutoff and len(dico_outliers[n])>np.ceil((1-trim)*N):
                
                print ("hsignif >= cutoff, le nombre estimé de dimensions dans les données est d'au moins : ",n,".")
                
            else:
                
                dico_outliers[n] = list(np.where( honmediandist >cutoff)[0])
                
                #if n in dico_outliers.keys():
                for i in dico_outliers[n]:
                    for j in [x for x in range(N) if x!=i]:
                        #cdata[i,j] = cdata[j,i] = np.sqrt(np.max((0.,(data[i,j]**2 - h[i]**2))))
                        cdata[i,j] = cdata[j,i] = np.sqrt(np.max((0.,(cdata[i,j]**2 - h[i]**2))))
                        # Limitation : le calcul ci-dessus ne tient pas compte d'éventuels clusters d'outliers.
                
                if len(dico_outliers[n])>0: 
                    dico_cdata[n] = cdata
                
                if hsignif >= cutoff:
                    
                    print ("hsignif >= cutoff aprés réduction de dimension, le nombre estimé de dimensions dans les données est d'au moins : ",n,".")
                    h2,hs2,Jhn2 = nSimplwhichh(N,cdata,trim,n,seed=n+1)
                    print ("n, neg(log.cosh), neg(exp), rkurtosis aprés réductiond de dimension: ", np.mean(np.array(Jhn),0))
                    
                    if ngmetric=="negentropy":
                        hsignif2 = np.mean(np.array(Jhn2)[:,0]) / np.mean(np.array(dico_h_hs_Jhn[(nm1)][2])[:,0])
                    elif ngmetric=="negentropyexp":
                        hsignif2 = np.mean(np.array(Jhn2)[:,1]) / np.mean(np.array(dico_h_hs_Jhn[(nm1)][2])[:,1])
                    elif ngmetric=="rkurtosis":
                        hsignif2 = np.mean(np.array(Jhn2)[:,2]) / np.mean(np.array(dico_h_hs_Jhn[(nm1)][2])[:,2])
                    
                    print ("hsignif2="+str(np.mean(np.array(Jhn2)[:,0]) / np.mean(np.array(dico_h_hs_Jhn[(nm1)][2])[:,0]))+","+str(np.mean(np.array(Jhn2)[:,1]) / np.mean(np.array(dico_h_hs_Jhn[(nm1)][2])[:,1]))+","+str(np.mean(np.array(Jhn2)[:,2]) / np.mean(np.array(dico_h_hs_Jhn[(nm1)][2])[:,2])))
                    
                    if hsignif2 < cutoff:
                        
                        dico_h_hs_Jhn[n] = h2,hs2,Jhn2
                        print ("hsignif2 < cutoff, le nombre estimé de dimensions dans les données est : "+str(n-1)+".")
                        stop=True
                        
                    else:
                        print ("hsignif2 > cutoff, le nombre estimé de dimensions dans les données est d'au moins : ",n,".")
                    
                else:
                    print ("hsignif > cutoff, le nombre estimé de dimensions dans les données est : "+str(n-1)+".")
                    stop=True
    
    
    #plt.show()
    
    return dico_h_hs_Jhn, dico_hsignif, dico_outliers,dico_cdata



