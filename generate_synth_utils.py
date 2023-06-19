# -*- coding: utf-8 -*-
"""

@author: Mounir
"""
import numpy as np


def create_N_clusters(N,class_,mean_range=[-2,2],sig_range=[0.5,2]):
    classes = np.repeat(class_,N)
    means = np.zeros(N,dtype=object)
    sigs = np.zeros(N,dtype=object)
    for k in range(N):
        means[k]=np.random.uniform(low=mean_range[0], high=mean_range[1], size=2)
        sigs[k]=np.diag(np.random.uniform(low=sig_range[0], high=sig_range[1], size=2))
    return means,sigs,classes

def create_clusters(N0,N1,mean_range=[-2,2],sig_range=[0.5,2]):
    means_0,sigs_0,classes_0=create_N_clusters(N0,0,mean_range=mean_range,sig_range=sig_range)
    means_1,sigs_1,classes_1=create_N_clusters(N1,1,mean_range=mean_range,sig_range=sig_range)
    
    means = np.concatenate((means_0,means_1))
    sigs = np.concatenate((sigs_0,sigs_1))
    classes = np.concatenate((classes_0,classes_1))
    return means,sigs,classes

def translate_N_clusters(N,means,sigs,classes,mean_range=[-2,2],sig_range=[0.5,2]):
    indexes = np.random.choice(classes.size,size=N,replace=False)
    means_new = means.copy()
    sigs_new = sigs.copy()
    classes_new = classes.copy()

    for k in indexes:
        means_new[k]=np.random.uniform(low=mean_range[0], high=mean_range[1], size=2) 
    return means_new,sigs_new,classes_new

def shrink_N_clusters(N,means,sigs,classes,mean_range=[-2,2],sig_range=[0.5,2]):
    indexes = np.random.choice(classes.size,size=N,replace=False)
    means_new = means.copy()
    sigs_new = sigs.copy()
    classes_new = classes.copy()
    for k in indexes:
        sigs_new[k]=np.diag(np.random.uniform(low=sig_range[0], high=sig_range[1], size=2))  
    return means_new,sigs_new,classes_new

def delete_N_clusters(N,means,sigs,classes,mean_range=[-2,2],sig_range=[0.5,2]):
    indexes = np.random.choice(classes.size,size=N,replace=False)
    means_new = means.copy()
    sigs_new = sigs.copy()
    classes_new = classes.copy()
    return np.delete(means_new,indexes,axis=0),np.delete(sigs_new,indexes,axis=0),np.delete(classes_new,indexes,axis=0)

def add_N_clusters(N,class_,means,sigs,classes,mean_range=[-2,2],sig_range=[0.5,2]):
    new_means,new_sigs,new_classes=create_N_clusters(N,class_,mean_range=mean_range,sig_range=sig_range)
    means_n = np.concatenate((means.copy(),new_means))
    sigs_n = np.concatenate((sigs.copy(),new_sigs))
    classes_n = np.concatenate((classes.copy(),new_classes))
    return means_n,sigs_n,classes_n

def generate_samples(n_by_cluster,means,sigs,classes):
    for k,c in enumerate(classes):
        X_ = np.random.multivariate_normal(means[k], sigs[k], size=n_by_cluster)
        y_ = np.repeat(c,n_by_cluster)
        if k == 0:
            X = X_
            y = y_
        else:
            X = np.r_[X,X_]
            y = np.r_[y,y_]
    return X,y