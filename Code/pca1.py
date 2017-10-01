import numpy as np
from numpy import linalg as LA
import pandas as pd
import csv
from sklearn.decomposition import TruncatedSVD
import time
from sklearn.manifold import TSNE
from numpy import inf


test=np.loadtxt('pca_c.txt', delimiter="\t",dtype='str').shape[1] #read the file which contains the original matrix

initial = np.loadtxt('pca_c.txt', delimiter="\t",dtype='str')
lastcol= initial[:,-1]
matrix = np.loadtxt('pca_c.txt', delimiter="\t",usecols=range(test-1))
row=len(matrix)
t=np.asmatrix(lastcol, dtype='U')
t=np.transpose(t)


matrix_t=np.transpose(matrix)
#normalizing the matrix by calculating mean of every column and then subtracting the mean of a particular column from that column
mean=np.mean(matrix,axis=0)
normalize=matrix-mean
col= normalize.shape[1];
matrix_calc=np.matrix(normalize[0,:])

print(matrix_calc.shape)
for i in range(1,col):
	mat_col=np.matrix(normalize[i,:])
	
	matrix_calc =np.vstack((matrix_calc,mat_col))

print(matrix_calc.shape)

#calculate the covariance matrix 
covmat=np.cov(matrix_calc)


#calculate the eigen values and eigen vectors
eigvals, eigvecs = LA.eig(covmat)
idx = eigvals.argsort()[::-1]   
eigvecs = eigvecs[idx]
eigvecs = eigvecs[:,idx]
w=eigvecs[:,:2]
w_trans=np.transpose(w)
matrix_transpose=np.transpose(matrix)

#computing the reduced dimensionality matrix
final_matrix = np.dot(w_trans,matrix_transpose)
final_matrix= np.transpose(ans)


print("ans", final_matrix.shape)
print("t", t.shape)
temp=np.concatenate((final_matrix,t),axis=1)

 
df=pd.DataFrame(temp)
df.to_csv("pca_c_file.csv")

#computing the Singular Value Dimension (SVD)
svd= TruncatedSVD(n_components=2)
svd_result=svd.fit_transform(matrix)
svd_result=np.concatenate((svd_result,t),axis=1)
df_svd=pd.DataFrame(svd_result)
df.to_csv("svd_c_file.csv")


#computing the t-SNE of the original matrix
tsne=TSNE(n_components=2)
np.set_printoptions(suppress=True)
tsne_result=tsne.fit_transform(matrix)
tsne_result=np.concatenate((tsne_result,t),axis=1)
df_svd=pd.DataFrame(svd_result)
df.to_csv("tsne_c_file.csv")




