import numpy as np
import matplotlib.pyplot as mpl

def getcs(l,k,A):
    phi=-(2*A[k,l])/(A[k,k]-A[l,l])
    c=np.sqrt((1+np.sqrt(1/1+phi**2))/2)
    s=np.sqrt((1-np.sqrt(1/1+phi**2))/2)
    return c,s

def remkl(l,k,A):
    c,s=getcs(l,k,A)
    n=A[0].size
    B=A.copy()
    for i in range(n):
        B[k,i]=c*A[k,i]-s*A[l,i]
        B[i,k]=B[k,i]
        B[l,i]=c*A[l,i]+s*A[k,i]
        B[i,l]=B[l,i]
        B[k,l]=0
        B[l,k]=0
        B[k,k]=c**2*A[k,k]+s**A[l,l]-2*c*s*A[k,l]
        B[l,l]=c**2*A[l,l]+s**2*A[k,k]+2*c*s*A[k,l]
        return B
    
    #find the specify which k and l used for each step to find the biggest matrix elements
    def findmax(A):
        imax=0
        jmax=1
        maxval=abs(A[0,1])
        n=(A[0]).size
        for i in range(n):
            for j in range(i+1,n):
                if abs(A[i,j]>maxval):
                    imax=i
                    jmax=j
                    maxval=abs(A[i,j])
    return imax,jmax,maxval

#diagonalise procedure
def diagonalise(A):
    B=A.copy()
    maxc=1
    while maxc>10**(-6):
        i,j,maxc=findmax(B)
        B=remkl(i,j,B)
        mpl.imshow(B,interpolation='nearest')
        mpl.colorbar()
        mpl.show()
    return B

A=np.random.random((4,4))
print(A)
A=A+np.transpose(A)
diagonalise(A)