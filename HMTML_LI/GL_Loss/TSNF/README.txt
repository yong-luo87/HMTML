1. install the tensor toolbox in the package "tensor toolbox 2.2"
2. How to use the function "TSNF":

[outT, errdx, errdf, iternum] = TSNF(dataT, R, iterMax,...
    ERRORDXTOLERANCE,...
    ERRORDFTOLERANCE,...
    L_maxIter,...
    L_tol,...
    L_Lambda);

//////////////// input /////////////////////
inData: 	the target matrix (M = A*B)
R: 		the size of the core tensor
iterMax: 	the max iteration#
tol: 		the max tolerance for sparse nonnegative tensor factorization
L_maxIter: 	the max iteration # of LASSO
L_Lambda: 	the penalty factor for sparse term; the larger, the sparser
L_tol:		the max tolerance for LASSO
//////////////////////////////////////////// 

///////////////// output ///////////////////
outT: 		the output tensor (tucker structure);
errdx: 		the difference of the kth and k+1th steps;
errdf:		the function values at all iterations	
iterNum: 	the practical iteration # 
////////////////////////////////////////////

3. please check the "example.m" file

