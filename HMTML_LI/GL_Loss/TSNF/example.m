clear;
clc;
addpath('../tensor_toolbox');
addpath('../tensor_toolbox/met');
%%%%%%%%%%%parameter setting%%%%%%%%%%%%%%%%%%%%%
% parameters for the main procedure
INTERVAL = 1;   % sample frequency
R = [10, 10, 10];         % code tensor size
iterMax = 200;    % iteration times
ERRORDXTOLERANCE = 1e-2;    % the error tolerance for the function
ERRORDFTOLERANCE = 1e-2;    % the error tolerance for the solution variance

% parameters for the lasso algorithm
L_maxIter = 1000;   % the data for 
L_tol = 1e-2;
L_lambda_max = [0.1, 0.1, 0.1]';
L_Lambda = L_lambda_max*[100, 50, 10, 1];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%% data initial %%%%%%%%%%%%%%%%%%%%%%%%%%
data = rand(100,100,100);      % randomly generate the data
dataT = tensor(data);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% tensor factorization %%%%%%%%%%%%%%%
tic;
[outT, errdx, errdf, iternum] = TSNF(dataT, R, iterMax,...
    ERRORDXTOLERANCE,...
    ERRORDFTOLERANCE,...
    L_maxIter,...
    L_tol,...
    L_Lambda);
toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% result output %%%%%%%%%%%%%%%%%%%%%%
%figure, plot(errdx);
figure, plot(errdf); title('objective');
iternum
[zeroRitio, compressRitio, errorRitio] = Evaluation(dataT, outT, 1e-5)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

