%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Sparse Nonnegative Factorization for Tensor
%by Ji Liu
%Sep. 26th 2008
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [outTensor, errordxList, errordfList, iterNum] = TSNF(...
    inTensor,...
    inDimList,...
    inMaxIter,...
    ERRORDXTOLERANCE,...
    ERRORDFTOLERANCE,...
    L_maxIter,...
    L_tol,...
    L_Lambda,...
    U)

%%%%%% initial the outTensor %%%%%%%
modeNum = ndims(inTensor);
outCore = GenerateCoreTensor(inDimList);

if nargin < 9
    U = cell(1, modeNum);
    for i = 1:modeNum
        U{i} = rand(inTensor.size(i), inDimList(i));%...
            %*(abs(max(inTensor.data(:))-min(inTensor.data(:)))/modeNum)^(1/modeNum); 
        %U{i} = ones(inTensor.size(i), inDimList(i))*10;
    end
end

outTensor = ttensor(outCore, U);

L_tolList = inDimList .* inTensor.size * L_tol;
if size(L_Lambda, 1) == modeNum+1
    L_tolList(end+1) = prod(inDimList) * L_tol;
end

L_tolList = ones(inMaxIter, 1)* L_tolList;

errordx = inf;
errordf = inf;
errordxList = zeros(1, inMaxIter);
d0 = 0;
for i = 1:modeNum
    d0 = d0 + L_Lambda(i, end)*sum(sum(outTensor.U{i}));
end
errordf = TensorNorm(full(outTensor) - inTensor)^2 + d0;
errordfList = zeros(1, inMaxIter+2);
errordfList(1, 1:2) = [errordf, errordf-2*ERRORDFTOLERANCE ];
iterNum = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%5
saveForLambda = L_Lambda;
%L_Lambda = L_Lambda * 0.1;
%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% while errordx > ERRORDXTOLERANCE && abs(errordfList(iterNum+1)-...
%         errordfList(iterNum+2)) > ERRORDFTOLERANCE &&...
%         iterNum < inMaxIter
%     iterNum = iterNum + 1
    
for iterNum = 1: inMaxIter
    iterNum
    errordx = 0;
    
    for i = 1:modeNum%modeNum:-1:1%
        midT = tensor(outTensor.core);
        for m = 1:modeNum
            if m == i
                continue;
            end
            midT = ttm(midT, outTensor.U{m}, m);
        end
        unfoldMidT = tenmat(midT, i);   %unfoldi(inTensor) = U{i} * unfolodi(midT)
        unfoldInT = tenmat(inTensor, i);
        
        if iterNum < inMaxIter
            [dx, outTensor.U{i}] = SuperLasso(unfoldMidT.data', outTensor.U{i}',...
                unfoldInT.data', L_Lambda(i, :), L_maxIter, L_tolList(iterNum, i), 1);
        else
            [dx, outTensor.U{i}] = SuperLasso(unfoldMidT.data', outTensor.U{i}',...
                unfoldInT.data', L_Lambda(i, :), L_maxIter, L_tolList(iterNum, i), 0);
        end
        errordx = errordx + dx;
    end
    
%     if size(L_Lambda,1) == modeNum+1 % indicate need to optimize core tensor
%         [err, core] = SuperLassoForCoreTensor(outTensor, ...
%             inTensor, L_Lambda(end, :), L_maxIter(end, :), L_tolList(end));
%         outTensor.core = tensor(core);
%     end
%     optimize the core tensor
%     U = 1;
%     for i = 1:modeNum
%         U = kron(U, outTensor.U{i});
%     end
%     VCore = tenmat(outTensor.core, 1:modeNum);
%     VT = tenmat(inTensor, 1:modeNum);
%     [dx, Core] = SuperLasso(U, VCore', VT', L_Lambda(i, :), L_maxIter, L_tol);
%     errordx = errordx + dx;
%     outTensor.core = reshape(Core, inDimList);
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    errordxList(iterNum) = errordx;
    d0 = 0;
    for i = 1:modeNum
        %d0 = d0 + L_Lambda(i, end)*sum(sum(outTensor.U{i}));
        d0 = d0 + saveForLambda(i, end)*sum(sum(outTensor.U{i}));
    end
    errordf = 0.5*TensorNorm(full(outTensor) - inTensor)^2+d0;
    %errordf = TensorNorm(full(outTensor) - inTensor, 2)^2;
    errordfList(iterNum+2) = errordf;
    
    if errordx < ERRORDXTOLERANCE | abs(errordfList(iterNum+1)-...
        errordfList(iterNum+2)) < ERRORDFTOLERANCE
        break;
    end
end

errordxList = errordxList(1:iterNum);
errordfList = errordfList(3:iterNum+2);% / TensorNorm(inTensor, 2);
