function [tarMets] = HMTML(singleSrcTrnFeaL, singleTarTrnFeaL, singleSrcTrnLabelsL, singleTarTrnLabelsL, ...
    src_base_cls, tar_base_cls, set, para, option)
% -------------------------------------------------------------------------
% Heterogenous Multi-Task Metric Learning
% Input: src_base_cls, tar_base_cls are classifier weight vectors in the source
% and target domains respectively.
% Note: require to install matlab tensor toolbox, which is available at:
% https://www.sandia.gov/~tgkolda/TensorToolbox/index-2.6.html
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% Initialization
% -------------------------------------------------------------------------
nbP = size(tar_base_cls{1}, 2);
nbV = set.nbTarV + set.nbSrcV;
matUs = cell(nbV, 1);
for v = 1:set.nbTarV
    tarFeaDim(v) = size(tar_base_cls{v}, 1);
    rand('seed', v);
    matUs{v} = rand(tarFeaDim(v), para.rDim);
end
for v = (set.nbTarV+1):nbV
    srcFeaDim(v-set.nbTarV) = size(src_base_cls{v-set.nbTarV}, 1);
    rand('seed', v);
    matUs{v} = rand(srcFeaDim(v-set.nbTarV), para.rDim);
end

% ---------------------------------------------------------------------
% Compute the pairwise features and labels
% ---------------------------------------------------------------------
% fprintf('Computing the pairwise features and labels ... ');
if set.nbSrcV == 0
    nbLs = set.nbTarL;
    feaDims = tarFeaDim;
else
    nbLs = [set.nbTarL set.nbSrcL];
    feaDims = [tarFeaDim srcFeaDim];
end
Delta = cell(nbV, 1); vecYs = cell(nbV, 1);
for v = 1:set.nbTarV
    set.nbPw(v) = (set.nbTarL(v)*(set.nbTarL(v)-1)) / 2;
    Delta{v} = zeros(set.nbPw(v), feaDims(v));
    vecYs{v} = zeros(set.nbPw(v), 1);
    k = 1;
    for i = 1:nbLs(v)
        for j = (i+1):nbLs(v)
            Delta{v}(k,:) = singleTarTrnFeaL{v}(i,:) - singleTarTrnFeaL{v}(j,:);
            if singleTarTrnLabelsL{v}(i) == singleTarTrnLabelsL{v}(j)
                vecYs{v}(k) = 1;
            end
            k = k + 1;
        end
    end
    Delta{v} = Delta{v}';
    vecYs{v}(vecYs{v} == 0) = -1;
end
for v = (set.nbTarV+1):nbV
    v2 = v - set.nbTarV;
    set.nbPw(v) = (set.nbSrcL(v2)*(set.nbSrcL(v2)-1)) / 2;
    Delta{v} = zeros(set.nbPw(v), feaDims(v));
    vecYs{v} = zeros(set.nbPw(v), 1);
    k = 1;
    for i = 1:nbLs(v)
        for j = (i+1):nbLs(v)
            Delta{v}(k,:) = singleSrcTrnFeaL{v2}(i,:) - singleSrcTrnFeaL{v2}(j,:);
            if singleSrcTrnLabelsL{v2}(i) == singleSrcTrnLabelsL{v2}(j)
                vecYs{v}(k) = 1;
            end
            k = k + 1;
        end
    end
    Delta{v} = Delta{v}';
    vecYs{v}(vecYs{v} == 0) = -1;
end
% fprintf('Finished! \n');

% -------------------------------------------------------------------------
% Pre-calculation
% -------------------------------------------------------------------------
for p = 1:nbP
    w = cell(nbV, 1);
    for v = 1:set.nbTarV
        w{v} = tar_base_cls{v}(:,p);
    end
    for v = (set.nbTarV+1):nbV
        v2 = v - set.nbTarV;
        w{v} = src_base_cls{v2}(:,p);
    end
    cov_w = ktensor(w); clear w
    if p == 1
        tenW = full(cov_w);
    else
        tenW = tenW + full(cov_w);
    end
    clear cov_w
end
tenW = tenW / nbP;
normTenW = norm(tenW)^2;
para.nbP = nbP;

vecNormDDs = cell(nbV, 1);
for v = 1:nbV
    nbPw = length(vecYs{v});
    vecNormDDs{v} = zeros(nbPw, 1);
    for k = 1:nbPw
        vecNormDDs{v}(k) = norm((Delta{v}(:,k)*Delta{v}(:,k)'), 2);
    end
end

% -------------------------------------------------------------------------
% Optimize U{v}, v = 1, ..., V alternatively until converge
% -------------------------------------------------------------------------
if rem(nbV, 2) == 0
    tenI = teneye(nbV, para.rDim);
else
    if para.rDim == 1
        tenI = tensor(para.rDim, ones(1,nbV));
    else
        tenI = GenerateCoreTensor(para.rDim*ones(1,nbV));
    end
end

obj(1,1) = computeObjUs(Delta, vecYs, tenW, tenI, matUs, para);
loop = 1; iter = 1;
while loop
    for v = 1:nbV
        tenBv = ttm(tenI, matUs, -v);
        mat_tenBv = double(tenmat(tenBv, v)); clear tenBv
        
        % ------------------------------------------------------
        % Pre-calculation
        % ------------------------------------------------------
        pre_calc.matBBv = mat_tenBv * mat_tenBv';
        pre_calc.normBBv = norm(pre_calc.matBBv, 2);
        
        mat_tenWv = double(tenmat(tenW, v));
        pre_calc.matWBv = mat_tenWv * mat_tenBv';
        pre_calc.normWv = normTenW; % norm(mat_tenWv,'fro')^2;
        clear mat_tenWv mat_tenBv
        
        pre_calc.vecNormDDv = vecNormDDs{v};
        
        [matUv, obj_Phi_1(v), obj_Omega(v)] = ...
            optimizeUv_OGM(Delta{v}, vecYs{v}, matUs{v}, pre_calc, para);
        matUs{v} = matUv; clear matUv
        clear pre_calc
    end
    
    iter = iter + 1;
    % ------------------------------------------------------
    % Update the objective value
    % ------------------------------------------------------
    obj(iter,1) = computeObjUs_Pre(obj_Phi_1, obj_Omega, tenW, tenI, matUs, para);
    % obj(iter,1) = computeObjUs(Delta, vecYs, tenW, tenI, matUs, para);
    
    % ------------------------------------------------------
    % Check convergence
    % ------------------------------------------------------
    % obj_diff = abs(obj(iter,1) - obj(iter-1,1)) / abs(obj(iter,1) - obj(1,1));
    % if abs(obj(iter,1) - obj(1,1)) < eps || obj_diff <= epsilon || iter >= maxit
    %     loop = 0;
    % end
    loop = checkConvergence(obj(iter,1), obj(iter-1,1), obj(1,1), iter-1, set, para, option);
    
    % ------------------------------------------------------
    % Update variables
    % ------------------------------------------------------
    if loop
    end
    
    clear obj_Phi_1 obj_Omega
end

% -------------------------------------------------------------------------
% Derive the target metrics
% -------------------------------------------------------------------------
tarMets = cell(set.nbTarV, 1);
for v = 1:set.nbTarV
    tarMets{v} = matUs{v} * matUs{v}';
end

end


function obj = computeObjUs_Pre(obj_Phi_1, obj_Omega, tenW, tenI, matUs, para)

nbV = length(matUs);

tenA = ttm(tenI, matUs);

obj_Phi_temp2 = norm(minus(full(tenW),full(tenA)))^2;
obj_Phi_2 = para.gammaA * obj_Phi_temp2;


obj = sum(obj_Phi_1(:)) + obj_Phi_2 + sum(obj_Omega(:));

end


function obj = computeObjUs(Delta, vecYs, tenW, tenI, matUs, para)

mu = para.mu;
rho = para.rho;

nbV = length(matUs);

for v = 1:nbV
    nbPw = length(vecYs{v});
    [feaDim, rDim] = size(matUs{v});
    
    vecUD = cell(nbPw, 1);
    vecZv = zeros(nbPw, 1);
    for k = 1:nbPw
        vecUD{k} = matUs{v}' * Delta{v}(:,k);
        vecZv(k) = vecYs{v}(k) * (1-(vecUD{k}'*vecUD{k}));
    end
    
    obj_Phi_temp1 = zeros(nbPw, 1);
    for k = 1:nbPw
        temp_exp = exp(-rho*vecZv(k));
        if isinf(temp_exp)
            obj_Phi_temp1(k) = -vecZv(k);
        else
            obj_Phi_temp1(k) = (1.0/rho)*log(1.0+temp_exp);
        end
        clear temp_exp
    end
    
    clear vecUD vecZv coeff
    
    
    idx1 = find(matUs{v} < -mu);
    idx2 = find(matUs{v} > mu);
    idx3 = setdiff((1:(feaDim*rDim))', [idx1; idx2]);
    
    obj_Omega_temp = zeros(size(matUs{v}));
    obj_Omega_temp(idx1) = -matUs{v}(idx1) - 0.5*mu;
    obj_Omega_temp(idx2) = matUs{v}(idx2) - 0.5*mu;
    obj_Omega_temp(idx3) = matUs{v}(idx3).^2 / (2.0*mu);
    
    clear idx1 idx2 idx3
    
    obj_Phi_1(v) = 1.0/nbPw*sum(obj_Phi_temp1(:));
    obj_Omega(v) = para.gamma*sum(obj_Omega_temp(:));
    clear obj_Phi_temp1 obj_Omega_temp
end


tenA = ttm(tenI, matUs);

obj_Phi_temp2 = norm(minus(full(tenW),full(tenA)))^2;
obj_Phi_2 = para.gammaA * obj_Phi_temp2;


obj = sum(obj_Phi_1(:)) + obj_Phi_2 + sum(obj_Omega(:));

end

