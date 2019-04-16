function [matUv_opt, obj_Phi_1_opt, obj_Omega_opt] = ...
    optimizeUv_OGM(Delta_v, vecYv, matUv, pre_calc, para)
% -------------------------------------------------------------------------
% Optimization of the projection matrix
% -------------------------------------------------------------------------

maxit = 200;
mu = para.mu;
rho = para.rho;
epsilon = 1e-3;

% -------------------------------------------------------------------------
% Optimize Uv (under the non-negative constraints) using the optimal 
% gradient method (OGM) utilized in 'NeNMF, Guan et al., 2012'
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% Initialization of the objective, gradient, and Lipschitz constant
% -------------------------------------------------------------------------
[obj(1,1), grad, lipsc, obj_Phi_1, obj_Omega] = ...
    evaluate_cost(Delta_v, vecYv, matUv, mu, rho, pre_calc, para);
Y = matUv; alpha = 1; % matUv_neg1 = Y;

loop = 1; t = 1;
while loop
    % ------------------------------------------------------
    % Calculate matUv_new
    % ------------------------------------------------------
    matUv_new = Y - (1.0/lipsc)*grad; clear grad lipsc
    matUv_new(matUv_new < 0) = 0;
    
    % ------------------------------------------------------
    % Update the combination coefficient
    % ------------------------------------------------------
    alpha_new = 0.5*(1.0 + sqrt(4*alpha^2+1));
    
    % ------------------------------------------------------
    % Update Y
    % ------------------------------------------------------
    Y_new = matUv_new + ((alpha-1.0)/alpha_new)*(matUv_new - matUv);
    
    t = t + 1;
    
    % ------------------------------------------------------
    % Update the objective value, gradient, and Lipschitz
    % ------------------------------------------------------
    [obj(t,1), grad_new, lipsc_new, obj_Phi_1_new, obj_Omega_new] = ...
        evaluate_cost(Delta_v, vecYv, matUv_new, mu, rho, pre_calc, para);
    
    % ------------------------------------------------------
    % Check convergence
    % ------------------------------------------------------
    obj_diff = abs(obj(t,1) - obj(t-1,1)) / abs(obj(t,1) - obj(1,1));
    if abs(obj(t,1) - obj(1,1)) < eps || obj_diff <= epsilon || t >= maxit
        loop = 0;
    end
    
    % ------------------------------------------------------
    % Update variables
    % ------------------------------------------------------
    if loop
        clear matUv alpha Y grad lipsc obj_Phi_1 obj_Omega
        matUv = matUv_new; alpha = alpha_new; Y = Y_new;
        grad = grad_new; lipsc = lipsc_new; obj_Phi_1 = obj_Phi_1_new; obj_Omega = obj_Omega_new;
        clear matUv_new alpha_new Y_new grad_new lipsc_new obj_Phi_1_new obj_Omega_new
    end
end
matUv_opt = matUv_new;
obj_Phi_1_opt = obj_Phi_1_new;
obj_Omega_opt = obj_Omega_new;

end



function [obj, grad, lipsc, obj_Phi_1, obj_Omega] = ...
    evaluate_cost(Delta_v, vecYv, matUv, mu, rho, pre_calc, para)
% function [obj, grad, lipsc, obj_Phi_1, obj_Omega] = ...
%     evaluate_cost(Delta_v, vecYv, mat_tenWvs, mat_tenBv, matUv, mu, rho, pre_calc, para)
% -------------------------------------------------------------------------
% Compute the objective value, gradient and Lipschitz constant
% -------------------------------------------------------------------------

nbPw = length(vecYv);
[feaDim, rDim] = size(matUv);

vecUD = cell(nbPw, 1);
vecZv = zeros(nbPw, 1);
for k = 1:nbPw
    vecUD{k} = matUv' * Delta_v(:,k);
    vecZv(k) = vecYv(k) * (1-(vecUD{k}'*vecUD{k}));
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

grad_Phi_temp1 = zeros(size(matUv));
coeff = zeros(nbPw, 1);
for k = 1:nbPw
    coeff(k) = 2.0*vecYv(k) / (1.0+exp(rho*vecZv(k)));
    grad_Phi_temp1 = grad_Phi_temp1 + coeff(k)*(Delta_v(:,k)*vecUD{k}');
end

lipsc_Phi_temp1 = zeros(nbPw, 1);
for k = 1:nbPw
    lipsc_Phi_temp1(k) = coeff(k)*pre_calc.vecNormDDv(k);
end

clear vecUD vecZv coeff


% obj_Phi_temp2 = norm((mat_tenWv-matUv*mat_tenBv), 'fro')^2;
obj_Phi_temp2 = pre_calc.normWv - 2.0*trace(matUv'*pre_calc.matWBv) ...
    + trace(pre_calc.matBBv*(matUv'*matUv));

% grad_Phi_temp2 = matUv*pre_calc.matBBv - mat_tenWv*mat_tenBv';
grad_Phi_temp2 = matUv*pre_calc.matBBv - pre_calc.matWBv;

lipsc_Phi_temp2 = pre_calc.normBBv;


idx1 = find(matUv < -mu);
idx2 = find(matUv > mu);
idx3 = setdiff((1:(feaDim*rDim))', [idx1; idx2]);

matO = zeros(size(matUv));
matO(idx1) = -1;
matO(idx2) = 1;
matO(idx3) = matUv(idx3) ./ mu;

obj_Omega_temp = zeros(size(matUv));
obj_Omega_temp(idx1) = -matUv(idx1) - 0.5*mu;
obj_Omega_temp(idx2) = matUv(idx2) - 0.5*mu;
obj_Omega_temp(idx3) = matUv(idx3).^2 / (2.0*mu);

grad_Omega_temp = matO;

lipsc_Omega_temp = 1.0 / mu;

clear idx1 idx2 idx3 matO


obj_Phi_1 = 1.0/nbPw*sum(obj_Phi_temp1(:));
obj_Phi =  obj_Phi_1 + para.gammaA*sum(obj_Phi_temp2(:));
obj_Omega = para.gamma*sum(obj_Omega_temp(:));
obj = obj_Phi + obj_Omega;

grad = 1.0/nbPw*grad_Phi_temp1 + 2.0*para.gammaA*grad_Phi_temp2 + para.gamma*grad_Omega_temp;

lipsc = max(lipsc_Phi_temp1(:)) + 2.0*para.gammaA*lipsc_Phi_temp2 + para.gamma*lipsc_Omega_temp;

end

