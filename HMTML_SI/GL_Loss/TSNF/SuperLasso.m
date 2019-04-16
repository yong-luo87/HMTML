function [err, X] = SuperLasso(A, X0, B, lambda, maxIter, tol, SIGN)
[n, p] = size(A);
[p, m] = size(X0);

ATA = A'*A;
ATB = A'*B;

D = diag(ATA);
ATA(logical(eye(p))) = 0;


% tic;
% for i = 1:p
%     D(i) = ATA(i,i);
%     ATA(i,i) = 0;
% end
% toc;

X = X0;
for n = 1:length(lambda)
   ATBLambda = ATB - lambda(n);
   iterNum = 0;
   err = inf;
   while(err > tol && iterNum < maxIter)
      XLast = X; 
      for j = 1:p
%          x = (ATBLambda(j, :) - ATA(j,:) * X) / D(j);
%          minusPos = x<0;
%          x(minusPos) = 0;
%          X(j, :) = x;
           X(j, :) = max(0, (ATBLambda(j, :) - ATA(j,:) * X) / D(j));
      end
      err = sum(sum(abs(X-XLast)));    
      iterNum = iterNum+1;
      %err = sum(sum(abs(A*X-B))) + lambda(n) * sum(sum(X))
   end
   %subIter = iterNum
end
X = X';
%%check 0 %%%%%
if m == 1 || SIGN == 0
    return;
else
    count = 0;
    site = sum(X.^2)< 1e-10;
    for i = 1:p
        if site(i) == 1
            ss = ceil(sum(sum(rand(2,2)))/4*m);
            offset = ceil(rand(1,1)*p);
            if i == 1
%                 X(1:floor(m/2), i) = X(1:floor(m/2), p);
%                 X(1:floor(m/2), p) = 0;

%                 X(1:ss, i) = X(1:ss, p);
%                 X(1:ss, p) = 0;
                
                X(1:ss, i) = X(1:ss, offset);
                X(1:ss, offset) = 0;
            else
                X(1:ss, i) = X(1:ss, offset);
                X(1:ss, offset) = 0;
            end
        end
    end
    count = sum(site);
end
%%%%%%%

