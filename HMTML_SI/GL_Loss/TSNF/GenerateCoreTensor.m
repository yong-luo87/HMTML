function coreTensor = GenerateCoreTensor(rList)
%coreTensor = tenzeros(rList);
% r = min(rList);
% for i = 1:r
%     coreTensor(i*ones(length(rList))) = 1;
% end
mode = rList;
[a, order] = sort(mode);
notationList = zeros(a(end), length(mode));
notationList(:, order(end)) = [1 : a(end)]';

for i = 1:length(mode)-1
    if prod(a(1:i)) >= a(end)
        if i == 1
            notationList(:, order(i)) = [1:a(end)]';
            break;
        end
        mulF = prod(a(1:i-1));
        if mulF > a(i)
            m = order(i);
            M = order(1:i);
            for k = 1:a(end)
                notationList(k, m) = ceil(k * mode(m) / a(end));
            end
            combination = zeros(mulF, i-1);
            mul = 1;
            for t = 1:i-2
                mul(t) = prod(a(t+1:i-1));
            end
            for q = 1:mulF
                combination(q, 1) = floor((q-1) / mul(1)) + 1;
                for p = 2:i-1
                    combination(q, p) = floor((q-1 - (combination(q, 1:p-1)-1)...
                        *mul(1:p-1)') / prod(a(p+1:i-1))) + 1;
                end
            end
            %order = zeros(prod(a(1:i-1), i-1));
            for p = 1:i-1
                site = order(p);
                for k = 1:a(end)     
%                     notationList(k, site) = ceil(mod(k, mulF) ...
%                         / prod(a(p+1:i-1)));
                    notationList(k, site) = combination(mod(k-1, mulF)+1, p);
                end
            end
        else
            M = order(i);
            m = order(1:i-1);
            for k = 1:a(end)
                notationList(k, M) = mod(k-1, mode(M))+1;
            end
            combination = zeros(mulF, i-1);
            mul = 1;
            for t = 1:i-2
                mul(t) = prod(a(t+1:i-1));
            end
            for q = 1:mulF
                combination(q, 1) = floor((q-1) / mul(1)) + 1;
                for p = 2:i-1
                    combination(q, p) = floor((q-1 - (combination(q, 1:p-1)-1)...
                        *mul(1:p-1)') / prod(a(p+1:i-1))) + 1;
                end
            end
            for p = 1:i-1
                site = order(p);
                for k = 1:a(end)
                    notationList(k, site) = combination(ceil(k*mulF/a(end)), p);
                end
            end
        end
        break;
    end
end 

for p = i+1 : length(mode)-1
    site = order(p);
    for q = 1:a(end)
        notationList(q, site) = mod(q-1, mode(site)) + 1;
    end
end

coreTensor = tensor(double(sptensor(notationList, 1, mode)));
% table = ones(1, length(mode));
% for i = 1:length(mode)
%     table(i) = prod(mode(1:i-1));
% end
% c = zeros(prod(mode), 1);
% for i = 1:a(end)
%     c(table * (notationList(i, :)-1)' + 1) = 1;
% end
% 
% coreTensor = tensor(c, mode);
