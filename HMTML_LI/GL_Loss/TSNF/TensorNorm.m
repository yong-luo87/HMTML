% function outValue = TensorNorm(inTensor, p)
% if p <= 0
%     inTensor = tenones(inTensor.size); 
% end
% inTensor = tensor(abs(inTensor.data) .^ p);
% outValue = inTensor.data;
% for i = 1: ndims(inTensor)
%     outValue = sum(outValue, i);
% end
% if p <=0
%     return;
% end
% outValue = outValue ^ (1/p); 

function outValue = TensorNorm(inTensor)
outValue = norm(inTensor.data(:), 'fro');

