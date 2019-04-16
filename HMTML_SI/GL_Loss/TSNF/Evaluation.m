function [zeroRitio, compressRitio, errorRitio] = Evaluation(inT, outT, tol)
dataNum = 0;
zeroNum = 0;
for i = 1:ndims(inT)
    zeroNum = zeroNum + numel(find(abs(outT.U{i})<tol));
    dataNum = dataNum + prod(size(outT.U{i}));
end
zeroRitio = zeroNum / dataNum;

compressRitio = dataNum / prod(inT.size);

errorRitio = TensorNorm(tensor(outT) - inT) / TensorNorm(inT);