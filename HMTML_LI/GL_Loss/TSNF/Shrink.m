function outM = Shrink(inM, ratio, VALUE) %set the elements at the bottom ratio percentage to Value
R = size(inM);
num = prod(R);
data = reshape(inM, [1, num]);
list = sort(data);
lowBound = list(ceil(ratio*num));
for i = 1:num
    if data(i) <= lowBound
        data(i) = VALUE;
    end
end
outM = reshape(data, R);