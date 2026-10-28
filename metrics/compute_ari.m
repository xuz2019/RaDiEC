function AR = compute_ari(c1, c2)

x = double(c1(:));
y = double(c2(:));
N = numel(x);

assert(N == numel(y), '标签长度不一致');

if N < 2
    AR = NaN;
    return;
end

k1 = max(x) - min(x) + 1;
k2 = max(y) - min(y) + 1;

if all(isfinite(x)) && all(isfinite(y)) && ...
        all(x == fix(x)) && all(y == fix(y)) && ...
        all(abs(x) <= flintmax) && ...
        all(abs(y) <= flintmax) && ...
        k1 >= 1 && k2 >= 1 && k1*k2 <= 1e6

    idx = (x-min(x)+1) + k1*(y-min(y));
    C = reshape(accumarray(idx,1,[k1*k2,1]),k1,k2);

else

    [~,~,a] = unique(x);
    [~,~,b] = unique(y);
    C = sparse(a,b,1);
end

z = nonzeros(C);
a = sum(z.*(z-1))/2;

s1 = full(sum(C,2));
s2 = full(sum(C,1));

b = sum(s1.*(s1-1))/2;
c = sum(s2.*(s2-1))/2;

total = N*(N-1)/2;
expected = b*c/total;
denominator = (b+c)/2 - expected;

if denominator == 0
    AR = 0;
else
    AR = (a-expected)/denominator;
end
end
