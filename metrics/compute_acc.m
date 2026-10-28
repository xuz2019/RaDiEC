function acc = compute_acc(yTrue, yPred)

yTrue = double(yTrue(:));
yPred = double(yPred(:));

N = numel(yTrue);

if nargin < 3
    K = max(max(yTrue), max(yPred));
end

assert(N > 0 && numel(yPred) == N, ...
    '标签长度不一致或为空');

assert(all(isfinite(yTrue)) && ...
    all(isfinite(yPred)) && ...
    all(yTrue >= 1 & yTrue <= K) && ...
    all(yPred >= 1 & yPred <= K) && ...
    all(yTrue == fix(yTrue)) && ...
    all(yPred == fix(yPred)) && ...
    isscalar(K) && isfinite(K) && ...
    K >= 1 && K == fix(K), ...
    '标签必须是1到K的整数');

idx = yPred + K * (yTrue - 1);
G = reshape(accumarray(idx, 1, [K*K, 1]), K, K);

C = localHungarian(max(G(:)) - G);

ind = sub2ind([K, K], C(:), (1:K)');
acc = sum(G(ind)) / N;
end

function C = localHungarian(cost)
k = size(cost, 1);

p = zeros(1, k+1);
u = zeros(1, k+1);
v = zeros(1, k+1);

for i = 1:k
    p(1) = i;
    j0 = 1;

    minv = inf(1, k+1);
    used = false(1, k+1);
    way = zeros(1, k+1);

    while true
        used(j0) = true;
        i0 = p(j0);

        free = find(~used(2:end)) + 1;

        r = cost(i0, free-1) ...
            - u(i0+1) - v(free);

        improve = r < minv(free);
        minv(free(improve)) = r(improve);
        way(free(improve)) = j0;

        [delta, bestIdx] = min(minv(free));

        u(p(used)+1) = u(p(used)+1) + delta;
        v(used) = v(used) - delta;
        minv(free) = minv(free) - delta;

        j0 = free(bestIdx);

        if p(j0) == 0
            break;
        end
    end

    while j0 ~= 1
        j1 = way(j0);
        p(j0) = p(j1);
        j0 = j1;
    end
end

C = p(2:end);
end
