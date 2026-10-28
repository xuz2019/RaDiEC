function score = compute_purity(clusters, labels)

if ~(isnumeric(clusters) || islogical(clusters)) || ...
        ~(isnumeric(labels) || islogical(labels)) || ...
        ~isreal(clusters) || ~isreal(labels)
    error('标签必须为实数或逻辑类型');
end

H = double(clusters(:));
T = double(labels(:));

N = numel(T);

if numel(H) ~= N
    error('clusters和labels长度必须一致');
end

if N == 0
    score = NaN;
    return;
end

if any(~isfinite(H)) || any(~isfinite(T))
    error('标签不能包含NaN或Inf');
end

minH = min(H);
minT = min(T);

Kh = max(H) - minH + 1;
Kt = max(T) - minT + 1;

fastMode = ...
    all(H == fix(H)) && ...
    all(T == fix(T)) && ...
    isfinite(Kh) && isfinite(Kt) && ...
    Kh >= 1 && Kt >= 1 && ...
    Kh * Kt <= 1e6;

if fastMode

    Kh = round(Kh);
    Kt = round(Kt);

    idx = (H - minH + 1) + ...
        Kh * (T - minT);

    counts = accumarray(idx, 1, ...
        [Kh * Kt, 1]);

    C = reshape(counts, Kh, Kt);

else

    [~,~,hi] = unique(H);
    [~,~,ti] = unique(T);

    Kh = max(hi);
    Kt = max(ti);

    C = sparse(hi, ti, 1, Kh, Kt);

end

maxCount = max(C, [], 2);

score = full(sum(maxCount)) / N;

end
