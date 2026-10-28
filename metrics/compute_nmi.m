function nmi = compute_nmi(H, T)

    if ~(isnumeric(H) || islogical(H)) || ...
       ~(isnumeric(T) || islogical(T)) || ...
       ~isreal(H) || ~isreal(T)
        error('标签必须是实数或逻辑类型');
    end

    H = double(H(:));
    T = double(T(:));

    N = numel(T);

    if numel(H) ~= N
        error('H和T长度必须一致');
    end

    if N == 0
        nmi = NaN;
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

        C = sparse(hi, ti, ones(N,1), Kh, Kt);

    end

    a = full(sum(C, 2));

    b = full(sum(C, 1)).';

    [i,j,c] = find(C);

    mi = sum((c ./ N) .* ...
         log2((N .* c) ./ (a(i) .* b(j))));

    ph = a(a > 0) ./ N;
    entH = -sum(ph .* log2(ph));

    pt = b(b > 0) ./ N;
    entT = -sum(pt .* log2(pt));

    denominator = entH + entT;

    if denominator == 0

        nmi = NaN;
    else
        nmi = 2 * mi / denominator;

        nmi = max(0, min(1, nmi));
    end
end
