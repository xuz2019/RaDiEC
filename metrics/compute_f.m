function [f,p,r] = compute_f(H,T)

    H = double(H(:));
    T = double(T(:));

    N = numel(T);

    if numel(H) ~= N
        error('H和T长度必须相同');
    end

    if N == 0
        f = 1;
        p = 1;
        r = 1;
        return;
    end

    if any(~isfinite(H)) || any(~isfinite(T))
        error('标签不能包含NaN或Inf');
    end

    minH = min(H);
    minT = min(T);

    Kh = max(H) - minH + 1;
    Kt = max(T) - minT + 1;

    fastMode = all(H == fix(H)) && ...
               all(T == fix(T)) && ...
               isfinite(Kh) && isfinite(Kt) && ...
               Kh * Kt <= 1e6;

    if fastMode

        idx = (H - minH + 1) + Kh * (T - minT);

        counts = accumarray(idx, 1, [Kh*Kt, 1]);
        G = reshape(counts, Kh, Kt);

    else

        [~,~,hi] = unique(H);
        [~,~,ti] = unique(T);

        Kh = max(hi);
        Kt = max(ti);

        G = sparse(hi, ti, 1, Kh, Kt);

    end

    hCount = full(sum(G, 2));

    tCount = full(sum(G, 1));

    jointCount = nonzeros(G);

    numH = sum(hCount .* (hCount - 1)) / 2;

    numT = sum(tCount .* (tCount - 1)) / 2;

    numI = sum(jointCount .* (jointCount - 1)) / 2;

    p = 1;
    r = 1;

    if numH > 0
        p = numI / numH;
    end

    if numT > 0
        r = numI / numT;
    end

    if p + r == 0
        f = 0;
    else
        f = 2 * p * r / (p + r);
    end

end
