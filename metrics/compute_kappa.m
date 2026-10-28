function kappa = compute_kappa(result, gt)

    result = double(result(:));
    gt = double(gt(:));

    N = numel(gt);

    if numel(result) ~= N
        error('预测标签与真实标签长度必须一致');
    end

    if N == 0
        kappa = NaN;
        return;
    end

    if any(~isfinite(result)) || any(~isfinite(gt))
        error('标签不能包含 NaN 或 Inf');
    end

    minG = min(gt);
    minR = min(result);

    Kg = max(gt) - minG + 1;
    Kr = max(result) - minR + 1;

    maxCells = 5e6;

    fastMode = all(gt == fix(gt)) && ...
               all(result == fix(result)) && ...
               isfinite(Kg) && isfinite(Kr) && ...
               Kg >= 1 && Kr >= 1 && ...
               Kg * Kr <= maxCells;

    if fastMode

        idx = (gt - minG + 1) + ...
              Kg * (result - minR);

        counts = accumarray(idx, 1, ...
                            [Kg * Kr, 1]);

        M = reshape(counts, Kg, Kr);

        M = M(any(M,2), any(M,1));

    else

        [~,~,gi] = unique(gt);
        [~,~,ri] = unique(result);

        Kg = max(gi);
        Kr = max(ri);

        if double(Kg) * double(Kr) > maxCells
            error('类别数量过大，稠密最优匹配内存开销过高');
        end

        idx = gi + Kg * (ri - 1);

        counts = accumarray(idx, 1, ...
                            [Kg * Kr, 1]);

        M = reshape(counts, Kg, Kr);
    end

    pairs = matchpairs(M, -1, 'max');

    rowSum = sum(M, 2);
    colSum = sum(M, 1).';

    rows = pairs(:,1);
    cols = pairs(:,2);

    linearIdx = sub2ind(size(M), rows, cols);
    po = sum(M(linearIdx)) / N;

    pe = sum(rowSum(rows) .* colSum(cols)) / (N*N);

    if abs(1 - pe) < 1e-14
        kappa = NaN;
    else
        kappa = (po - pe) / (1 - pe);
    end
end
