function [labels, U, w, history] = RaDiEC(clusterings, k, opts)
    warning('off', 'all');
    if nargin < 3
        opts = struct();
    end
    opts.maxIter    = getOpt(opts, 'maxIter', 100);
    opts.tol        = getOpt(opts, 'tol', 1e-6);
    opts.stepU      = getOpt(opts, 'stepU', 0.5);
    opts.maxLS      = getOpt(opts, 'maxLS', 15);
    opts.warmup     = getOpt(opts, 'warmup', 10);
    opts.verbose    = getOpt(opts, 'verbose', true);
    opts.large      = getOpt(opts, 'large', false);

    [n, M] = size(clusterings);
    d = k - 1;

    [B, c, offset, numClusters] = buildIndicator(clusterings);

    sz = full(sum(B, 1))';
    offset = offset(:);
    groupStart = offset(:) + 1;
    groupEnd   = [offset(2:end); c];

    denom = sqrt(sz .* (1 - sz / n));
    invDenom = zeros(c, 1);
    valid = denom > 0;
    invDenom(valid) = 1 ./ denom(valid);

    W = buildClusterGraph(B, sz, denom, n, c);
    degree = full(sum(W, 2));
    if any(degree <= 0)
        error('Cluster graph contains zero-degree nodes.');
    end
    P = spdiags(1 ./ degree, 0, c, c) * W;

    ctx.B       = B;
    ctx.sz      = sz;
    ctx.invDenom = invDenom;
    ctx.P       = P;
    ctx.n       = n;
    ctx.c       = c;
    ctx.diffusion = prepareDiffusion(W, degree);

    model.c           = c;
    model.M           = M;
    model.numClusters = numClusters(:);
    model.groupStart  = groupStart;
    model.groupEnd    = groupEnd;

    w = zeros(c, 1);
    for m = 1:M
        ids = groupStart(m):groupEnd(m);

        w(ids) = 1;
    end

    U = randn(n, d);
    U = U - mean(U, 1);
    [U, ~] = qr(U, 0);

    for it = 1:opts.warmup
        [~, gradU, ~] = objectiveGrad(ctx, U, w, true);
        Q = gradU - mean(gradU, 1);
        [U, ~] = qr(Q, 0);
    end

    [~, ~, q] = objectiveGrad(ctx, U, w, true);
    w = rankWeightsFromScore(q, model);

    history.obj         = zeros(opts.maxIter + 1, 1);
    history.time        = zeros(opts.maxIter + 1, 1);
    history.rankChanges = zeros(opts.maxIter + 1, 1);
    history.q = cell(opts.maxIter + 1, 1);
    history.weights = cell(opts.maxIter + 1, 1);
    history.ranks = cell(opts.maxIter + 1, 1);

    timer = tic;
    [obj, ~, ~] = objectiveGrad(ctx, U, w, false);
    history.obj(1)     = obj;
    history.time(1)    = 0;
    prevRank = getWithinGroupRanks(w, model);
    history.q{1} = q;
    history.weights{1} = w;
    history.ranks{1} = prevRank;

    for it = 1:opts.maxIter
        oldObj = obj;

        [~, gradU, ~] = objectiveGrad(ctx, U, w, true);

        gradU = gradU - mean(gradU, 1);
        S = (U' * gradU + gradU' * U) / 2;
        gradU = gradU - U * S;

        step = opts.stepU;
        accepted = false;
        for ls = 1:opts.maxLS
            Unew = U + step * gradU;
            Unew = Unew - mean(Unew, 1);
            [Unew, ~] = qr(Unew, 0);

            [~, ~, qNew] = objectiveGrad(ctx, Unew, w, false);
            wNew = rankWeightsFromScore(qNew, model);
            objNew = objectiveGrad(ctx, Unew, wNew, false);

            if objNew >= obj - 1e-12
                U = Unew;
                w = wNew;
                q = qNew;
                obj = objNew;
                accepted = true;
                break;
            end
            step = step / 2;
        end

        currentRank = getWithinGroupRanks(w, model);
        history.rankChanges(it+1) = sum(currentRank ~= prevRank);
        prevRank = currentRank;
        history.q{it+1} = q;
        history.weights{it+1} = w;
        history.ranks{it+1} = currentRank;

        history.obj(it+1)  = obj;
        history.time(it+1) = toc(timer);

        if opts.verbose
            fprintf('Iter %3d | RobustObj %.8f | Delta %.3e | RankChanges %d | Accepted %d | Time %.2fs\n', ...
                it, obj, obj - oldObj, history.rankChanges(it+1), accepted, history.time(it+1));
        end

        smallImprovement = obj - oldObj <= opts.tol * (1 + abs(oldObj));
        stableRanks = history.rankChanges(it+1) == 0;
        if ~accepted || (smallImprovement && stableRanks)
            break;
        end
    end

    history.obj         = history.obj(1:it+1);
    history.time        = history.time(1:it+1);
    history.rankChanges = history.rankChanges(1:it+1);
    history.q = history.q(1:it+1);
    history.weights = history.weights(1:it+1);
    history.ranks = history.ranks(1:it+1);

    history.iterations  = it;
    history.totalTime   = toc(timer);
    
    if opts.large
        labels = litekmeans(U, k, 'MaxIter',100, 'Replicates',10);
    else
        [labels, ~] = getFinalLabels(U, k);
    end
end

function val = getOpt(opts, field, default)
    if isfield(opts, field)
        val = opts.(field);
    else
        val = default;
    end
end
