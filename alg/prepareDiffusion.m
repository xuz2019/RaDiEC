function diffusion = prepareDiffusion(W, degree)
    c = numel(degree);
    diffusion.degree = degree;
    if norm(W - W', 'fro') > 1e-10 * max(1, norm(W, 'fro'))
        error('The degree-based stationary distribution requires symmetric W.');
    end
    component = conncomp(graph(W))';
    componentCount = max(component);

    invSqrtDegree = 1 ./ sqrt(degree);
    S = spdiags(invSqrtDegree, 0, c, c) * W * spdiags(invSqrtDegree, 0, c, c);
    [Q, Lambda] = eig(full((S + S') / 2), 'vector');
    [lambda, order] = sort(real(Lambda));
    Q = real(Q(:, order));
    lambda(end-componentCount+1:end) = 0;
    if any(abs(lambda) >= 1)
        error('The centered random walk is not stable.');
    end

    residual = 1 ./ (1 - lambda.^2);
    maxDiagonal = max(residual);
    L = zeros(c, 0);
    idx = 0;
    while max(residual) > 1e-10 * maxDiagonal && size(L, 2) < c
        idx = idx + 1;
        [pivotResidual, j] = max(residual);
        column = 1 ./ (1 - lambda * lambda(j));
        if ~isempty(L)
            column = column - L * L(j, :)';
        end
        column = column / sqrt(pivotResidual);
        L(:, end+1) = column;
        residual = max(residual - column.^2, 0);
    end
    diffusion.Q = Q;
    diffusion.L = L;
    diffusion.sqrtDegree = sqrt(degree);
    diffusion.invSqrtDegree = invSqrtDegree;
end
