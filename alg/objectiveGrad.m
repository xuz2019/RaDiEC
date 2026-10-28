function [f, gU, q] = objectiveGrad(ctx, Ucur, wcur, needGrad)

    C = ctx.B' * Ucur;
    C = C - ctx.sz * (sum(Ucur, 1) / ctx.n);
    Cnorm = bsxfun(@times, C, ctx.invDenom);

    R0 = Cnorm';
    q = zeros(ctx.c, 1);
    D = ctx.diffusion;
    B = bsxfun(@times, R0, D.invSqrtDegree') * D.Q;
    if needGrad
        adjB = zeros(size(B));
    end
    for j = 1:size(D.L, 2)
        Zj = bsxfun(@times, B, D.L(:, j)') * D.Q';
        Zj = bsxfun(@times, Zj, D.sqrtDegree');
        q = q + sum(Zj.^2, 1)';
        if needGrad
            weighted = 2 * bsxfun(@times, Zj, wcur');
            weighted = bsxfun(@times, weighted, D.sqrtDegree');
            adjB = adjB + bsxfun(@times, weighted * D.Q, D.L(:, j)');
        end
    end
    if needGrad
        adj = bsxfun(@times, adjB * D.Q', D.invSqrtDegree');
    end

    f = q' * wcur;

    if ~needGrad
        gU = [];
        return;
    end

    R = bsxfun(@times, adj', ctx.invDenom);
    gU = ctx.B * R - ones(ctx.n,1) * ((ctx.sz' / ctx.n) * R);
end
