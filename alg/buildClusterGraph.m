function W = buildClusterGraph(B, sz, denom, n, c)
    overlap = B' * B;
    [ii, jj, intersection] = find(overlap);
    expected = sz(ii) .* sz(jj) / n;
    normalizer = denom(ii) .* denom(jj);
    sim = zeros(size(intersection));
    mask = normalizer > 0;
    sim(mask) = (intersection(mask) - expected(mask)) ./ normalizer(mask);

    keep = (ii ~= jj) & (sim > 0);
    W = sparse(ii(keep), jj(keep), sim(keep), c, c);
end
