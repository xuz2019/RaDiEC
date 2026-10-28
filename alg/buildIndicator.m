function [B, c, offsets, numClusters] = buildIndicator(clusterings)

    [n, M] = size(clusterings);
    numClusters = max(clusterings, [], 1);
    offsets = [0, cumsum(numClusters(1:end-1))];
    globalLabels = clusterings + offsets;
    c = sum(numClusters);
    rowIdx = repmat((1:n)', M, 1);
    B = sparse(rowIdx, globalLabels(:), 1, n, c);
end