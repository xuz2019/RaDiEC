function [labels, info] = getFinalLabels(U, k)
    n = size(U, 1);
    validateattributes(k, {'numeric'}, {'scalar','integer','>=',1,'<=',n}, mfilename, 'k');
    if k == 1
        labels = ones(n, 1);
        info = struct('method', 'linkage', 'numClusters', 1);
        return;
    end
    tree = linkage(U, 'ward', 'euclidean');
    labels = cluster(tree, 'maxclust', k);
    labels = labels(:);
    info = struct('method', 'linkage', 'numClusters', numel(unique(labels)));
end