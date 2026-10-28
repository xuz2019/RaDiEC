function wout = rankWeightsFromScore(score, model)

    score = score(:);
    if numel(score) ~= model.c || any(~isfinite(score))
        error('rankWeightsFromScore:InvalidScore', ...
            'score must contain one finite value for each cluster.');
    end

    wout = zeros(model.c, 1);
    for m = 1:model.M
        ids = model.groupStart(m):model.groupEnd(m);
        km  = model.numClusters(m);

        if numel(ids) ~= km
            error('rankWeightsFromScore:InvalidGroup', ...
                'Cluster range and cluster count do not match for base clustering %d.', m);
        end

        invIdx = 1 ./ (1:km);
        profile = fliplr(cumsum(fliplr(invIdx)))';
        [~, order] = sort(score(ids), 'ascend');
        wout(ids(order)) = profile;
    end
end
