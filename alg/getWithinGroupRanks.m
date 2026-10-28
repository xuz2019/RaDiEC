function rankVec = getWithinGroupRanks(weight, model)

    narginchk(2, 2);

    rankVec = zeros(model.c, 1);
    for mm = 1:model.M
        ids = model.groupStart(mm):model.groupEnd(mm);
        [~, order] = sort(weight(ids), 'descend');
        r = zeros(numel(ids), 1);
        r(order) = (1:numel(ids))';
        rankVec(ids) = r;
    end
end
