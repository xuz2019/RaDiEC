function bigMatrix = read(resultFolder)

files = dir(fullfile(resultFolder, '*_RaDiEC.mat'));

numDatasets = numel(files);
numMetrics = 8;
cntTimes = 10;

datasetNames = strings(numDatasets, 1);

meanMatrix = zeros(numDatasets, numMetrics);
stdMatrix = zeros(numDatasets, numMetrics);
runMatrix = zeros(numDatasets, numMetrics * cntTimes);

for d = 1:numDatasets

    data = load(fullfile(files(d).folder, files(d).name));

    result = data.result;

    datasetNames(d) = string(result.dataset);

    meanMatrix(d, :) = result.mean(:)';
    stdMatrix(d, :) = result.std(:)';

    runMatrix(d, :) = result.measure(:)';

end

bigMatrix = [meanMatrix, stdMatrix];

save(fullfile(resultFolder, 'merged_results.mat'), ...
    'datasetNames', 'meanMatrix', 'stdMatrix', ...
    'bigMatrix', 'runMatrix');

variableNames = [ ...
    "Mean" + string(1:numMetrics), ...
    "Std" + string(1:numMetrics)];

T = array2table(bigMatrix, ...
    'VariableNames', cellstr(variableNames));

T = addvars(T, datasetNames, ...
    'Before', 1, ...
    'NewVariableNames', 'Dataset');

writetable(T, fullfile(resultFolder, 'merged_results.csv'));

disp(T);

fprintf('\nMerged %d datasets.\n', numDatasets);

end
