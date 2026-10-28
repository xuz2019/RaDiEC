clc; clear; close all;
projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot, 'alg'));
addpath(fullfile(projectRoot, 'metrics'));
addpath(fullfile(projectRoot, 'utils'));

dataName = 'zoo';
load(fullfile(projectRoot, 'datasets', [dataName, '.mat']));

M = 20;
cntTimes = 10;
poolSize = 100;
k = length(unique(gt));

opts.maxIter = 10;
opts.tol = 1e-3;
opts.stepU = 0.1;
opts.warmup = 5;
opts.verbose = true;
opts.large = false;
% For very large datasets or when encountering out-of-memory errors, set
% opts.large = true;

result = struct();
metricNames = {'NMI', 'ARI', 'F', 'Purity', 'ACC', 'Kappa', 'Precision', 'Recall'};

resultFolder = fullfile(projectRoot, 'results');
if ~exist(resultFolder, 'dir')
    mkdir(resultFolder);
end

rng(2026, 'twister');
bcIdx = initialRandom(poolSize, M, cntTimes);

measure1 = zeros(8, cntTimes);
runTime = zeros(1, cntTimes);
runHistory = cell(cntTimes, 1);
runWeights = cell(cntTimes, 1);
experimentTimer = tic;
% parfor idx = 1:cntTimes
for idx = 1:cntTimes
    s = RandStream('Threefry', 'Seed', 2026);
    s.Substream = idx;
    RandStream.setGlobalStream(s);
    clusterings = members(:, bcIdx(idx, :));
    runTimer = tic;
    [labels, U, w, history] = RaDiEC(clusterings, k, opts);
    runTime(idx) = toc(runTimer);
    metrics1 = computeMetrics(labels, gt);
    measure1(:, idx) = metrics1(:);
    runHistory{idx} = history;
    runWeights{idx} = w;
end
wallTime = toc(experimentTimer);

mean_measure1 = mean(measure1, 2);
var_measure1 = var(measure1, 0, 2);
std_measure1 = std(measure1, 0, 2);

result.dataset = dataName;
result.metricNames = metricNames;
result.mean = mean_measure1;
result.var = var_measure1;
result.std = std_measure1;
result.measure = measure1;
result.time = runTime;
result.meanTime = mean(runTime);
result.varTime = var(runTime, 0);
result.stdTime = std(runTime, 0);
result.wallTime = wallTime;

resultFile = fullfile(resultFolder, [dataName, '_RaDiEC.mat']);
save(resultFile, 'result', 'opts', 'M', 'cntTimes', 'poolSize', ...
    'bcIdx', 'runHistory', 'runWeights');

fprintf('\nResults: %s\n', dataName);
for j = 1:numel(metricNames)
    fprintf('%s: mean = %.4f, var = %.6f, std = %.4f\n', ...
        metricNames{j}, result.mean(j), result.var(j), result.std(j));
end
fprintf('Algorithm time (s): mean = %.4f, var = %.6f, std = %.4f\n', ...
    result.meanTime, result.varTime, result.stdTime);
fprintf('Parallel wall time (s): %.4f\n', result.wallTime);
fprintf('Saved: %s\n', resultFile);
