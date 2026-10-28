clc; clear; close all;
addpath(genpath(pwd));

M = 20;
cntTimes = 10;
poolSize = 100;

folder = fullfile(fileparts(mfilename('fullpath')), 'datasets');
files = dir(fullfile(folder, '*.mat'));

opts.maxIter = 10;
opts.tol = 1e-3;
opts.stepU = 0.1;
opts.warmup = 5;
opts.verbose = true;
opts.large = false;
% For very large datasets or when encountering out-of-memory errors, set
% opts.large = true


results = struct();

rng(2026, 'twister');
bcIdx = initialRandom(poolSize, M, cntTimes);

resultFolder = ['./results', num2str(M)];
if ~exist(resultFolder, 'dir')
    mkdir(resultFolder);
end

for d = 1:numel(files)
    [~, dataName, ~] = fileparts(files(d).name);
    fprintf('\n========== Dataset: %s ==========\n', dataName);
    data = load(fullfile(files(d).folder, files(d).name));

    gt = data.gt;
    members = data.members;
    k = length(unique(gt));
    measure1 = zeros(8, cntTimes);

    parfor idx = 1:cntTimes
        s = RandStream('Threefry', 'Seed', 2026);
        s.Substream = idx;
        RandStream.setGlobalStream(s);
        clusterings = members(:, bcIdx(idx, :));
        [labels, U, w, history] = RaDiEC(clusterings, k, opts);
        metrics1 = computeMetrics(labels, gt);
        measure1(:, idx) = metrics1(:);
    end

    mean_measure1 = mean(measure1, 2);
    std_measure1 = std(measure1, 0, 2);

    results(d).dataset = dataName;
    results(d).mean = mean_measure1;
    results(d).std = std_measure1;
    results(d).measure = measure1;

    fprintf('\nResults: %s\n', dataName);

    for j = 1:8
        fprintf('Metric %d: %.4f ± %.4f\n', j, mean_measure1(j), std_measure1(j));
    end

    result = results(d);
    save(fullfile(resultFolder, [dataName, '_M_', num2str(M), '_RaDiEC.mat']), 'result', 'opts', 'M', 'cntTimes', 'poolSize');

    save(fullfile(resultFolder, 'RaDiEC_results.mat'), 'results');
    fprintf('Saved: %s\n', dataName);
end

bigMatrix = read(resultFolder);
disp(mean(bigMatrix))
