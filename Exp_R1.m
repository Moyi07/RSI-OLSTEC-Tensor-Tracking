%% Experiment R1
% =========================================================================
% Prerequisites: Run `run_me_first.m` once to add all dependencies to path.
%
% Objective:
%   Evaluate the convergence performance and late-window tracking error of
%   RSI-OLSTEC against baseline methods on real-world WAAM video data.
%
% Data Source:
%   - Video: WAAM process monitoring video (video_1.mp4).
%   - Metadata: Synchronized weld-bead width measurements (Width_mm).
% =========================================================================
clear; clc; close all;

%% 1. Configuration
% -------------------------------------------------------------------------
fprintf('Starting Experiment R1: Real-World Benchmark...\n');

% Data Paths
repo_root = fileparts(mfilename('fullpath'));
video_filename = fullfile(repo_root, 'dataset', 'video', ...
    '250312-110206-video_1.mp4');
meta_filename = fullfile(repo_root, 'dataset', 'WAMVID_metadata.csv');

% Experiment Parameters
num_trials      = 50;  % Number of Monte Carlo trials
max_frames      = 623;   % Truncate video length for efficiency
scale_ratio     = 0.2;   % Downsampling ratio (Speed/Accuracy trade-off)
rank_r          = 20;    % CP-Rank for tensor methods
fraction        = 0.1;   % Observation ratio (10% pixels observed)
tolcost         = 1e-8;  % Convergence tolerance
rsi_lambda_min = 0.70;
rsi_lambda_max = 0.80;
rsi_mu = 0.01;
rsi_grad_ema_alpha = 0.999;
rsi_irls_max_iters = 3;
rsi_irls_tolerance = 1e-3;
rsi_normalization_epsilon = 1e-3;
initial_calibration_frames = 30;
paired_ci_level = 0.95;
paired_ci_resamples = 10000;
paired_ci_seed = 20260728;
paired_ci_method = 'paired_percentile_bootstrap_linear_quantile';
export_results  = true;
result_dir      = fullfile(repo_root, 'result', 'R1');
resume_from_checkpoint = true;
checkpoint_file = fullfile(result_dir, 'R1_checkpoint.mat');
if export_results && ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

% Algorithm Switches
run_cpwopt      = true;
run_petrels     = true;
run_grasta      = true;
run_grouse      = true;
run_tecpsgd     = true;
run_olstec      = true;
run_rsi_olstec  = true;

%% 2. Data Loading & Preprocessing
% -------------------------------------------------------------------------
fprintf('=== [Phase 1] Data Preprocessing ===\n');

% Check File Existence
if ~exist(video_filename, 'file')
    error('Video file not found: %s', video_filename);
end

% A. Load Video Tensor
v = VideoReader(video_filename);
Tensor_Y_Original = [];
frame_idx = 0;
fprintf('Reading video frames...\n');

while hasFrame(v) && frame_idx < max_frames
    frame_idx = frame_idx + 1;
    raw_frame = readFrame(v);

    % Convert to grayscale and resize
    if size(raw_frame, 3) == 3
        gray_frame = rgb2gray(raw_frame);
    else
        gray_frame = raw_frame;
    end
    img_resized = imresize(gray_frame, scale_ratio);

    % Initialize storage on first frame
    if isempty(Tensor_Y_Original)
        Tensor_Y_Original = zeros(size(img_resized,1), size(img_resized,2), max_frames);
    end

    % Normalize to [0,1]
    Tensor_Y_Original(:, :, frame_idx) = im2double(img_resized); %#ok<SAGROW>
end

% Trim unused pre-allocated frames
if frame_idx < max_frames
    Tensor_Y_Original = Tensor_Y_Original(:,:,1:frame_idx);
end
[rows, cols, total_slices] = size(Tensor_Y_Original);
tensor_dims = [rows, cols, total_slices];
fprintf('Data Loaded: %d x %d x %d (Frames: %d)\n', rows, cols, total_slices, total_slices);

% B. Load Auxiliary Signal (Side Information)
fprintf('Loading Metadata...\n');
[aux_width, aux_meta] = load_waam_width_signal(meta_filename, video_filename, total_slices, 'trim_leading_nan');
fprintf('Metadata loaded successfully from row %d.\n', aux_meta.row_idx);
if aux_meta.num_trimmed_front > 0
    Tensor_Y_Original = Tensor_Y_Original(:, :, aux_meta.trim_start_frame:end);
    [rows, cols, total_slices] = size(Tensor_Y_Original);
    tensor_dims = [rows, cols, total_slices];
    fprintf('Trimmed first %d frames with unavailable Width_mm. Effective data: %d x %d x %d.\n', ...
        aux_meta.num_trimmed_front, rows, cols, total_slices);
end
if numel(aux_width) ~= total_slices
    error('Exp_R1:AuxLengthMismatch', ...
        'Auxiliary signal length (%d) does not match effective frame count (%d).', numel(aux_width), total_slices);
end
if total_slices <= initial_calibration_frames
    error('Exp_R1:InsufficientCalibrationData', ...
        ['The sequence must contain more than %d frames so that the ' ...
         'initial calibration interval is followed by evaluation data.'], ...
        initial_calibration_frames);
end

% Estimate the side-information threshold from the initial calibration interval.
diff_aux = diff(aux_width(1:initial_calibration_frames));
mad_aux = median(abs(diff_aux - median(diff_aux, 'omitnan')), 'omitnan');
est_aux_sigma = (1.4826 * mad_aux) / sqrt(2);
adaptive_min_grad = max(0.05, 3 * sqrt(2) * est_aux_sigma);
fprintf(['Initial-calibration min_grad_threshold estimated from %d ' ...
    'frames: %.4f\n'], initial_calibration_frames, adaptive_min_grad);

% Matrix baselines use the same target rank as the tensor methods.
numr = rows * cols;
numc = total_slices;
matrix_rank = rank_r;
Matrix_Y_Original = reshape(Tensor_Y_Original, [numr, numc]);
fprintf('Matrix baseline rank set to %d.\n', matrix_rank);

late_window_length = 50;
late_window_start = max(1, total_slices - late_window_length + 1);
late_window = late_window_start:total_slices;

%% 3. Monte Carlo Loop
% -------------------------------------------------------------------------
fprintf('\n=== [Phase 2] Monte Carlo Simulation (%d Trials) ===\n', num_trials);

% OLSTEC Lambda grid
lambda_list = [0.70, 0.80, 0.90, 0.99];

% Storage Initialization
stats = struct();
alg_list = {};
if run_cpwopt,  alg_list{end+1} = 'CP_WOPT'; end
if run_petrels, alg_list{end+1} = 'Petrels'; end
if run_grasta,  alg_list{end+1} = 'Grasta'; end
if run_grouse,  alg_list{end+1} = 'Grouse'; end
if run_tecpsgd, alg_list{end+1} = 'TeCPSGD'; end

% Generate OLSTEC variants
if run_olstec
    for lam = lambda_list
        alg_list{end+1} = sprintf('OLSTEC_%02d', round(lam*100)); %#ok<SAGROW>
    end
end
if run_rsi_olstec, alg_list{end+1} = 'RSI_OLSTEC'; end

alg_list_sorted = alg_list(~strcmp(alg_list, 'RSI_OLSTEC'));
if any(strcmp(alg_list, 'RSI_OLSTEC'))
    alg_list_sorted{end+1} = 'RSI_OLSTEC';
end

for i = 1:length(alg_list)
    stats.(alg_list{i}) = NaN(num_trials, total_slices);
end
trial_seeds = (1:num_trials)' + 1000;
matrix_init_seeds = 40000 + (1:num_trials)';
completed_trials = false(num_trials, 1);
trial_elapsed_seconds = NaN(num_trials, 1);
huber_delta_by_trial = NaN(num_trials, 1);
failure_counts = struct();
failure_messages = struct();
for i = 1:length(alg_list)
    failure_counts.(alg_list{i}) = 0;
    failure_messages.(alg_list{i}) = {};
end

checkpoint_signature = struct();
checkpoint_signature.video_filename = video_filename;
checkpoint_signature.meta_filename = meta_filename;
checkpoint_signature.tensor_dims = tensor_dims;
checkpoint_signature.num_trials = num_trials;
checkpoint_signature.max_frames = max_frames;
checkpoint_signature.scale_ratio = scale_ratio;
checkpoint_signature.rank_r = rank_r;
checkpoint_signature.matrix_rank = matrix_rank;
checkpoint_signature.fraction = fraction;
checkpoint_signature.tolcost = tolcost;
checkpoint_signature.lambda_list = lambda_list;
checkpoint_signature.adaptive_min_grad = adaptive_min_grad;
checkpoint_signature.rsi_lambda_min = rsi_lambda_min;
checkpoint_signature.rsi_lambda_max = rsi_lambda_max;
checkpoint_signature.rsi_mu = rsi_mu;
checkpoint_signature.rsi_grad_ema_alpha = rsi_grad_ema_alpha;
checkpoint_signature.rsi_irls_max_iters = rsi_irls_max_iters;
checkpoint_signature.rsi_irls_tolerance = rsi_irls_tolerance;
checkpoint_signature.rsi_normalization_epsilon = rsi_normalization_epsilon;
checkpoint_signature.initial_calibration_frames = ...
    initial_calibration_frames;
checkpoint_signature.paired_ci_level = paired_ci_level;
checkpoint_signature.paired_ci_resamples = paired_ci_resamples;
checkpoint_signature.paired_ci_seed = paired_ci_seed;
checkpoint_signature.paired_ci_method = paired_ci_method;
checkpoint_signature.alg_list = alg_list;
checkpoint_signature.trial_seeds = trial_seeds;
checkpoint_signature.matrix_init_seeds = matrix_init_seeds;
checkpoint_signature.algorithm_switches = [run_cpwopt, run_petrels, run_grasta, ...
    run_grouse, run_tecpsgd, run_olstec, run_rsi_olstec];

if resume_from_checkpoint && exist(checkpoint_file, 'file')
    loaded_checkpoint = load(checkpoint_file, 'checkpoint');
    if ~isfield(loaded_checkpoint, 'checkpoint')
        error('Exp_R1:InvalidCheckpoint', ...
            'Checkpoint file does not contain the expected checkpoint structure.');
    end
    checkpoint = loaded_checkpoint.checkpoint;
    required_fields = {'signature', 'stats', 'completed_trials', ...
        'trial_elapsed_seconds', 'trial_seeds', ...
        'huber_delta_by_trial', 'failure_counts', 'failure_messages'};
    if ~all(isfield(checkpoint, required_fields))
        error('Exp_R1:InvalidCheckpoint', ...
            'Checkpoint file is incomplete and cannot be resumed safely.');
    end
    if ~isequaln(checkpoint.signature, checkpoint_signature)
        error('Exp_R1:CheckpointConfigurationMismatch', ...
            ['The checkpoint configuration does not match the current experiment. ', ...
             'Remove the checkpoint or restore the matching configuration.']);
    end
    if numel(checkpoint.completed_trials) ~= num_trials
        error('Exp_R1:InvalidCheckpoint', ...
            'Checkpoint trial count does not match the current experiment.');
    end
    for i = 1:numel(alg_list)
        name = alg_list{i};
        if ~isfield(checkpoint.stats, name) || ...
                ~isequal(size(checkpoint.stats.(name)), [num_trials, total_slices])
            error('Exp_R1:InvalidCheckpoint', ...
                'Checkpoint statistics for %s have incompatible dimensions.', name);
        end
    end

    stats = checkpoint.stats;
    completed_trials = checkpoint.completed_trials;
    trial_elapsed_seconds = checkpoint.trial_elapsed_seconds;
    trial_seeds = checkpoint.trial_seeds;
    huber_delta_by_trial = checkpoint.huber_delta_by_trial;
    failure_counts = checkpoint.failure_counts;
    failure_messages = checkpoint.failure_messages;
    fprintf('Resuming R1 from checkpoint: %d/%d trials already completed.\n', ...
        nnz(completed_trials), num_trials);
    clear loaded_checkpoint checkpoint;
end

total_start_time = tic;

for trial = 1:num_trials
    if completed_trials(trial)
        fprintf('Skipping completed Trial %d / %d.\n', trial, num_trials);
        continue;
    end

    iter_timer = tic;
    fprintf('Processing Trial %d / %d... \n', trial, num_trials);

    % 3.1. Generate Random Missing Mask
    rng(trial_seeds(trial), 'twister'); % Fixed per-trial seed
    OmegaTensor = rand(rows, cols, total_slices) < fraction;
    GammaTensor = [];

    % Estimate the Huber threshold from the initial calibration interval.
    diff_pixels = cell(initial_calibration_frames - 1, 1);
    for t_idx = 2:initial_calibration_frames
        common_mask = OmegaTensor(:,:,t_idx) & OmegaTensor(:,:,t_idx-1);
        diff_frame = Tensor_Y_Original(:,:,t_idx) - Tensor_Y_Original(:,:,t_idx-1);
        diff_pixels{t_idx-1} = diff_frame(common_mask);
    end
    diff_pixels = vertcat(diff_pixels{:});
    if ~isempty(diff_pixels)
        mad_val = median(abs(diff_pixels - median(diff_pixels, 'omitnan')), 'omitnan');
        est_sigma = (1.4826 * mad_val) / sqrt(2);
        trial_huber_delta = max(0.10, min(0.15, 6 * est_sigma));
    else
        trial_huber_delta = 0.15;
    end
    huber_delta_by_trial(trial) = trial_huber_delta;

    % Reshape for Matrix Algorithms
    OmegaMatrix = reshape(OmegaTensor, [numr, numc]);
    GammaMatrix = [];

    % 3.2. Random Initialization
    Xinit.A = randn(rows, rank_r);
    Xinit.B = randn(cols, rank_r);
    Xinit.C = randn(total_slices, rank_r);
    matrix_init_stream = RandStream('mt19937ar', ...
        'Seed', matrix_init_seeds(trial));
    matrix_init = struct();
    matrix_init.U = randn(matrix_init_stream, numr, matrix_rank);
    matrix_init.Weight = randn(matrix_init_stream, matrix_rank, numc);

    % --- Execute Algorithms ---
    % 0. CP-WOPT (Batch Tensor Baseline)
    if run_cpwopt
        try
            opts = struct('maxepochs', 30, 'display_iters', 1, 'verbose', 0, ...
                          'tolcost', tolcost, 'store_matrix', false, 'store_subinfo', true);
            [~, ~, sub_info] = cp_wopt_mod(Tensor_Y_Original, OmegaTensor, GammaTensor, tensor_dims, rank_r, Xinit, opts);

            raw_err = get_info_metric(sub_info);
            stats.CP_WOPT(trial, :) = normalize_r1_metric( ...
                raw_err, 'CP_WOPT', total_slices, trial);
        catch ME
            if is_process_interruption(ME), rethrow(ME); end
            fprintf('Trial %d: CP-WOPT failed (%s)\n', trial, ME.message);
            failure_counts.CP_WOPT = failure_counts.CP_WOPT + 1;
            failure_messages.CP_WOPT{end+1} = format_failure_record( ...
                trial, trial_seeds(trial), ME);
            stats.CP_WOPT(trial, :) = NaN(1, total_slices);
        end
    end

    % 1. Petrels (Matrix Baseline)
    if run_petrels
        try
            opts = struct('maxepochs', 1, 'rank', matrix_rank, 'lambda', 0.98, 'verbose', 0, ...
                          'tolcost', tolcost, 'permute_on', false, 'store_matrix', false, 'store_subinfo', true);
            [~, ~, sub_info, ~] = petrels_mod(matrix_init, Matrix_Y_Original, ...
                OmegaMatrix, GammaMatrix, numr, numc, opts);
            raw_err = get_info_metric(sub_info);
            stats.Petrels(trial, :) = normalize_r1_metric( ...
                raw_err, 'Petrels', total_slices, trial);
        catch ME
            if is_process_interruption(ME), rethrow(ME); end
            fprintf('Trial %d: Petrels failed (%s)\n', trial, ME.message);
            failure_counts.Petrels = failure_counts.Petrels + 1;
            failure_messages.Petrels{end+1} = format_failure_record( ...
                trial, trial_seeds(trial), ME);
            stats.Petrels(trial, :) = NaN(1, total_slices);
        end
    end

    % 2. GRASTA (Robust Matrix Baseline)
    if run_grasta
        try
            opts = struct('maxepochs', 1, 'RANK', matrix_rank, 'rho', 1.8, 'ITER_MAX', 20, ...
                          'MAX_MU', 10000, 'MIN_MU', 1, 'DIM_M', numr, 'USE_MEX', 0, ...
                          'verbose', 0, 'tolcost', tolcost, 'permute_on', false, 'store_matrix', false, 'store_subinfo', true);
            % Reproduce GRASTA's internal orthonormal basis from the shared matrix seed.
            rng(matrix_init_seeds(trial), 'twister');
            [~, ~, sub_info, ~] = grasta_mod(matrix_init, Matrix_Y_Original, ...
                OmegaMatrix, GammaMatrix, numr, numc, opts);
            raw_err = get_info_metric(sub_info);
            stats.Grasta(trial, :) = normalize_r1_metric( ...
                raw_err, 'Grasta', total_slices, trial);
        catch ME
            if is_process_interruption(ME), rethrow(ME); end
            fprintf('Trial %d: GRASTA failed (%s)\n', trial, ME.message);
            failure_counts.Grasta = failure_counts.Grasta + 1;
            failure_messages.Grasta{end+1} = format_failure_record( ...
                trial, trial_seeds(trial), ME);
            stats.Grasta(trial, :) = NaN(1, total_slices);
        end
    end

    % 3. Grouse (SGD Matrix Baseline)
    if run_grouse
        try
            opts = struct('maxepochs', 1, 'maxrank', matrix_rank, 'step_size', 0.0001, ...
                          'verbose', 0, 'tolcost', tolcost, 'permute_on', false, 'store_matrix', false, 'store_subinfo', true);
            [~, ~, sub_info, ~] = grouse_mod(matrix_init, Matrix_Y_Original, ...
                OmegaMatrix, GammaMatrix, numr, numc, opts);
            raw_err = get_info_metric(sub_info);
            stats.Grouse(trial, :) = normalize_r1_metric( ...
                raw_err, 'Grouse', total_slices, trial);
        catch ME
            if is_process_interruption(ME), rethrow(ME); end
            fprintf('Trial %d: Grouse failed (%s)\n', trial, ME.message);
            failure_counts.Grouse = failure_counts.Grouse + 1;
            failure_messages.Grouse{end+1} = format_failure_record( ...
                trial, trial_seeds(trial), ME);
            stats.Grouse(trial, :) = NaN(1, total_slices);
        end
    end

    % 4. TeCPSGD (Tensor SGD Baseline)
    if run_tecpsgd
        try
            opts = struct('maxepochs', 1, 'lambda', 0.99, 'stepsize', 0.1, 'mu', 0.01, ...
                          'verbose', 0, 'tolcost', tolcost, 'permute_on', false, 'store_matrix', false, 'store_subinfo', true);
            [~, ~, sub_info] = TeCPSGD(Tensor_Y_Original, OmegaTensor, GammaTensor, tensor_dims, rank_r, Xinit, opts);
            raw_err = get_info_metric(sub_info);
            stats.TeCPSGD(trial, :) = normalize_r1_metric( ...
                raw_err, 'TeCPSGD', total_slices, trial);
        catch ME
            if is_process_interruption(ME), rethrow(ME); end
            fprintf('Trial %d: TeCPSGD failed (%s)\n', trial, ME.message);
            failure_counts.TeCPSGD = failure_counts.TeCPSGD + 1;
            failure_messages.TeCPSGD{end+1} = format_failure_record( ...
                trial, trial_seeds(trial), ME);
            stats.TeCPSGD(trial, :) = NaN(1, total_slices);
        end
    end

    % 5. OLSTEC (Standard Tensor Baseline)
    if run_olstec
        for k = 1:length(lambda_list)
            lam = lambda_list(k);
            alg_name = sprintf('OLSTEC_%02d', round(lam*100));

            try
                opts = struct('maxepochs', 1, 'lambda', lam, 'mu', 0.01, 'verbose', 0, ...
                              'tw_flag', 0, 'tw_len', 10, 'tolcost', tolcost, 'permute_on', false, 'store_matrix', false, 'store_subinfo', true);
                [~, ~, sub_info] = olstec(Tensor_Y_Original, OmegaTensor, GammaTensor, tensor_dims, rank_r, Xinit, opts);
                raw_err = get_info_metric(sub_info);
                stats.(alg_name)(trial, :) = normalize_r1_metric( ...
                    raw_err, alg_name, total_slices, trial);
            catch ME
                if is_process_interruption(ME), rethrow(ME); end
                fprintf('Trial %d: %s failed (%s)\n', trial, alg_name, ME.message);
                failure_counts.(alg_name) = failure_counts.(alg_name) + 1;
                failure_messages.(alg_name){end+1} = format_failure_record( ...
                    trial, trial_seeds(trial), ME);
                stats.(alg_name)(trial, :) = NaN(1, total_slices);
            end
        end
    end

    % 6. RSI-OLSTEC (Proposed)
    if run_rsi_olstec
        try
            opts = struct('maxepochs', 1, ...
                          'lambda_max', rsi_lambda_max, ...
                          'lambda_min', rsi_lambda_min, ...
                          'huber_delta', trial_huber_delta, ...
                          'min_grad_threshold', adaptive_min_grad, ... % Initial-calibration MAD estimate from the measured auxiliary signal.
                          'grad_ema_alpha', rsi_grad_ema_alpha, ...
                          'mu', rsi_mu, ...
                          'irls_max_iters', rsi_irls_max_iters, ...
                          'irls_tolerance', rsi_irls_tolerance, ...
                          'normalization_epsilon', rsi_normalization_epsilon, ...
                          'verbose', 0, 'tolcost', tolcost, 'permute_on', false, 'store_matrix', false, 'store_subinfo', true);
            [~, ~, sub_info] = rsi_olstec(Tensor_Y_Original, OmegaTensor, GammaTensor, tensor_dims, rank_r, Xinit, opts, aux_width);
            raw_err = get_info_metric(sub_info);
            stats.RSI_OLSTEC(trial, :) = normalize_r1_metric( ...
                raw_err, 'RSI_OLSTEC', total_slices, trial);
        catch ME
            if is_process_interruption(ME), rethrow(ME); end
            fprintf('Trial %d: RSI-OLSTEC failed (%s)\n', trial, ME.message);
            failure_counts.RSI_OLSTEC = failure_counts.RSI_OLSTEC + 1;
            failure_messages.RSI_OLSTEC{end+1} = format_failure_record( ...
                trial, trial_seeds(trial), ME);
            stats.RSI_OLSTEC(trial, :) = NaN(1, total_slices);
        end
    end

    clear OmegaTensor OmegaMatrix GammaTensor GammaMatrix Xinit matrix_init ...
        matrix_init_stream sub_info raw_err opts diff_pixels diff_frame common_mask;

    trial_elapsed_seconds(trial) = toc(iter_timer);
    completed_trials(trial) = true;

    if export_results
        checkpoint = struct();
        checkpoint.signature = checkpoint_signature;
        checkpoint.stats = stats;
        checkpoint.completed_trials = completed_trials;
        checkpoint.trial_elapsed_seconds = trial_elapsed_seconds;
        checkpoint.trial_seeds = trial_seeds;
        checkpoint.huber_delta_by_trial = huber_delta_by_trial;
        checkpoint.failure_counts = failure_counts;
        checkpoint.failure_messages = failure_messages;
        save_checkpoint_atomic(checkpoint_file, checkpoint);
        clear checkpoint;
    end

    fprintf('Done (%.1fs)\n', trial_elapsed_seconds(trial));
end
invocation_elapsed_seconds = toc(total_start_time);
total_trial_elapsed_seconds = sum(trial_elapsed_seconds(isfinite(trial_elapsed_seconds)));
fprintf('\nSimulation Completed in %.1f minutes (current invocation).\n', ...
    invocation_elapsed_seconds / 60);

paired_comparison_table = build_paired_nre_comparison( ...
    stats, alg_list_sorted, 'RSI_OLSTEC', late_window, completed_trials, ...
    failure_counts, paired_ci_level, paired_ci_resamples, paired_ci_seed, ...
    paired_ci_method);

if export_results
    save(fullfile(result_dir, 'R1_stats.mat'), ...
        'num_trials', 'max_frames', 'scale_ratio', 'rank_r', ...
        'matrix_rank', 'fraction', ...
        'tolcost', 'lambda_list', 'adaptive_min_grad', 'aux_meta', ...
        'rsi_lambda_min', 'rsi_lambda_max', 'rsi_mu', ...
        'rsi_grad_ema_alpha', 'rsi_irls_max_iters', ...
        'rsi_irls_tolerance', 'rsi_normalization_epsilon', ...
        'initial_calibration_frames', ...
        'paired_ci_level', 'paired_ci_resamples', 'paired_ci_seed', ...
        'paired_ci_method', 'paired_comparison_table', ...
        'stats', 'alg_list', 'alg_list_sorted', 'failure_counts', ...
        'failure_messages', 'trial_seeds', 'matrix_init_seeds', ...
        'completed_trials', ...
        'trial_elapsed_seconds', 'huber_delta_by_trial', ...
        'invocation_elapsed_seconds', ...
        'total_trial_elapsed_seconds', 'late_window', ...
        'checkpoint_signature', '-v7');
    write_r1_result_tables(result_dir, stats, alg_list_sorted, trial_seeds, ...
        matrix_init_seeds, completed_trials, trial_elapsed_seconds, ...
        huber_delta_by_trial, late_window, failure_counts, failure_messages, ...
        paired_comparison_table);
    fprintf('Saved raw R1 statistics before visualization.\n');
end

%% 4. Statistical Analysis & Visualization
% -------------------------------------------------------------------------
fprintf('=== [Phase 3] Generating Plots ===\n');

% Color Palette & Line Styles
cols_map = containers.Map();
line_styles = containers.Map();

if run_cpwopt,  cols_map('CP_WOPT') = [0 0 0]; line_styles('CP_WOPT') = '-'; end
if run_petrels, cols_map('Petrels') = [0.6 0.6 0.6]; line_styles('Petrels') = '-'; end
if run_grasta,  cols_map('Grasta')  = [0.929 0.694 0.125]; line_styles('Grasta') = '-'; end
if run_grouse,  cols_map('Grouse')  = [0.466 0.674 0.188]; line_styles('Grouse') = '-'; end
if run_tecpsgd, cols_map('TeCPSGD') = [0.000 0.447 0.741]; line_styles('TeCPSGD') = '-'; end
if run_rsi_olstec, cols_map('RSI_OLSTEC') = [0.850 0.325 0.098]; line_styles('RSI_OLSTEC') = '-'; end

% Assign Cyan color and different line styles for OLSTEC variants
if run_olstec
    olstec_styles = {':', '--', '-', '-.'};
    for k = 1:length(lambda_list)
        alg_name = sprintf('OLSTEC_%02d', round(lambda_list(k)*100));
        cols_map(alg_name) = [0.301 0.745 0.933];
        line_styles(alg_name) = olstec_styles{k};
    end
end

% Font Settings
linewidth_normal = 1.5;
font_name = 'Times New Roman';
font_size = 14;

% --- Figure 1: Convergence Time-Series ---
fig1 = figure('Name', 'Exp R1: Real-World Benchmark', 'Position', [100, 100, 800, 550], 'Color', 'w');
hold on; grid on; box on;
ax = gca;
ax.FontName = font_name; ax.FontSize = font_size;
ax.LineWidth = 1.2; ax.TickLabelInterpreter = 'latex';
set(gca, 'YScale', 'log');

x_axis = 1:total_slices;
legend_handles = [];
legend_names = {};

global_min = inf;
global_max = -inf;

for i = 1:length(alg_list_sorted)
    name = alg_list_sorted{i};
    data = stats.(name);

    valid = isfinite(data);
    count = sum(valid, 1);
    data_zero = data;
    data_zero(~valid) = 0;
    mu = sum(data_zero, 1) ./ count;
    mu(count == 0) = NaN;
    dev = data - mu;
    dev(~valid) = 0;
    variance = sum(dev.^2, 1);
    sigma = NaN(size(variance));
    sigma(count > 1) = sqrt(variance(count > 1) ./ (count(count > 1) - 1));
    conf_interval = 1.96 * sigma ./ sqrt(count);
    conf_interval(count == 0) = NaN;

    col = cols_map(name);
    l_style = line_styles(name);

    % Calculate global minimum based on valid mean values
    valid_mu = mu(mu > 0);
    if ~isempty(valid_mu)
        global_min = min(global_min, min(valid_mu));
    end

    % Calculate global maximum (excluding initial transient errors)
    half_idx = max(1, floor(total_slices / 2));
    global_candidates = mu(half_idx:end) + conf_interval(half_idx:end);
    global_candidates = global_candidates(isfinite(global_candidates));
    if ~isempty(global_candidates)
        global_max = max(global_max, max(global_candidates));
    end

    % Ensure lower bound is strictly positive for log-scale plotting
    lower_bound = mu - conf_interval;
    lower_bound(lower_bound <= 0) = mu(lower_bound <= 0) * 0.1;

    % Plot Error Band
    fill([x_axis, fliplr(x_axis)], [mu+conf_interval, fliplr(lower_bound)],...
         col, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    % Plot Mean Line
    h = plot(x_axis, mu, 'Color', col, 'LineWidth', linewidth_normal, 'LineStyle', l_style);
    legend_handles(end+1) = h; %#ok<SAGROW>

    % Legend formatting
    if startsWith(name, 'OLSTEC_')
        lam_val = str2double(name(8:end))/100;
        legend_names{end+1} = sprintf('OLSTEC ($\\lambda=%.2f$)', lam_val); %#ok<SAGROW>
    elseif strcmp(name, 'RSI_OLSTEC')
        legend_names{end+1} = '\textbf{RSI-OLSTEC}'; %#ok<SAGROW>
    elseif strcmp(name, 'CP_WOPT')
        legend_names{end+1} = 'CP-WOPT (Batch, offline)'; %#ok<SAGROW>
    else
        legend_names{end+1} = strrep(name, '_', '-'); %#ok<SAGROW>
    end
end

xlabel('Time Index (Frames)', 'Interpreter', 'latex', 'FontSize', font_size+2);
ylabel('Full-frame Normalized Residual Error (log)', 'Interpreter', 'latex', 'FontSize', font_size+2);
title('Full-frame Evaluation (CP-WOPT is Offline Batch)', 'Interpreter', 'latex', 'FontSize', font_size+2);

lgd = legend(legend_handles, legend_names, 'Location', 'southwest');
lgd.Interpreter = 'latex'; lgd.FontSize = font_size - 2; lgd.Box = 'off';

% Dynamic Y-axis and strict X-axis limits
if global_min > 0 && global_max > global_min
    ylim([global_min * 0.5, global_max * 1.5]);
end
xlim([1, total_slices]);

% --- Figure 1: Inset Plot (Time Series) ---
ax1_pos = get(ax, 'Position');
inset_ax = axes('Position', [ax1_pos(1) + ax1_pos(3)*0.55, ax1_pos(2) + ax1_pos(4)*0.50, ax1_pos(3)*0.40, ax1_pos(4)*0.40]);
box on; grid on; hold on;
set(inset_ax, 'YScale', 'log', 'FontName', font_name, 'FontSize', 10, 'TickLabelInterpreter', 'latex');

zoom_start = max(1, total_slices - 100);
zoom_end = total_slices;
zoom_x = zoom_start:zoom_end;

% Select algorithms for inset plot
zoom_algs = {};
if run_petrels, zoom_algs{end+1} = 'Petrels'; end
if run_olstec
    for k = 1:length(lambda_list)
        zoom_algs{end+1} = sprintf('OLSTEC_%02d', round(lambda_list(k)*100)); %#ok<SAGROW>
    end
end
if run_rsi_olstec, zoom_algs{end+1} = 'RSI_OLSTEC'; end

zoom_min = inf;
zoom_max = -inf;

for i = 1:length(zoom_algs)
    name = zoom_algs{i};
    if isfield(stats, name)
        mu = col_nanmean(stats.(name)(:, zoom_x));
        plot(inset_ax, zoom_x, mu, 'Color', cols_map(name), 'LineWidth', 1.5, 'LineStyle', line_styles(name));
        finite_mu = mu(isfinite(mu) & mu > 0);
        if ~isempty(finite_mu)
            zoom_min = min(zoom_min, min(finite_mu));
            zoom_max = max(zoom_max, max(finite_mu));
        end
    end
end
xlim(inset_ax, [zoom_start, zoom_end]);
if zoom_min > 0 && zoom_max > zoom_min
    ylim(inset_ax, [zoom_min * 0.9, zoom_max * 1.1]);
end
title(inset_ax, 'Zoom-in (Late Window)', ...
    'Interpreter', 'latex', 'FontSize', 11);

% =========================================================================
% --- Figure 2: Late-Window Accuracy Distribution (Boxplot) ---
% =========================================================================
fig2 = figure('Name', 'Exp R1: Late-Window Accuracy Boxplot', ...
    'Position', [150, 150, 800, 550], 'Color', 'w');
hold on; grid on; box on;

late_window_errors = [];
group_labels = {};
colors_for_boxplot = [];

for i = 1:length(alg_list_sorted)
    name = alg_list_sorted{i};
    % Average the last 50 frames for the late-window error.
    late_window_values = row_nanmean(stats.(name)(:, late_window));

    late_window_errors = [late_window_errors, late_window_values]; %#ok<AGROW>

    colors_for_boxplot = [colors_for_boxplot; cols_map(name)]; %#ok<AGROW>

    if startsWith(name, 'OLSTEC_')
        lam_val = str2double(name(8:end))/100;
        group_labels{end+1} = sprintf('OLSTEC ($\\lambda=%.2f$)', lam_val); %#ok<SAGROW>
    elseif strcmp(name, 'RSI_OLSTEC')
        group_labels{end+1} = '\textbf{RSI-OLSTEC}'; %#ok<SAGROW>
    else
        group_labels{end+1} = strrep(name, '_', '-'); %#ok<SAGROW>
    end
end

% Plot Boxplot
hBox = boxplot(late_window_errors, 'Labels', group_labels, ...
    'Symbol', 'o', 'Widths', 0.6);
set(gca, 'TickLabelInterpreter', 'tex');

% Style Outliers and Boxes
hOutliers = findobj(gca,'Tag','Outliers');
set(hOutliers, 'MarkerEdgeColor', [0.5 0.5 0.5], 'MarkerSize', 4);
set(hBox, {'LineWidth'}, {1.2});

boxes = findobj(gca, 'Tag', 'Box');
if length(boxes) == length(alg_list_sorted)
    for j = 1:length(boxes)
        idx = length(boxes) - j + 1;
        patch(get(boxes(j),'XData'), get(boxes(j),'YData'), colors_for_boxplot(idx,:), ...
            'FaceAlpha', 0.5, 'EdgeColor', colors_for_boxplot(idx,:), 'LineWidth', 1.5);
    end
end

% Axis formatting
ax2 = gca;
ax2.FontName = font_name; ax2.FontSize = font_size;
ax2.LineWidth = 1.2; ax2.TickLabelInterpreter = 'latex';
xlabel('Algorithm', 'Interpreter', 'latex', 'FontSize', font_size+2);
ylabel('Late-window mean full-frame NRE (log)', ...
    'Interpreter', 'latex', 'FontSize', font_size+2);
title('Statistical Accuracy Distribution', 'Interpreter', 'latex', 'FontSize', font_size+2);
xtickangle(30);
ax2.XMinorTick = 'off'; ax2.YMinorTick = 'off';
set(gca, 'YScale', 'log');

% Apply truncated Y-axis range
finite_box_vals = late_window_errors( ...
    isfinite(late_window_errors) & late_window_errors > 0);
if ~isempty(finite_box_vals)
    box_min = min(finite_box_vals);
    box_max_all = max(finite_box_vals);
    ylim([box_min * 0.5, box_max_all * 1.5]);
end

% --- Figure 2: Inset Plot (Boxplot) ---
ax2_pos = get(ax2, 'Position');
inset_ax2 = axes('Position', [ax2_pos(1) + ax2_pos(3)*0.55, ax2_pos(2) + ax2_pos(4)*0.55, ax2_pos(3)*0.40, ax2_pos(4)*0.30]);
box on; grid on; hold on;
set(inset_ax2, 'YScale', 'log', 'FontName', font_name, 'FontSize', 10, 'TickLabelInterpreter', 'latex');

zoom_algs_box = {};
if run_olstec
    for k = 1:length(lambda_list)
        zoom_algs_box{end+1} = sprintf('OLSTEC_%02d', round(lambda_list(k)*100)); %#ok<SAGROW>
    end
end
if run_rsi_olstec, zoom_algs_box{end+1} = 'RSI_OLSTEC'; end

zoom_data_box = [];
zoom_labels_box = {};
zoom_colors_box = [];

for i = 1:length(zoom_algs_box)
    name = zoom_algs_box{i};
    if isfield(stats, name)
        late_window_values = row_nanmean(stats.(name)(:, late_window));
        zoom_data_box = [zoom_data_box, late_window_values]; %#ok<AGROW>
        zoom_colors_box = [zoom_colors_box; cols_map(name)]; %#ok<AGROW>

        if startsWith(name, 'OLSTEC_')
            lam_val = str2double(name(8:end))/100;
            zoom_labels_box{end+1} = sprintf('OLSTEC(%.2f)', lam_val); %#ok<SAGROW>
        elseif strcmp(name, 'RSI_OLSTEC')
            zoom_labels_box{end+1} = '\bf RSI-OLSTEC'; %#ok<SAGROW>
        end
    end
end

if ~isempty(zoom_data_box)
    axes(inset_ax2);
    hBox_inset = boxplot(zoom_data_box, 'Labels', zoom_labels_box, 'Symbol', 'o', 'Widths', 0.5);
    set(inset_ax2, 'TickLabelInterpreter', 'tex');
    xtickangle(inset_ax2, 30);

    hOutliers_inset = findobj(inset_ax2, 'Tag', 'Outliers');
    set(hOutliers_inset, 'MarkerEdgeColor', [0.5 0.5 0.5], 'MarkerSize', 3);
    set(hBox_inset, {'LineWidth'}, {1.2});

    boxes_inset = findobj(inset_ax2, 'Tag', 'Box');
    if length(boxes_inset) == length(zoom_algs_box)
        for j = 1:length(boxes_inset)
            idx = length(boxes_inset) - j + 1;
            patch(get(boxes_inset(j),'XData'), get(boxes_inset(j),'YData'), zoom_colors_box(idx,:), ...
                'FaceAlpha', 0.5, 'EdgeColor', zoom_colors_box(idx,:), 'LineWidth', 1.5, 'Parent', inset_ax2);
        end
    end

    finite_zoom_vals = zoom_data_box(isfinite(zoom_data_box) & zoom_data_box > 0);
    if ~isempty(finite_zoom_vals)
        y_min_inset = min(finite_zoom_vals);
        y_max_inset = max(finite_zoom_vals);
        ylim(inset_ax2, [y_min_inset * 0.95, y_max_inset * 1.05]);
    end
    title(inset_ax2, 'OLSTEC vs Ours', 'Interpreter', 'latex', 'FontSize', 11);
end

%% 5. Quantitative Results Output
% -------------------------------------------------------------------------
fprintf('\n==========================================================================================\n');
fprintf('QUANTITATIVE RESULTS COMPARISON (Monte Carlo: %d Trials)\n', num_trials);
fprintf('==========================================================================================\n');
fprintf('%-25s | %-55s \n', 'Algorithm', ...
    'Late-window mean full-frame NRE (Mean +/- Std, valid/fail)');
fprintf('------------------------------------------------------------------------------------------\n');

for i = 1:length(alg_list_sorted)
    name = alg_list_sorted{i};
    if isfield(stats, name)
        late_window_values = row_nanmean(stats.(name)(:, late_window));

        valid_late_window_values = late_window_values( ...
            isfinite(late_window_values));
        mean_err = mean(valid_late_window_values);
        std_err = std(valid_late_window_values);

        if startsWith(name, 'OLSTEC_')
            lam_val = str2double(name(8:end))/100;
            display_name = sprintf('OLSTEC (lam=%.2f)', lam_val);
        elseif strcmp(name, 'RSI_OLSTEC')
            display_name = 'RSI-OLSTEC (Ours)';
        elseif strcmp(name, 'CP_WOPT')
            display_name = 'CP-WOPT (Batch, offline)';
        else
            display_name = strrep(name, '_', '-');
        end

        fprintf('%-25s | %.6e +/- %.6e (valid=%d/%d, fail=%d)\n', ...
            display_name, mean_err, std_err, ...
            numel(valid_late_window_values), num_trials, ...
            failure_counts.(name));
    end
end
fprintf('------------------------------------------------------------------------------------------\n');
if export_results
    savefig(fig1, fullfile(result_dir, 'Fig_R1_Convergence.fig'));
    savefig(fig2, fullfile(result_dir, 'Fig_R1_Boxplot.fig'));
    fprintf('Saved: Fig_R1_Convergence.fig, Fig_R1_Boxplot.fig\n');
end
fprintf('Experiment R1 Completed.\n');

function metric = normalize_r1_metric(raw_metric, algorithm, expected_length, trial)
    if ~(isnumeric(raw_metric) && isreal(raw_metric) && isvector(raw_metric))
        error('Exp_R1:InvalidMetricLayout', ...
            '%s returned a non-real or non-vector metric output in trial %d.', ...
            algorithm, trial);
    end
    raw_metric = reshape(raw_metric, 1, []);
    has_initial_placeholder = strcmp(algorithm, 'CP_WOPT') || ...
        strcmp(algorithm, 'TeCPSGD') || ...
        startsWith(algorithm, 'OLSTEC_') || ...
        strcmp(algorithm, 'RSI_OLSTEC');

    if has_initial_placeholder
        if numel(raw_metric) ~= expected_length + 1
            error('Exp_R1:InvalidMetricLayout', ...
                ['%s returned %d metric values in trial %d; its documented ' ...
                 'layout requires one initial placeholder followed by %d frame values.'], ...
                algorithm, numel(raw_metric), trial, expected_length);
        end
        if ~isnan(raw_metric(1))
            error('Exp_R1:InvalidMetricLayout', ...
                '%s did not return the expected initial NaN placeholder in trial %d.', ...
                algorithm, trial);
        end
        metric = raw_metric(2:end);
    else
        if numel(raw_metric) ~= expected_length
            error('Exp_R1:InvalidMetricLayout', ...
                '%s returned %d metric values in trial %d; %d values were expected.', ...
                algorithm, numel(raw_metric), trial, expected_length);
        end
        metric = raw_metric;
    end
    metric = validate_complete_nre(metric, expected_length, algorithm, trial);
end

function metric = get_info_metric(sub_info)
    if isfield(sub_info, 'err_residual') && ~isempty(sub_info.err_residual)
        metric = sub_info.err_residual;
    else
        error('Exp_R1:MissingResidualMetric', ...
            'Expected err_residual for full-frame NRE evaluation.');
    end
end

function y = col_nanmean(X)
    valid = isfinite(X);
    count = sum(valid, 1);
    X(~valid) = 0;
    y = sum(X, 1) ./ count;
    y(count == 0) = NaN;
end

function y = row_nanmean(X)
    valid = isfinite(X);
    count = sum(valid, 2);
    X(~valid) = 0;
    y = sum(X, 2) ./ count;
    y(count == 0) = NaN;
end

function write_r1_result_tables(result_dir, stats, alg_list, trial_seeds, ...
    matrix_init_seeds, completed_trials, trial_elapsed_seconds, ...
    huber_delta_by_trial, late_window, failure_counts, failure_messages, ...
    paired_comparison_table)

    num_trials = numel(trial_seeds);
    trial_table = table((1:num_trials)', trial_seeds(:), ...
        matrix_init_seeds(:), completed_trials(:), ...
        trial_elapsed_seconds(:), huber_delta_by_trial(:), ...
        'VariableNames', {'Trial', 'TrialSeed', ...
        'MatrixInitializationSeed', 'Completed', 'ElapsedSeconds', ...
        'HuberDelta'});

    num_algorithms = numel(alg_list);
    summary_mean = NaN(num_algorithms, 1);
    summary_std = NaN(num_algorithms, 1);
    valid_trials = zeros(num_algorithms, 1);
    failed_trials = zeros(num_algorithms, 1);

    for i = 1:num_algorithms
        name = alg_list{i};
        late_window_values = row_nanmean(stats.(name)(:, late_window));
        trial_table.(name) = late_window_values;

        if failure_counts.(name) ~= numel(failure_messages.(name))
            error('Exp_R1:FailureAuditMismatch', ...
                'Failure count and failure records disagree for %s.', name);
        end

        valid = isfinite(late_window_values);
        valid_trials(i) = nnz(valid);
        failed_trials(i) = failure_counts.(name);
        if valid_trials(i) > 0
            summary_mean(i) = mean(late_window_values(valid));
            summary_std(i) = std(late_window_values(valid));
        end
    end

    summary_table = table(alg_list(:), summary_mean, summary_std, ...
        valid_trials, failed_trials, repmat(num_trials, num_algorithms, 1), ...
        'VariableNames', {'Algorithm', 'MeanLateWindowNRE', ...
        'StdLateWindowNRE', ...
        'ValidTrials', 'RecordedFailures', 'TotalTrials'});

    failure_algorithm = cell(0, 1);
    failure_trial = zeros(0, 1);
    failure_seed = zeros(0, 1);
    failure_category = cell(0, 1);
    failure_identifier = cell(0, 1);
    failure_message = cell(0, 1);
    failure_record = cell(0, 1);
    for i = 1:num_algorithms
        name = alg_list{i};
        records = failure_messages.(name);
        for j = 1:numel(records)
            [record_trial, record_seed, record_category, ...
                record_identifier, record_message] = ...
                parse_failure_record(records{j});
            failure_algorithm{end+1, 1} = name; %#ok<AGROW>
            failure_trial(end+1, 1) = record_trial; %#ok<AGROW>
            failure_seed(end+1, 1) = record_seed; %#ok<AGROW>
            failure_category{end+1, 1} = record_category; %#ok<AGROW>
            failure_identifier{end+1, 1} = record_identifier; %#ok<AGROW>
            failure_message{end+1, 1} = record_message; %#ok<AGROW>
            failure_record{end+1, 1} = records{j}; %#ok<AGROW>
        end
    end
    failure_table = table(failure_algorithm, failure_trial, failure_seed, ...
        failure_category, failure_identifier, failure_message, failure_record, ...
        'VariableNames', {'Algorithm', 'Trial', 'Seed', 'Category', ...
        'Identifier', 'Message', 'FailureRecord'});

    summary_algorithm = cell(0, 1);
    summary_category = cell(0, 1);
    summary_count = zeros(0, 1);
    for i = 1:num_algorithms
        name = alg_list{i};
        algorithm_mask = strcmp(failure_algorithm, name);
        categories = unique(failure_category(algorithm_mask));
        for j = 1:numel(categories)
            category = categories{j};
            summary_algorithm{end+1, 1} = name; %#ok<AGROW>
            summary_category{end+1, 1} = category; %#ok<AGROW>
            summary_count(end+1, 1) = nnz(algorithm_mask & ...
                strcmp(failure_category, category)); %#ok<AGROW>
        end
    end
    failure_summary_table = table(summary_algorithm, summary_category, ...
        summary_count, 'VariableNames', {'Algorithm', 'Category', 'Count'});

    writetable(trial_table, ...
        fullfile(result_dir, 'R1_per_trial_late_window_nre.csv'));
    writetable(summary_table, fullfile(result_dir, 'R1_summary_statistics.csv'));
    writetable(failure_table, fullfile(result_dir, 'R1_failure_log.csv'));
    writetable(failure_summary_table, ...
        fullfile(result_dir, 'R1_failure_summary.csv'));
    writetable(paired_comparison_table, ...
        fullfile(result_dir, 'R1_paired_difference_ci.csv'));
end

function comparison_table = build_paired_nre_comparison( ...
    stats, algorithm_names, reference_name, late_window, completed_trials, ...
    failure_counts, ci_level, num_resamples, bootstrap_seed, ci_method)

    if ~(isscalar(ci_level) && isfinite(ci_level) && ...
            ci_level > 0 && ci_level < 1)
        error('Exp_R1:InvalidCILevel', ...
            'paired_ci_level must be a finite scalar in the interval (0, 1).');
    end
    if ~(isscalar(num_resamples) && isfinite(num_resamples) && ...
            num_resamples >= 1 && num_resamples == floor(num_resamples))
        error('Exp_R1:InvalidBootstrapCount', ...
            'paired_ci_resamples must be a positive integer.');
    end
    if ~(isscalar(bootstrap_seed) && isfinite(bootstrap_seed) && ...
            bootstrap_seed >= 0 && bootstrap_seed == floor(bootstrap_seed))
        error('Exp_R1:InvalidBootstrapSeed', ...
            'paired_ci_seed must be a nonnegative integer.');
    end

    baseline_names = algorithm_names(~strcmp(algorithm_names, reference_name));
    num_comparisons = numel(baseline_names);
    reference_algorithm = repmat({reference_name}, num_comparisons, 1);
    total_trials = repmat(numel(completed_trials), num_comparisons, 1);
    baseline_valid_trials = zeros(num_comparisons, 1);
    reference_valid_trials = zeros(num_comparisons, 1);
    valid_pairs = zeros(num_comparisons, 1);
    baseline_failures = zeros(num_comparisons, 1);
    reference_failures = zeros(num_comparisons, 1);
    mean_baseline_nre = NaN(num_comparisons, 1);
    mean_reference_nre = NaN(num_comparisons, 1);
    mean_paired_difference = NaN(num_comparisons, 1);
    median_paired_difference = NaN(num_comparisons, 1);
    ci_lower = NaN(num_comparisons, 1);
    ci_upper = NaN(num_comparisons, 1);
    reference_lower_nre_fraction = NaN(num_comparisons, 1);
    ci_level_column = repmat(ci_level, num_comparisons, 1);
    bootstrap_resamples = repmat(num_resamples, num_comparisons, 1);
    bootstrap_seeds = (bootstrap_seed:(bootstrap_seed + ...
        num_comparisons - 1))';
    ci_methods = repmat({ci_method}, num_comparisons, 1);

    if isfield(stats, reference_name)
        reference_values = row_nanmean( ...
            stats.(reference_name)(:, late_window));
        completed = logical(completed_trials(:));
        reference_valid = completed & isfinite(reference_values);
        reference_valid_count = nnz(reference_valid);
        if isfield(failure_counts, reference_name)
            reference_failure_count = failure_counts.(reference_name);
        else
            reference_failure_count = 0;
        end

        for i = 1:num_comparisons
            name = baseline_names{i};
            baseline_values = row_nanmean(stats.(name)(:, late_window));
            baseline_valid = completed & isfinite(baseline_values);
            paired_valid = baseline_valid & reference_valid;
            differences = baseline_values(paired_valid) - ...
                reference_values(paired_valid);

            baseline_valid_trials(i) = nnz(baseline_valid);
            reference_valid_trials(i) = reference_valid_count;
            valid_pairs(i) = nnz(paired_valid);
            if isfield(failure_counts, name)
                baseline_failures(i) = failure_counts.(name);
            end
            reference_failures(i) = reference_failure_count;

            if valid_pairs(i) > 0
                mean_baseline_nre(i) = mean(baseline_values(paired_valid));
                mean_reference_nre(i) = mean(reference_values(paired_valid));
                mean_paired_difference(i) = mean(differences);
                median_paired_difference(i) = median(differences);
                reference_lower_nre_fraction(i) = mean(differences > 0);
            end
            [ci_lower(i), ci_upper(i)] = paired_bootstrap_mean_ci( ...
                differences, num_resamples, ci_level, bootstrap_seeds(i));
        end
    end

    comparison_table = table(baseline_names(:), reference_algorithm, ...
        total_trials, baseline_valid_trials, reference_valid_trials, ...
        valid_pairs, baseline_failures, reference_failures, ...
        mean_baseline_nre, mean_reference_nre, mean_paired_difference, ...
        median_paired_difference, ci_lower, ci_upper, ...
        reference_lower_nre_fraction, ci_level_column, ...
        bootstrap_resamples, bootstrap_seeds, ci_methods, ...
        'VariableNames', {'BaselineAlgorithm', 'ReferenceAlgorithm', ...
        'TotalTrials', 'BaselineValidTrials', 'ReferenceValidTrials', ...
        'ValidPairs', 'BaselineFailures', 'ReferenceFailures', ...
        'MeanBaselineNRE', 'MeanReferenceNRE', ...
        'MeanBaselineMinusReferenceNRE', ...
        'MedianBaselineMinusReferenceNRE', ...
        'BaselineMinusReferenceCILower', ...
        'BaselineMinusReferenceCIUpper', ...
        'ReferenceLowerNREFraction', 'CILevel', 'BootstrapResamples', ...
        'BootstrapSeed', 'CIMethod'});
end

function [ci_lower, ci_upper] = paired_bootstrap_mean_ci( ...
    differences, num_resamples, ci_level, bootstrap_seed)

    differences = differences(:);
    if numel(differences) < 2
        ci_lower = NaN;
        ci_upper = NaN;
        return;
    end

    stream = RandStream('mt19937ar', 'Seed', bootstrap_seed);
    sample_indices = randi(stream, numel(differences), ...
        numel(differences), num_resamples);
    bootstrap_means = mean(differences(sample_indices), 1);
    tail_probability = (1 - ci_level) / 2;
    ci_lower = linear_quantile(bootstrap_means, tail_probability);
    ci_upper = linear_quantile(bootstrap_means, 1 - tail_probability);
end

function value = linear_quantile(samples, probability)
    sorted_samples = sort(samples(:));
    position = 1 + (numel(sorted_samples) - 1) * probability;
    lower_index = floor(position);
    upper_index = ceil(position);
    if lower_index == upper_index
        value = sorted_samples(lower_index);
    else
        fraction = position - lower_index;
        value = sorted_samples(lower_index) + fraction * ...
            (sorted_samples(upper_index) - sorted_samples(lower_index));
    end
end

function record = format_failure_record(trial, seed, exception)
    identifier = exception.identifier;
    if isempty(identifier)
        identifier = 'unidentified_exception';
    end
    category = classify_failure(identifier, exception.message);
    message_text = regexprep(strtrim(exception.message), '[\r\n]+', ' | ');
    record = sprintf( ...
        'trial=%d; seed=%d; category=%s; identifier=%s; message=%s', ...
        trial, seed, category, identifier, message_text);
end

function tf = is_process_interruption(exception)
    identifier = lower(exception.identifier);
    message_text = lower(exception.message);
    tf = contains(identifier, 'operationterminatedbyuser') || ...
        contains(identifier, 'nomem') || ...
        contains(identifier, 'outofmemory') || ...
        contains(message_text, 'operation terminated by user') || ...
        contains(message_text, 'out of memory') || ...
        contains(message_text, 'memory allocation') || ...
        contains(message_text, 'requested array exceeds') || ...
        contains(message_text, 'unable to allocate');
end

function category = classify_failure(identifier, message_text)
    diagnostic_text = lower([identifier, ' ', message_text]);
    if contains(diagnostic_text, 'validate_complete_nre') || ...
            contains(diagnostic_text, 'invalidmetriclayout') || ...
            contains(diagnostic_text, 'missingresidualmetric') || ...
            contains(diagnostic_text, 'nonfinitemetric') || ...
            contains(diagnostic_text, 'non-finite') || ...
            contains(diagnostic_text, 'emptymetric')
        category = 'invalid_metric';
    elseif contains(diagnostic_text, 'nomem') || ...
            contains(diagnostic_text, 'out of memory') || ...
            contains(diagnostic_text, 'memory allocation')
        category = 'memory_or_resource';
    elseif contains(diagnostic_text, 'operationterminatedbyuser') || ...
            contains(diagnostic_text, 'terminated by user')
        category = 'user_interruption';
    elseif contains(diagnostic_text, 'singular') || ...
            contains(diagnostic_text, 'rank deficient') || ...
            contains(diagnostic_text, 'ill-conditioned') || ...
            contains(diagnostic_text, 'positive definite')
        category = 'numerical_linear_algebra';
    else
        category = 'runtime_exception';
    end
end

function [trial, seed, category, identifier, message_text] = ...
    parse_failure_record(record)

    tokens = regexp(record, ...
        ['^trial=(\d+); seed=(\d+); category=([^;]+); ' ...
         'identifier=([^;]+); message=(.*)$'], ...
        'tokens', 'once');
    if isempty(tokens)
        trial = NaN;
        seed = NaN;
        category = 'unparsed_record';
        identifier = '';
        message_text = record;
        return;
    end

    trial = str2double(tokens{1});
    seed = str2double(tokens{2});
    category = tokens{3};
    identifier = tokens{4};
    message_text = tokens{5};
end
