%% Experiment R3
% =========================================================================
% Prerequisites: Run `run_me_first.m` once to add all dependencies to path.
%
% Objective:
%   Evaluate RSI-OLSTEC and the baseline algorithms on one real WAAM video
%   under paired random observation masks and injected saturated spatter.
%
% Statistical scope:
%   The Monte Carlo trials quantify variability due to observation masks,
%   synthetic spatter, and random initialization for this video sequence.
%   They do not constitute independent-video or cross-view validation.
% =========================================================================
clear;
clc;
close all;

fprintf('Starting Experiment R3: paired Monte Carlo spatter stress test...\n');

%% 1. Configuration
% -------------------------------------------------------------------------
repo_root = fileparts(mfilename('fullpath'));

% Data paths
video_filename = fullfile(repo_root, 'dataset', 'video', ...
    '250312-110206-video_1.mp4');
meta_filename = fullfile(repo_root, 'dataset', 'WAMVID_metadata.csv');

% Experiment settings
num_trials = 50;
observation_ratios = [0.70, 0.50, 0.30, 0.10];
nominal_spatter_density = 0.01;
spatter_saturation_value = 1.0;
rank_r = 20;
max_frames = 623;
scale_ratio = 0.2;
final_window_length = 50;
initial_calibration_frames = 30;
huber_scale_multiplier = 6;
huber_delta_lower_bound = 0.10;
huber_delta_upper_bound = 0.15;
side_threshold_sigma_multiplier = 3;
side_threshold_lower_bound = 0.05;

% Prespecified statistical analysis
metric_names = {'PostCalibrationMeanReferenceNRE', ...
    'FinalFrameReferenceNRE', 'LateWindowMeanReferenceNRE'};
primary_metric = 'PostCalibrationMeanReferenceNRE';
paired_ci_level = 0.95;
paired_ci_resamples = 10000;
summary_ci_seed = 20260729;
paired_ci_seed = 20261729;
ci_method = 'percentile_bootstrap_linear_quantile';

% Prespecified implementation-level computational benchmark
timing_observation_ratio = 0.10;
timing_method_names = {'OLSTEC_80', 'RSI_OLSTEC'};
timing_warmup_runs_per_method = 1;
timing_ci_level = 0.95;
timing_ci_resamples = 10000;
timing_bootstrap_seed = 20262729;
timing_ci_method = ...
    'percentile_bootstrap_paired_geometric_mean_time_ratio';

% Prespecified visualization, not selected from observed performance
representative_trial = 1;
representative_rho = 0.10;
representative_frames_requested = [50, 200, 400, 600];
visual_algorithm_names = {'Grouse', 'TeCPSGD', 'Grasta', 'Petrels', ...
    'OLSTEC_80', 'RSI_OLSTEC'};
figure_visibility = 'off';
if strcmp(getenv('RSI_IMAGE_DISPLAY'), '1')
    figure_visibility = 'on';
end

% Algorithm switches
run_cpwopt = true;
run_petrels = true;
run_grasta = true;
run_grouse = true;
run_tecpsgd = true;
run_olstec = true;
run_rsi_olstec = true;

% Common algorithm settings
maxepochs = 1;
tolcost = 1e-8;
permute_on_flag = false;
verbose = 0;
lambda_list = [0.70, 0.80, 0.90, 0.99];

% Explicit RSI-OLSTEC settings
rsi_lambda_min = 0.70;
rsi_lambda_max = 0.80;
rsi_mu = 0.01;
rsi_grad_ema_alpha = 0.999;
rsi_irls_max_iters = 3;
rsi_irls_tolerance = 1e-3;
rsi_normalization_epsilon = 1e-3;

% Output and checkpoint settings
export_results = true;
resume_from_checkpoint = true;
result_dir = fullfile(repo_root, 'result', 'R3');
checkpoint_file = fullfile(result_dir, 'R3_checkpoint.mat');
if export_results && ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

%% 2. Data Loading and Causal Preprocessing
% -------------------------------------------------------------------------
if ~exist(video_filename, 'file')
    error('Exp_R3:VideoNotFound', 'Video file not found: %s', video_filename);
end
if ~exist(meta_filename, 'file')
    error('Exp_R3:MetadataNotFound', ...
        'Metadata file not found: %s', meta_filename);
end

fprintf('Reading video: %s\n', video_filename);
video_reader = VideoReader(video_filename);
Tensor_Y_Original = [];
frame_idx = 0;

while hasFrame(video_reader) && frame_idx < max_frames
    frame_idx = frame_idx + 1;
    raw_frame = readFrame(video_reader);
    if size(raw_frame, 3) == 3
        gray_frame = rgb2gray(raw_frame);
    else
        gray_frame = raw_frame;
    end
    resized_frame = imresize(gray_frame, scale_ratio);
    if isempty(Tensor_Y_Original)
        Tensor_Y_Original = zeros(size(resized_frame, 1), ...
            size(resized_frame, 2), max_frames);
    end
    Tensor_Y_Original(:, :, frame_idx) = ...
        im2double(resized_frame); %#ok<SAGROW>
end

if isempty(Tensor_Y_Original)
    error('Exp_R3:NoFramesRead', ...
        'No frames were read from video: %s', video_filename);
end
if frame_idx < max_frames
    Tensor_Y_Original = Tensor_Y_Original(:, :, 1:frame_idx);
end

[rows, cols, total_slices] = size(Tensor_Y_Original);
fprintf('Loaded video tensor: %d x %d x %d.\n', rows, cols, total_slices);

fprintf('Loading and causally aligning side information...\n');
[aux_width, aux_meta] = load_waam_width_signal(meta_filename, ...
    video_filename, total_slices, 'trim_leading_nan');
if aux_meta.num_trimmed_front > 0
    Tensor_Y_Original = Tensor_Y_Original(:, :, ...
        aux_meta.trim_start_frame:end);
    [rows, cols, total_slices] = size(Tensor_Y_Original);
    fprintf('Trimmed %d leading frames without valid width metadata.\n', ...
        aux_meta.num_trimmed_front);
end
if numel(aux_width) ~= total_slices
    error('Exp_R3:AuxLengthMismatch', ...
        'Auxiliary signal length (%d) does not match frame count (%d).', ...
        numel(aux_width), total_slices);
end
if any(~isfinite(aux_width))
    error('Exp_R3:NonfiniteAuxiliarySignal', ...
        'The causally aligned auxiliary signal contains non-finite values.');
end
if any(~isfinite(Tensor_Y_Original(:)))
    error('Exp_R3:NonfiniteVideoData', ...
        'The preprocessed video contains non-finite values.');
end

Tensor_Y_PreInjectionReference = Tensor_Y_Original;
tensor_dims = [rows, cols, total_slices];
numr = rows * cols;
numc = total_slices;
matrix_rank = rank_r;
if initial_calibration_frames < 2 || ...
        initial_calibration_frames ~= floor(initial_calibration_frames) || ...
        initial_calibration_frames >= total_slices
    error('Exp_R3:InvalidCalibrationLength', ...
        ['initial_calibration_frames must be an integer between 2 and ' ...
         'the number of frames minus one.']);
end
evaluation_start_frame = initial_calibration_frames + 1;
evaluation_frames = evaluation_start_frame:total_slices;
final_window_start = max(1, total_slices - final_window_length + 1);
final_window = final_window_start:total_slices;
representative_frames = representative_frames_requested( ...
    representative_frames_requested <= total_slices);
if isempty(representative_frames)
    error('Exp_R3:NoRepresentativeFrames', ...
        'No prespecified representative frame lies within the video.');
end

representative_rho_index = find( ...
    abs(observation_ratios - representative_rho) < 1e-12, 1);
if isempty(representative_rho_index)
    error('Exp_R3:RepresentativeRatioMissing', ...
        'representative_rho must be included in observation_ratios.');
end

timing_rho_index = find( ...
    abs(observation_ratios - timing_observation_ratio) < 1e-12, 1);
if isempty(timing_rho_index)
    error('Exp_R3:TimingRatioMissing', ...
        'timing_observation_ratio must be included in observation_ratios.');
end

% Estimate the side-information threshold once from the initial interval.
diff_aux = diff(aux_width(1:initial_calibration_frames));
mad_aux = median(abs(diff_aux - median(diff_aux, 'omitnan')), 'omitnan');
est_aux_sigma = (1.4826 * mad_aux) / sqrt(2);
adaptive_min_grad = max(side_threshold_lower_bound, ...
    side_threshold_sigma_multiplier * sqrt(2) * est_aux_sigma);
fprintf(['Side-information threshold estimated from the initial %d-frame ' ...
    'calibration interval: %.6f.\n'], initial_calibration_frames, ...
    adaptive_min_grad);

%% 3. Paired Randomization and Run State
% -------------------------------------------------------------------------
algorithm_names = {};
if run_cpwopt, algorithm_names{end+1} = 'CP_WOPT'; end
if run_petrels, algorithm_names{end+1} = 'Petrels'; end
if run_grasta, algorithm_names{end+1} = 'Grasta'; end
if run_grouse, algorithm_names{end+1} = 'Grouse'; end
if run_tecpsgd, algorithm_names{end+1} = 'TeCPSGD'; end
if run_olstec
    for lambda = lambda_list
        algorithm_names{end+1} = sprintf( ...
            'OLSTEC_%02d', round(100 * lambda)); %#ok<SAGROW>
    end
end
if run_rsi_olstec, algorithm_names{end+1} = 'RSI_OLSTEC'; end

num_algorithms = numel(algorithm_names);
num_ratios = numel(observation_ratios);
num_timing_methods = numel(timing_method_names);
if num_algorithms == 0
    error('Exp_R3:NoAlgorithmsEnabled', ...
        'At least one algorithm must be enabled.');
end
if ~isequal(timing_method_names, {'OLSTEC_80', 'RSI_OLSTEC'})
    error('Exp_R3:InvalidTimingMethods', ...
        'The computational benchmark must compare OLSTEC_80 and RSI_OLSTEC.');
end

trial_ids = (1:num_trials)';
mask_seeds = 10000 + trial_ids;
spatter_seeds = 20000 + trial_ids;
tensor_init_seeds = 30000 + trial_ids;
matrix_init_seeds = 40000 + trial_ids;
execution_seeds = zeros(num_trials, num_algorithms);
for trial = 1:num_trials
    for algorithm_idx = 1:num_algorithms
        execution_seeds(trial, algorithm_idx) = ...
            500000 + 100 * trial + algorithm_idx;
    end
end
% Match GRASTA's internal orthonormal initialization to matrix_init.U.
if run_grasta
    grasta_idx = strcmp(algorithm_names, 'Grasta');
    execution_seeds(:, grasta_idx) = matrix_init_seeds;
end

seed_ledger_table = table(trial_ids, mask_seeds, spatter_seeds, ...
    tensor_init_seeds, matrix_init_seeds, ...
    'VariableNames', {'Trial', 'MaskSeed', 'SpatterSeed', ...
    'TensorInitializationSeed', 'MatrixInitializationSeed'});
if export_results
    writetable(seed_ledger_table, ...
        fullfile(result_dir, 'R3_seed_ledger.csv'));
end

algorithm_parameters = struct();
algorithm_parameters.maxepochs = maxepochs;
algorithm_parameters.tolcost = tolcost;
algorithm_parameters.permute_on = permute_on_flag;
algorithm_parameters.verbose = verbose;
algorithm_parameters.lambda_list = lambda_list;
algorithm_parameters.rsi_lambda_min = rsi_lambda_min;
algorithm_parameters.rsi_lambda_max = rsi_lambda_max;
algorithm_parameters.rsi_mu = rsi_mu;
algorithm_parameters.rsi_grad_ema_alpha = rsi_grad_ema_alpha;
algorithm_parameters.rsi_irls_max_iters = rsi_irls_max_iters;
algorithm_parameters.rsi_irls_tolerance = rsi_irls_tolerance;
algorithm_parameters.rsi_normalization_epsilon = ...
    rsi_normalization_epsilon;
algorithm_parameters.cp_wopt_maxepochs = 30;
algorithm_parameters.petrels_lambda = 0.98;
algorithm_parameters.grasta_rho = 1.8;
algorithm_parameters.grasta_max_mu = 10000;
algorithm_parameters.grasta_min_mu = 1;
algorithm_parameters.grasta_inner_iterations = 20;
algorithm_parameters.grouse_step_size = 1e-4;
algorithm_parameters.tecpsgd_lambda = 0.99;
algorithm_parameters.tecpsgd_step_size = 0.10;
algorithm_parameters.tecpsgd_mu = 0.01;
algorithm_parameters.olstec_mu = 0.01;
algorithm_parameters.olstec_tw_flag = 0;
algorithm_parameters.olstec_tw_length = 10;
algorithm_parameters.grasta_use_mex = 0;
algorithm_parameters.cp_wopt_display_iterations = 1;
algorithm_parameters.store_online_reconstructions = true;
algorithm_parameters.store_subinformation = true;

video_file_info = dir(video_filename);
metadata_file_info = dir(meta_filename);
checkpoint_signature = struct();
checkpoint_signature.video_filename = video_filename;
checkpoint_signature.video_bytes = video_file_info.bytes;
checkpoint_signature.video_modified_datenum = video_file_info.datenum;
checkpoint_signature.meta_filename = meta_filename;
checkpoint_signature.metadata_bytes = metadata_file_info.bytes;
checkpoint_signature.metadata_modified_datenum = metadata_file_info.datenum;
checkpoint_signature.tensor_dims = tensor_dims;
checkpoint_signature.num_trials = num_trials;
checkpoint_signature.observation_ratios = observation_ratios;
checkpoint_signature.nominal_spatter_density = nominal_spatter_density;
checkpoint_signature.spatter_saturation_value = spatter_saturation_value;
checkpoint_signature.rank_r = rank_r;
checkpoint_signature.matrix_rank = matrix_rank;
checkpoint_signature.max_frames = max_frames;
checkpoint_signature.scale_ratio = scale_ratio;
checkpoint_signature.final_window = final_window;
checkpoint_signature.initial_calibration_frames = ...
    initial_calibration_frames;
checkpoint_signature.evaluation_start_frame = evaluation_start_frame;
checkpoint_signature.huber_scale_multiplier = huber_scale_multiplier;
checkpoint_signature.huber_delta_lower_bound = huber_delta_lower_bound;
checkpoint_signature.huber_delta_upper_bound = huber_delta_upper_bound;
checkpoint_signature.side_threshold_sigma_multiplier = ...
    side_threshold_sigma_multiplier;
checkpoint_signature.side_threshold_lower_bound = ...
    side_threshold_lower_bound;
checkpoint_signature.metric_names = metric_names;
checkpoint_signature.primary_metric = primary_metric;
checkpoint_signature.algorithm_names = algorithm_names;
checkpoint_signature.algorithm_parameters = algorithm_parameters;
checkpoint_signature.algorithm_switches = [run_cpwopt, run_petrels, ...
    run_grasta, run_grouse, run_tecpsgd, run_olstec, run_rsi_olstec];
checkpoint_signature.adaptive_min_grad = adaptive_min_grad;
checkpoint_signature.mask_seeds = mask_seeds;
checkpoint_signature.spatter_seeds = spatter_seeds;
checkpoint_signature.tensor_init_seeds = tensor_init_seeds;
checkpoint_signature.matrix_init_seeds = matrix_init_seeds;
checkpoint_signature.execution_seeds = execution_seeds;
checkpoint_signature.representative_trial = representative_trial;
checkpoint_signature.representative_rho = representative_rho;
checkpoint_signature.representative_frames = representative_frames;
checkpoint_signature.visual_algorithm_names = visual_algorithm_names;
checkpoint_signature.paired_ci_level = paired_ci_level;
checkpoint_signature.paired_ci_resamples = paired_ci_resamples;
checkpoint_signature.summary_ci_seed = summary_ci_seed;
checkpoint_signature.paired_ci_seed = paired_ci_seed;
checkpoint_signature.ci_method = ci_method;
checkpoint_signature.timing_observation_ratio = ...
    timing_observation_ratio;
checkpoint_signature.timing_method_names = timing_method_names;
checkpoint_signature.timing_warmup_runs_per_method = ...
    timing_warmup_runs_per_method;
checkpoint_signature.timing_store_subinformation = false;
checkpoint_signature.timing_store_online_reconstructions = false;
checkpoint_signature.timing_ci_level = timing_ci_level;
checkpoint_signature.timing_ci_resamples = timing_ci_resamples;
checkpoint_signature.timing_bootstrap_seed = timing_bootstrap_seed;
checkpoint_signature.timing_ci_method = timing_ci_method;

state = initialize_r3_state(num_algorithms, num_trials, num_ratios, ...
    total_slices, rows, cols, numel(representative_frames), ...
    num_timing_methods);

if resume_from_checkpoint && exist(checkpoint_file, 'file')
    loaded_checkpoint = load(checkpoint_file, 'checkpoint');
    if ~isfield(loaded_checkpoint, 'checkpoint')
        error('Exp_R3:InvalidCheckpoint', ...
            'Checkpoint file does not contain a checkpoint structure.');
    end
    checkpoint = loaded_checkpoint.checkpoint;
    if ~isfield(checkpoint, 'signature') || ~isfield(checkpoint, 'state')
        error('Exp_R3:InvalidCheckpoint', ...
            'Checkpoint is incomplete and cannot be resumed safely.');
    end
    if ~isequaln(checkpoint.signature, checkpoint_signature)
        error('Exp_R3:CheckpointConfigurationMismatch', ...
            ['Checkpoint settings differ from the current experiment. ' ...
             'Do not combine results generated with different settings.']);
    end
    validate_r3_state_dimensions(checkpoint.state, num_algorithms, ...
        num_trials, num_ratios, total_slices, rows, cols, ...
        numel(representative_frames), num_timing_methods);
    state = checkpoint.state;
    fprintf('Resuming R3: %d/%d algorithm-ratio-trial runs completed.\n', ...
        nnz(state.algorithm_status ~= 0), numel(state.algorithm_status));
    clear loaded_checkpoint checkpoint;
end

%% 4. Paired Monte Carlo Evaluation
% -------------------------------------------------------------------------
fprintf(['Running %d trials, %d observation ratios, and %d algorithms ' ...
    '(%d algorithm calls).\n'], num_trials, num_ratios, num_algorithms, ...
    num_trials * num_ratios * num_algorithms);
invocation_timer = tic;

for trial = 1:num_trials
    if all(reshape(state.algorithm_status(:, trial, :), [], 1) ~= 0)
        fprintf('Skipping completed trial %d/%d.\n', trial, num_trials);
        continue;
    end

    fprintf('\nTrial %d/%d\n', trial, num_trials);

    % One spatter realization is reused across all observation ratios in
    % this trial. It is independent of the observation-mask stream.
    spatter_stream = RandStream('mt19937ar', ...
        'Seed', spatter_seeds(trial));
    spatter_mask = rand(spatter_stream, rows, cols, total_slices) < ...
        nominal_spatter_density;
    Tensor_Y_Noisy = Tensor_Y_PreInjectionReference;
    Tensor_Y_Noisy(spatter_mask) = spatter_saturation_value;

    realized_spatter_density = nnz(spatter_mask) / numel(spatter_mask);
    effective_changed_density = nnz(spatter_mask & ...
        Tensor_Y_PreInjectionReference < spatter_saturation_value) / ...
        numel(spatter_mask);
    state.realized_spatter_density(trial) = record_or_validate_scalar( ...
        state.realized_spatter_density(trial), ...
        realized_spatter_density, 'realized spatter density');
    state.effective_changed_density(trial) = record_or_validate_scalar( ...
        state.effective_changed_density(trial), ...
        effective_changed_density, 'effective changed-pixel density');

    % Tensor methods share one explicit initialization within a trial.
    tensor_init_stream = RandStream('mt19937ar', ...
        'Seed', tensor_init_seeds(trial));
    tensor_init = struct();
    tensor_init.A = randn(tensor_init_stream, rows, rank_r);
    tensor_init.B = randn(tensor_init_stream, cols, rank_r);
    tensor_init.C = randn(tensor_init_stream, total_slices, rank_r);

    % Matrix methods share one initial column space within a trial.
    matrix_init_stream = RandStream('mt19937ar', ...
        'Seed', matrix_init_seeds(trial));
    matrix_init = struct();
    matrix_init.U = randn(matrix_init_stream, numr, matrix_rank);
    matrix_init.Weight = randn(matrix_init_stream, matrix_rank, numc);

    previous_higher_ratio_mask = [];
    for ratio_idx = 1:num_ratios
        observation_ratio = observation_ratios(ratio_idx);

        % Resetting the same trial-specific stream creates nested masks:
        % the lower-ratio observations are subsets of the higher-ratio ones.
        mask_stream = RandStream('mt19937ar', 'Seed', mask_seeds(trial));
        OmegaTensor = rand(mask_stream, rows, cols, total_slices) < ...
            observation_ratio;
        if ~isempty(previous_higher_ratio_mask) && ...
                any(OmegaTensor(:) & ~previous_higher_ratio_mask(:))
            error('Exp_R3:NonNestedObservationMasks', ...
                'Observation masks are not nested in trial %d.', trial);
        end
        previous_higher_ratio_mask = OmegaTensor;

        actual_observation_ratio = nnz(OmegaTensor) / numel(OmegaTensor);
        state.actual_observation_ratio(trial, ratio_idx) = ...
            record_or_validate_scalar( ...
            state.actual_observation_ratio(trial, ratio_idx), ...
            actual_observation_ratio, 'actual observation ratio');

        huber_delta = estimate_initial_calibration_huber_delta( ...
            Tensor_Y_Noisy, OmegaTensor, initial_calibration_frames, ...
            huber_scale_multiplier, huber_delta_lower_bound, ...
            huber_delta_upper_bound);
        state.huber_delta(trial, ratio_idx) = ...
            record_or_validate_scalar( ...
            state.huber_delta(trial, ratio_idx), ...
            huber_delta, 'Huber threshold');

        Matrix_Y_Noisy = reshape(Tensor_Y_Noisy, [numr, numc]);
        OmegaMatrix = reshape(OmegaTensor, [numr, numc]);

        is_representative_configuration = ...
            trial == representative_trial && ...
            ratio_idx == representative_rho_index;
        if is_representative_configuration
            state.representative_input = ...
                Tensor_Y_Noisy(:, :, representative_frames);
            frames_to_capture = representative_frames;
        else
            frames_to_capture = [];
        end

        fprintf('  rho=%.2f (actual %.6f, Huber delta %.6f)\n', ...
            observation_ratio, actual_observation_ratio, huber_delta);

        for algorithm_idx = 1:num_algorithms
            if state.algorithm_status(algorithm_idx, trial, ratio_idx) ~= 0
                continue;
            end

            algorithm_name = algorithm_names{algorithm_idx};
            execution_seed = execution_seeds(trial, algorithm_idx);
            fprintf('    %-14s ', algorithm_name);

            try
                rng(execution_seed, 'twister');
                [reference_nre, selected_reconstructions, ...
                    algorithm_seconds] = run_r3_algorithm( ...
                    algorithm_name, Tensor_Y_Noisy, Matrix_Y_Noisy, ...
                    OmegaTensor, OmegaMatrix, ...
                    Tensor_Y_PreInjectionReference, tensor_dims, rank_r, ...
                    matrix_rank, tensor_init, matrix_init, ...
                    algorithm_parameters, aux_width, adaptive_min_grad, ...
                    huber_delta, frames_to_capture);

                validate_reference_nre(reference_nre, algorithm_name, ...
                    trial, observation_ratio, total_slices);
                state.reference_nre(algorithm_idx, trial, ratio_idx, :) = ...
                    reshape(reference_nre, [1, 1, 1, total_slices]);
                state.algorithm_runtime_seconds( ...
                    algorithm_idx, trial, ratio_idx) = algorithm_seconds;
                state.algorithm_status(algorithm_idx, trial, ratio_idx) = 1;

                if is_representative_configuration
                    state.representative_reconstructions( ...
                        :, :, :, algorithm_idx) = selected_reconstructions;
                end
                fprintf('success (%.2fs)\n', algorithm_seconds);
            catch ME
                if is_process_interruption(ME)
                    rethrow(ME);
                end
                state.algorithm_status(algorithm_idx, trial, ratio_idx) = 2;
                state.algorithm_runtime_seconds( ...
                    algorithm_idx, trial, ratio_idx) = NaN;
                failure_record = make_failure_record(algorithm_name, ...
                    trial, observation_ratio, mask_seeds(trial), ...
                    spatter_seeds(trial), tensor_init_seeds(trial), ...
                    matrix_init_seeds(trial), execution_seed, ME);
                state.failure_records(end+1) = failure_record;
                fprintf('failed (%s)\n', ME.message);
            end

            if export_results
                checkpoint = struct();
                checkpoint.signature = checkpoint_signature;
                checkpoint.state = state;
                save_checkpoint_atomic(checkpoint_file, checkpoint);
                clear checkpoint;
            end

            clear reference_nre selected_reconstructions;
        end

        clear OmegaTensor OmegaMatrix Matrix_Y_Noisy mask_stream;
    end

    clear Tensor_Y_Noisy spatter_mask spatter_stream tensor_init ...
        matrix_init tensor_init_stream matrix_init_stream ...
        previous_higher_ratio_mask;
end

invocation_elapsed_seconds = toc(invocation_timer);
if any(state.algorithm_status(:) == 0)
    error('Exp_R3:IncompleteEvaluation', ...
        'The experiment ended with pending algorithm runs.');
end
fprintf('\nAll configured runs completed in this invocation after %.2f hours.\n', ...
    invocation_elapsed_seconds / 3600);

%% 5. Paired Computational Benchmark
% -------------------------------------------------------------------------
fprintf(['\nBenchmarking implementation-level computational cost at ' ...
    'rho=%.2f...\n'], timing_observation_ratio);
timing_invocation_timer = tic;
timing_warmed_this_invocation = false;
timing_rsi_index = find(strcmp(timing_method_names, 'RSI_OLSTEC'), 1);

for trial = 1:num_trials
    timing_complete = all(isfinite(state.timing_wall_seconds(:, trial))) && ...
        all(isfinite(state.timing_core_seconds(:, trial))) && ...
        isfinite(state.timing_prior_irls_iterations(trial)) && ...
        isfinite(state.timing_posterior_irls_iterations(trial));
    if timing_complete
        continue;
    end

    spatter_stream = RandStream('mt19937ar', ...
        'Seed', spatter_seeds(trial));
    spatter_mask = rand(spatter_stream, rows, cols, total_slices) < ...
        nominal_spatter_density;
    Tensor_Y_Noisy = Tensor_Y_PreInjectionReference;
    Tensor_Y_Noisy(spatter_mask) = spatter_saturation_value;

    mask_stream = RandStream('mt19937ar', 'Seed', mask_seeds(trial));
    OmegaTensor = rand(mask_stream, rows, cols, total_slices) < ...
        timing_observation_ratio;
    actual_observation_ratio = nnz(OmegaTensor) / numel(OmegaTensor);
    state.actual_observation_ratio(trial, timing_rho_index) = ...
        record_or_validate_scalar( ...
        state.actual_observation_ratio(trial, timing_rho_index), ...
        actual_observation_ratio, 'timing actual observation ratio');

    huber_delta = estimate_initial_calibration_huber_delta( ...
        Tensor_Y_Noisy, OmegaTensor, initial_calibration_frames, ...
        huber_scale_multiplier, huber_delta_lower_bound, ...
        huber_delta_upper_bound);
    state.huber_delta(trial, timing_rho_index) = ...
        record_or_validate_scalar( ...
        state.huber_delta(trial, timing_rho_index), ...
        huber_delta, 'timing Huber threshold');

    tensor_init_stream = RandStream('mt19937ar', ...
        'Seed', tensor_init_seeds(trial));
    tensor_init = struct();
    tensor_init.A = randn(tensor_init_stream, rows, rank_r);
    tensor_init.B = randn(tensor_init_stream, cols, rank_r);
    tensor_init.C = randn(tensor_init_stream, total_slices, rank_r);

    if ~timing_warmed_this_invocation
        fprintf('  Warming both computational paths without recording...\n');
        for warmup_run = 1:timing_warmup_runs_per_method
            for method_idx = 1:num_timing_methods
                run_r3_timing_method(timing_method_names{method_idx}, ...
                    Tensor_Y_Noisy, OmegaTensor, tensor_dims, rank_r, ...
                    tensor_init, algorithm_parameters, aux_width, ...
                    adaptive_min_grad, huber_delta);
            end
        end
        timing_warmed_this_invocation = true;
    end

    if mod(trial, 2) == 1
        method_order = 1:num_timing_methods;
    else
        method_order = num_timing_methods:-1:1;
    end

    fprintf('  Timing trial %d/%d: %s first\n', trial, num_trials, ...
        timing_method_names{method_order(1)});
    timing_results = cell(num_timing_methods, 1);
    for position = 1:num_timing_methods
        method_idx = method_order(position);
        timing_results{method_idx} = run_r3_timing_method( ...
            timing_method_names{method_idx}, Tensor_Y_Noisy, ...
            OmegaTensor, tensor_dims, rank_r, tensor_init, ...
            algorithm_parameters, aux_width, adaptive_min_grad, ...
            huber_delta);
    end

    for method_idx = 1:num_timing_methods
        state.timing_wall_seconds(method_idx, trial) = ...
            timing_results{method_idx}.wall_seconds;
        state.timing_core_seconds(method_idx, trial) = ...
            timing_results{method_idx}.core_seconds;
    end
    state.timing_prior_irls_iterations(trial) = ...
        timing_results{timing_rsi_index}.mean_prior_irls_iterations;
    state.timing_posterior_irls_iterations(trial) = ...
        timing_results{timing_rsi_index}.mean_posterior_irls_iterations;

    if export_results
        checkpoint = struct();
        checkpoint.signature = checkpoint_signature;
        checkpoint.state = state;
        save_checkpoint_atomic(checkpoint_file, checkpoint);
        clear checkpoint;
    end

    clear Tensor_Y_Noisy OmegaTensor spatter_mask spatter_stream ...
        mask_stream tensor_init tensor_init_stream timing_results;
end

timing_invocation_elapsed_seconds = toc(timing_invocation_timer);
if any(~isfinite(state.timing_wall_seconds(:))) || ...
        any(~isfinite(state.timing_core_seconds(:))) || ...
        any(~isfinite(state.timing_prior_irls_iterations)) || ...
        any(~isfinite(state.timing_posterior_irls_iterations))
    error('Exp_R3:IncompleteTimingBenchmark', ...
        'The paired computational benchmark is incomplete.');
end
fprintf('Paired computational benchmark completed in %.2f hours.\n', ...
    timing_invocation_elapsed_seconds / 3600);

%% 6. Prespecified Metrics and Statistical Summaries
% -------------------------------------------------------------------------
scalar_metrics = compute_r3_scalar_metrics( ...
    state.reference_nre, state.algorithm_status, evaluation_frames, ...
    final_window);

per_run_table = build_r3_per_run_table(algorithm_names, ...
    observation_ratios, state, scalar_metrics, mask_seeds, spatter_seeds, ...
    tensor_init_seeds, matrix_init_seeds, execution_seeds);

summary_table = build_r3_summary_table(algorithm_names, ...
    observation_ratios, state.algorithm_status, scalar_metrics, ...
    metric_names, paired_ci_level, paired_ci_resamples, ...
    summary_ci_seed, ci_method);

paired_comparison_table = build_r3_paired_comparison(algorithm_names, ...
    observation_ratios, state.algorithm_status, scalar_metrics, ...
    metric_names, 'RSI_OLSTEC', paired_ci_level, ...
    paired_ci_resamples, paired_ci_seed, ci_method);

[timing_per_trial_table, timing_summary_table, ...
    timing_overhead_table] = build_r3_timing_tables( ...
    timing_method_names, timing_observation_ratio, timing_rho_index, ...
    state, total_slices, mask_seeds, spatter_seeds, tensor_init_seeds, ...
    timing_ci_level, timing_ci_resamples, timing_bootstrap_seed, ...
    timing_ci_method);

failure_table = struct2table(state.failure_records);
failure_summary_table = build_r3_failure_summary( ...
    state.failure_records, algorithm_names, observation_ratios);

if export_results
    writetable(per_run_table, ...
        fullfile(result_dir, 'R3_per_trial_metrics.csv'));
    writetable(summary_table, ...
        fullfile(result_dir, 'R3_summary_statistics.csv'));
    writetable(paired_comparison_table, ...
        fullfile(result_dir, 'R3_paired_difference_ci.csv'));
    writetable(failure_table, ...
        fullfile(result_dir, 'R3_failure_log.csv'));
    writetable(failure_summary_table, ...
        fullfile(result_dir, 'R3_failure_summary.csv'));
    writetable(timing_per_trial_table, ...
        fullfile(result_dir, 'R3_timing_per_trial.csv'));
    writetable(timing_summary_table, ...
        fullfile(result_dir, 'R3_timing_summary.csv'));
    writetable(timing_overhead_table, ...
        fullfile(result_dir, 'R3_timing_overhead_ci.csv'));

    save(fullfile(result_dir, 'R3_stats.mat'), ...
        'state', 'scalar_metrics', 'summary_table', ...
        'paired_comparison_table', 'failure_table', ...
        'failure_summary_table', 'timing_per_trial_table', ...
        'timing_summary_table', 'timing_overhead_table', ...
        'seed_ledger_table', ...
        'checkpoint_signature', 'primary_metric', ...
        'initial_calibration_frames', 'evaluation_start_frame', ...
        'aux_meta', 'invocation_elapsed_seconds', ...
        'timing_invocation_elapsed_seconds', '-v7');
end

print_primary_metric_summary(summary_table, primary_metric);
disp(timing_summary_table);
disp(timing_overhead_table);

%% 7. Aggregate and Prespecified Visualizations
% -------------------------------------------------------------------------
plot_r3_mean_trajectories(state.reference_nre, ...
    state.algorithm_status, algorithm_names, observation_ratios, ...
    representative_rho, evaluation_start_frame, false, ...
    figure_visibility, result_dir, export_results, ...
    'Fig_Residual_Error_R3.fig');

plot_r3_mean_trajectories(state.reference_nre, ...
    state.algorithm_status, algorithm_names, observation_ratios, ...
    representative_rho, evaluation_start_frame, true, ...
    figure_visibility, result_dir, export_results, ...
    'Fig_Running_Average_Error_R3.fig');

plot_r3_representative_frames(state.representative_input, ...
    state.representative_reconstructions, algorithm_names, ...
    visual_algorithm_names, representative_frames, representative_trial, ...
    representative_rho, figure_visibility, result_dir, export_results);

fprintf('Experiment R3 completed. Results: %s\n', result_dir);

%% Local Functions
% -------------------------------------------------------------------------
function state = initialize_r3_state(num_algorithms, num_trials, ...
    num_ratios, total_slices, rows, cols, num_representative_frames, ...
    num_timing_methods)

    state = struct();
    state.reference_nre = NaN(num_algorithms, num_trials, ...
        num_ratios, total_slices);
    state.algorithm_status = zeros(num_algorithms, num_trials, ...
        num_ratios, 'uint8');
    state.algorithm_runtime_seconds = NaN(num_algorithms, ...
        num_trials, num_ratios);
    state.actual_observation_ratio = NaN(num_trials, num_ratios);
    state.huber_delta = NaN(num_trials, num_ratios);
    state.realized_spatter_density = NaN(num_trials, 1);
    state.effective_changed_density = NaN(num_trials, 1);
    state.representative_input = NaN(rows, cols, ...
        num_representative_frames);
    state.representative_reconstructions = NaN(rows, cols, ...
        num_representative_frames, num_algorithms);
    state.timing_wall_seconds = NaN(num_timing_methods, num_trials);
    state.timing_core_seconds = NaN(num_timing_methods, num_trials);
    state.timing_prior_irls_iterations = NaN(num_trials, 1);
    state.timing_posterior_irls_iterations = NaN(num_trials, 1);
    state.failure_records = empty_failure_records();
end

function validate_r3_state_dimensions(state, num_algorithms, num_trials, ...
    num_ratios, total_slices, rows, cols, num_representative_frames, ...
    num_timing_methods)

    required_fields = {'reference_nre', 'algorithm_status', ...
        'algorithm_runtime_seconds', 'actual_observation_ratio', ...
        'huber_delta', 'realized_spatter_density', ...
        'effective_changed_density', 'representative_input', ...
        'representative_reconstructions', 'timing_wall_seconds', ...
        'timing_core_seconds', 'timing_prior_irls_iterations', ...
        'timing_posterior_irls_iterations', 'failure_records'};
    if ~all(isfield(state, required_fields))
        error('Exp_R3:InvalidCheckpoint', ...
            'Checkpoint state is missing required fields.');
    end

    expected_curve_size = [num_algorithms, num_trials, ...
        num_ratios, total_slices];
    expected_status_size = [num_algorithms, num_trials, num_ratios];
    expected_ratio_size = [num_trials, num_ratios];
    expected_trial_size = [num_trials, 1];
    expected_input_size = [rows, cols, num_representative_frames];
    expected_reconstruction_size = [rows, cols, ...
        num_representative_frames, num_algorithms];
    expected_timing_size = [num_timing_methods, num_trials];

    if ~isequal(size(state.reference_nre), expected_curve_size) || ...
            ~isequal(size(state.algorithm_status), expected_status_size) || ...
            ~isequal(size(state.algorithm_runtime_seconds), ...
            expected_status_size) || ...
            ~isequal(size(state.actual_observation_ratio), ...
            expected_ratio_size) || ...
            ~isequal(size(state.huber_delta), expected_ratio_size) || ...
            ~isequal(size(state.realized_spatter_density), ...
            expected_trial_size) || ...
            ~isequal(size(state.effective_changed_density), ...
            expected_trial_size) || ...
            ~isequal(size(state.representative_input), expected_input_size) || ...
            ~isequal(size(state.representative_reconstructions), ...
            expected_reconstruction_size) || ...
            ~isequal(size(state.timing_wall_seconds), ...
            expected_timing_size) || ...
            ~isequal(size(state.timing_core_seconds), ...
            expected_timing_size) || ...
            ~isequal(size(state.timing_prior_irls_iterations), ...
            expected_trial_size) || ...
            ~isequal(size(state.timing_posterior_irls_iterations), ...
            expected_trial_size)
        error('Exp_R3:InvalidCheckpoint', ...
            'Checkpoint state dimensions do not match the current settings.');
    end

    if any(~ismember(state.algorithm_status(:), uint8([0, 1, 2])))
        error('Exp_R3:InvalidCheckpoint', ...
            'Checkpoint contains an invalid algorithm status code.');
    end

    successful_runs = find(state.algorithm_status == 1);
    for run_idx = reshape(successful_runs, 1, [])
        [algorithm_idx, trial, ratio_idx] = ind2sub( ...
            size(state.algorithm_status), run_idx);
        curve = reshape(state.reference_nre( ...
            algorithm_idx, trial, ratio_idx, :), 1, []);
        if any(~isfinite(curve))
            error('Exp_R3:InvalidCheckpoint', ...
                'A successful checkpoint run has an incomplete NRE curve.');
        end
        runtime = state.algorithm_runtime_seconds( ...
            algorithm_idx, trial, ratio_idx);
        if ~isfinite(runtime) || runtime < 0
            error('Exp_R3:InvalidCheckpoint', ...
                'A successful checkpoint run has an invalid runtime.');
        end
    end

    if numel(state.failure_records) ~= nnz(state.algorithm_status == 2)
        error('Exp_R3:InvalidCheckpoint', ...
            'Failed status count and failure audit records disagree.');
    end

    timing_complete = all(isfinite(state.timing_wall_seconds), 1) & ...
        all(isfinite(state.timing_core_seconds), 1) & ...
        isfinite(state.timing_prior_irls_iterations)' & ...
        isfinite(state.timing_posterior_irls_iterations)';
    timing_pending = all(isnan(state.timing_wall_seconds), 1) & ...
        all(isnan(state.timing_core_seconds), 1) & ...
        isnan(state.timing_prior_irls_iterations)' & ...
        isnan(state.timing_posterior_irls_iterations)';
    if any(~(timing_complete | timing_pending))
        error('Exp_R3:InvalidCheckpoint', ...
            'Checkpoint contains an incomplete paired timing result.');
    end
    if any(any(state.timing_wall_seconds(:, timing_complete) <= 0)) || ...
            any(any(state.timing_core_seconds(:, timing_complete) <= 0)) || ...
            any(state.timing_prior_irls_iterations(timing_complete) < 0) || ...
            any(state.timing_posterior_irls_iterations(timing_complete) < 0)
        error('Exp_R3:InvalidCheckpoint', ...
            'Checkpoint contains an invalid computational timing result.');
    end
end

function value = record_or_validate_scalar(existing_value, new_value, label)
    if ~isfinite(new_value)
        error('Exp_R3:InvalidDeterministicQuantity', ...
            '%s is not finite.', label);
    end
    if isnan(existing_value)
        value = new_value;
        return;
    end

    tolerance = 10 * eps(max(1, abs(new_value)));
    if abs(existing_value - new_value) > tolerance
        error('Exp_R3:DeterministicReplayMismatch', ...
            '%s changed during deterministic checkpoint replay.', label);
    end
    value = existing_value;
end

function huber_delta = estimate_initial_calibration_huber_delta( ...
    observed_tensor, observation_mask, calibration_frames, ...
    scale_multiplier, lower_bound, upper_bound)

    total_slices = size(observed_tensor, 3);
    if calibration_frames < 2 || calibration_frames > total_slices
        error('Exp_R3:InvalidHuberCalibrationLength', ...
            ['calibration_frames must be between 2 and the number of ' ...
             'available frames.']);
    end

    % Only masked corrupted observations are used; no reference is supplied.
    diff_pixels = cell(calibration_frames - 1, 1);
    for frame = 2:calibration_frames
        common_mask = observation_mask(:, :, frame) & ...
            observation_mask(:, :, frame - 1);
        diff_frame = observed_tensor(:, :, frame) - ...
            observed_tensor(:, :, frame - 1);
        diff_pixels{frame - 1} = diff_frame(common_mask);
    end
    diff_pixels = vertcat(diff_pixels{:});

    if isempty(diff_pixels)
        huber_delta = upper_bound;
        return;
    end

    centered_mad = median(abs(diff_pixels - ...
        median(diff_pixels, 'omitnan')), 'omitnan');
    estimated_sigma = (1.4826 * centered_mad) / sqrt(2);
    if ~isfinite(estimated_sigma)
        huber_delta = upper_bound;
        return;
    end
    huber_delta = max(lower_bound, ...
        min(upper_bound, scale_multiplier * estimated_sigma));
end

function result = run_r3_timing_method(method_name, tensor_data, ...
    tensor_mask, tensor_dims, rank_r, tensor_init, parameters, ...
    aux_width, adaptive_min_grad, huber_delta)

    GammaTensor = [];
    switch method_name
        case 'OLSTEC_80'
            options = struct( ...
                'maxepochs', parameters.maxepochs, ...
                'tolcost', parameters.tolcost, ...
                'early_stop_on', 'none', ...
                'permute_on', false, ...
                'lambda', 0.80, ...
                'mu', parameters.olstec_mu, ...
                'tw_flag', parameters.olstec_tw_flag, ...
                'tw_len', parameters.olstec_tw_length, ...
                'store_subinfo', false, ...
                'store_matrix', false, ...
                'verbose', 0);
            wall_timer = tic;
            [~, infos, ~] = olstec(tensor_data, tensor_mask, ...
                GammaTensor, tensor_dims, rank_r, tensor_init, options);
            wall_seconds = toc(wall_timer);
            mean_prior_irls_iterations = NaN;
            mean_posterior_irls_iterations = NaN;

        case 'RSI_OLSTEC'
            options = struct( ...
                'maxepochs', parameters.maxepochs, ...
                'lambda_max', parameters.rsi_lambda_max, ...
                'lambda_min', parameters.rsi_lambda_min, ...
                'huber_delta', huber_delta, ...
                'min_grad_threshold', adaptive_min_grad, ...
                'grad_ema_alpha', parameters.rsi_grad_ema_alpha, ...
                'mu', parameters.rsi_mu, ...
                'irls_max_iters', parameters.rsi_irls_max_iters, ...
                'irls_tolerance', parameters.rsi_irls_tolerance, ...
                'normalization_epsilon', ...
                parameters.rsi_normalization_epsilon, ...
                'tolcost', parameters.tolcost, ...
                'early_stop_on', 'none', ...
                'permute_on', false, ...
                'store_subinfo', false, ...
                'store_matrix', false, ...
                'verbose', 0);
            wall_timer = tic;
            [~, infos, sub_infos] = rsi_olstec(tensor_data, ...
                tensor_mask, GammaTensor, tensor_dims, rank_r, ...
                tensor_init, options, aux_width);
            wall_seconds = toc(wall_timer);
            mean_prior_irls_iterations = finite_mean( ...
                sub_infos.prior_irls_iterations);
            mean_posterior_irls_iterations = finite_mean( ...
                sub_infos.posterior_irls_iterations);

        otherwise
            error('Exp_R3:UnknownTimingMethod', ...
                'Unknown computational benchmark method: %s.', ...
                method_name);
    end

    if ~isstruct(infos) || ~isfield(infos, 'time') || ...
            isempty(infos.time)
        error('Exp_R3:MissingCoreRuntime', ...
            '%s did not return the required core-loop runtime.', ...
            method_name);
    end
    core_seconds = infos.time(end);
    if ~isfinite(wall_seconds) || wall_seconds <= 0 || ...
            ~isfinite(core_seconds) || core_seconds <= 0
        error('Exp_R3:InvalidTimingResult', ...
            '%s returned a nonpositive or nonfinite runtime.', method_name);
    end
    if strcmp(method_name, 'RSI_OLSTEC') && ...
            (~isfinite(mean_prior_irls_iterations) || ...
            mean_prior_irls_iterations < 0 || ...
            ~isfinite(mean_posterior_irls_iterations) || ...
            mean_posterior_irls_iterations < 0)
        error('Exp_R3:InvalidIRLSIterationHistory', ...
            'RSI-OLSTEC returned an invalid IRLS iteration history.');
    end

    result = struct( ...
        'wall_seconds', wall_seconds, ...
        'core_seconds', core_seconds, ...
        'mean_prior_irls_iterations', mean_prior_irls_iterations, ...
        'mean_posterior_irls_iterations', ...
        mean_posterior_irls_iterations);
end

function [reference_nre, selected_reconstructions, algorithm_seconds] = ...
    run_r3_algorithm(algorithm_name, tensor_data, matrix_data, ...
    tensor_mask, matrix_mask, reference_tensor, tensor_dims, rank_r, ...
    matrix_rank, tensor_init, matrix_init, parameters, aux_width, ...
    adaptive_min_grad, huber_delta, representative_frames)

    GammaTensor = [];
    GammaMatrix = [];
    store_matrix = parameters.store_online_reconstructions;
    store_subinfo = parameters.store_subinformation;

    switch algorithm_name
        case 'CP_WOPT'
            options = struct( ...
                'maxepochs', parameters.cp_wopt_maxepochs, ...
                'display_iters', ...
                parameters.cp_wopt_display_iterations, ...
                'tolcost', parameters.tolcost, ...
                'store_subinfo', store_subinfo, ...
                'store_matrix', false, ...
                'verbose', parameters.verbose);
            algorithm_timer = tic;
            [Xsol, ~, ~] = cp_wopt_mod(tensor_data, tensor_mask, ...
                GammaTensor, tensor_dims, rank_r, tensor_init, options);
            algorithm_seconds = toc(algorithm_timer);
            [reference_nre, selected_reconstructions] = ...
                reference_nre_from_cp_factors(reference_tensor, Xsol, ...
                representative_frames);

        case 'Petrels'
            options = struct( ...
                'maxepochs', parameters.maxepochs, ...
                'tolcost', parameters.tolcost, ...
                'rank', matrix_rank, ...
                'permute_on', parameters.permute_on, ...
                'store_subinfo', store_subinfo, ...
                'store_matrix', store_matrix, ...
                'verbose', parameters.verbose, ...
                'lambda', parameters.petrels_lambda);
            algorithm_timer = tic;
            [~, ~, sub_info, ~] = petrels_mod(matrix_init, ...
                matrix_data, matrix_mask, GammaMatrix, ...
                size(matrix_data, 1), size(matrix_data, 2), options);
            algorithm_seconds = toc(algorithm_timer);
            [reference_nre, selected_reconstructions] = ...
                reference_nre_from_stored_reconstruction( ...
                reference_tensor, require_reconstruction( ...
                sub_info, algorithm_name), representative_frames);

        case 'Grasta'
            options = struct( ...
                'maxepochs', parameters.maxepochs, ...
                'tolcost', parameters.tolcost, ...
                'permute_on', parameters.permute_on, ...
                'verbose', parameters.verbose, ...
                'store_subinfo', store_subinfo, ...
                'store_matrix', store_matrix, ...
                'RANK', matrix_rank, ...
                'rho', parameters.grasta_rho, ...
                'MAX_MU', parameters.grasta_max_mu, ...
                'MIN_MU', parameters.grasta_min_mu, ...
                'ITER_MAX', parameters.grasta_inner_iterations, ...
                'DIM_M', size(matrix_data, 1), ...
                'USE_MEX', parameters.grasta_use_mex);
            algorithm_timer = tic;
            [~, ~, sub_info, ~] = grasta_mod(matrix_init, ...
                matrix_data, matrix_mask, GammaMatrix, ...
                size(matrix_data, 1), size(matrix_data, 2), options);
            algorithm_seconds = toc(algorithm_timer);
            [reference_nre, selected_reconstructions] = ...
                reference_nre_from_stored_reconstruction( ...
                reference_tensor, require_reconstruction( ...
                sub_info, algorithm_name), representative_frames);

        case 'Grouse'
            options = struct( ...
                'maxrank', matrix_rank, ...
                'step_size', parameters.grouse_step_size, ...
                'maxepochs', parameters.maxepochs, ...
                'tolcost', parameters.tolcost, ...
                'permute_on', parameters.permute_on, ...
                'store_subinfo', store_subinfo, ...
                'store_matrix', store_matrix, ...
                'verbose', parameters.verbose);
            algorithm_timer = tic;
            [~, ~, sub_info, ~] = grouse_mod(matrix_init, ...
                matrix_data, matrix_mask, GammaMatrix, ...
                size(matrix_data, 1), size(matrix_data, 2), options);
            algorithm_seconds = toc(algorithm_timer);
            [reference_nre, selected_reconstructions] = ...
                reference_nre_from_stored_reconstruction( ...
                reference_tensor, require_reconstruction( ...
                sub_info, algorithm_name), representative_frames);

        case 'TeCPSGD'
            options = struct( ...
                'maxepochs', parameters.maxepochs, ...
                'tolcost', parameters.tolcost, ...
                'lambda', parameters.tecpsgd_lambda, ...
                'stepsize', parameters.tecpsgd_step_size, ...
                'mu', parameters.tecpsgd_mu, ...
                'permute_on', parameters.permute_on, ...
                'store_subinfo', store_subinfo, ...
                'store_matrix', store_matrix, ...
                'verbose', parameters.verbose);
            algorithm_timer = tic;
            [~, ~, sub_info] = TeCPSGD(tensor_data, tensor_mask, ...
                GammaTensor, tensor_dims, rank_r, tensor_init, options);
            algorithm_seconds = toc(algorithm_timer);
            [reference_nre, selected_reconstructions] = ...
                reference_nre_from_stored_reconstruction( ...
                reference_tensor, require_reconstruction( ...
                sub_info, algorithm_name), representative_frames);

        case 'RSI_OLSTEC'
            options = struct( ...
                'maxepochs', parameters.maxepochs, ...
                'lambda_max', parameters.rsi_lambda_max, ...
                'lambda_min', parameters.rsi_lambda_min, ...
                'huber_delta', huber_delta, ...
                'min_grad_threshold', adaptive_min_grad, ...
                'grad_ema_alpha', parameters.rsi_grad_ema_alpha, ...
                'mu', parameters.rsi_mu, ...
                'irls_max_iters', parameters.rsi_irls_max_iters, ...
                'irls_tolerance', parameters.rsi_irls_tolerance, ...
                'normalization_epsilon', ...
                parameters.rsi_normalization_epsilon, ...
                'tolcost', parameters.tolcost, ...
                'permute_on', parameters.permute_on, ...
                'store_subinfo', store_subinfo, ...
                'store_matrix', store_matrix, ...
                'verbose', parameters.verbose);
            algorithm_timer = tic;
            [~, ~, sub_info] = rsi_olstec(tensor_data, tensor_mask, ...
                GammaTensor, tensor_dims, rank_r, tensor_init, options, ...
                aux_width);
            algorithm_seconds = toc(algorithm_timer);
            [reference_nre, selected_reconstructions] = ...
                reference_nre_from_stored_reconstruction( ...
                reference_tensor, require_reconstruction( ...
                sub_info, algorithm_name), representative_frames);

        otherwise
            if startsWith(algorithm_name, 'OLSTEC_')
                lambda = str2double(algorithm_name(8:end)) / 100;
                if ~any(abs(parameters.lambda_list - lambda) < 1e-12)
                    error('Exp_R3:UnexpectedOLSTECLambda', ...
                        'Unconfigured OLSTEC lambda in %s.', algorithm_name);
                end
                options = struct( ...
                    'maxepochs', parameters.maxepochs, ...
                    'tolcost', parameters.tolcost, ...
                    'permute_on', parameters.permute_on, ...
                    'lambda', lambda, ...
                    'mu', parameters.olstec_mu, ...
                    'tw_flag', parameters.olstec_tw_flag, ...
                    'tw_len', parameters.olstec_tw_length, ...
                    'store_subinfo', store_subinfo, ...
                    'store_matrix', store_matrix, ...
                    'verbose', parameters.verbose);
                algorithm_timer = tic;
                [~, ~, sub_info] = olstec(tensor_data, tensor_mask, ...
                    GammaTensor, tensor_dims, rank_r, tensor_init, options);
                algorithm_seconds = toc(algorithm_timer);
                [reference_nre, selected_reconstructions] = ...
                    reference_nre_from_stored_reconstruction( ...
                    reference_tensor, require_reconstruction( ...
                    sub_info, algorithm_name), representative_frames);
            else
                error('Exp_R3:UnknownAlgorithm', ...
                    'Unknown algorithm: %s.', algorithm_name);
            end
    end
end

function reconstruction = require_reconstruction(sub_info, algorithm_name)
    if ~isstruct(sub_info) || ~isfield(sub_info, 'L') || ...
            isempty(sub_info.L)
        error('Exp_R3:MissingReconstruction', ...
            '%s did not return the required online reconstruction.', ...
            algorithm_name);
    end
    reconstruction = sub_info.L;
end

function [reference_nre, selected_reconstructions] = ...
    reference_nre_from_stored_reconstruction( ...
    reference_tensor, reconstruction, selected_frames)

    [rows, cols, total_slices] = size(reference_tensor);
    if ndims(reconstruction) == 3
        if ~isequal(size(reconstruction), size(reference_tensor))
            error('Exp_R3:ReconstructionSizeMismatch', ...
                'Tensor reconstruction dimensions do not match the reference.');
        end
        tensor_output = true;
    elseif ismatrix(reconstruction) && ...
            isequal(size(reconstruction), [rows * cols, total_slices])
        tensor_output = false;
    else
        error('Exp_R3:ReconstructionSizeMismatch', ...
            ['Reconstruction must be rows-by-cols-by-frames or ' ...
             '(rows*cols)-by-frames.']);
    end

    reference_nre = NaN(1, total_slices);
    selected_reconstructions = NaN(rows, cols, numel(selected_frames));
    selected_lookup = zeros(1, total_slices);
    selected_lookup(selected_frames) = 1:numel(selected_frames);

    for frame = 1:total_slices
        if tensor_output
            reconstructed_frame = reconstruction(:, :, frame);
        else
            reconstructed_frame = reshape( ...
                reconstruction(:, frame), [rows, cols]);
        end
        reference_frame = reference_tensor(:, :, frame);
        reference_nre(frame) = normalized_frame_error( ...
            reference_frame, reconstructed_frame);
        selected_idx = selected_lookup(frame);
        if selected_idx > 0
            selected_reconstructions(:, :, selected_idx) = ...
                reconstructed_frame;
        end
    end
end

function [reference_nre, selected_reconstructions] = ...
    reference_nre_from_cp_factors(reference_tensor, Xsol, selected_frames)

    if ~isstruct(Xsol) || ~all(isfield(Xsol, {'A', 'B', 'C'}))
        error('Exp_R3:UnexpectedCPOutput', ...
            'CP-WOPT did not return the expected A, B, and C factors.');
    end

    [rows, cols, total_slices] = size(reference_tensor);
    if size(Xsol.A, 1) ~= rows || size(Xsol.B, 1) ~= cols || ...
            size(Xsol.C, 1) ~= total_slices
        error('Exp_R3:CPFactorSizeMismatch', ...
            'CP-WOPT factor dimensions do not match the reference tensor.');
    end

    reference_nre = NaN(1, total_slices);
    selected_reconstructions = NaN(rows, cols, numel(selected_frames));
    selected_lookup = zeros(1, total_slices);
    selected_lookup(selected_frames) = 1:numel(selected_frames);

    for frame = 1:total_slices
        reconstructed_frame = Xsol.A * ...
            diag(Xsol.C(frame, :)) * Xsol.B';
        reference_nre(frame) = normalized_frame_error( ...
            reference_tensor(:, :, frame), reconstructed_frame);
        selected_idx = selected_lookup(frame);
        if selected_idx > 0
            selected_reconstructions(:, :, selected_idx) = ...
                reconstructed_frame;
        end
    end
end

function value = normalized_frame_error(reference_frame, reconstructed_frame)
    denominator = norm(reference_frame(:));
    if denominator <= eps
        value = norm(reconstructed_frame(:) - reference_frame(:));
    else
        value = norm(reconstructed_frame(:) - reference_frame(:)) / ...
            denominator;
    end
end

function validate_reference_nre(reference_nre, algorithm_name, trial, ...
    observation_ratio, total_slices)

    if ~isvector(reference_nre) || numel(reference_nre) ~= total_slices
        error('Exp_R3:InvalidMetricLength', ...
            ['%s returned %d reference-NRE values in trial %d at rho=%.2f; ' ...
             '%d values were required.'], algorithm_name, ...
            numel(reference_nre), trial, observation_ratio, total_slices);
    end

    nonfinite = find(~isfinite(reference_nre));
    if ~isempty(nonfinite)
        error('Exp_R3:NonfiniteMetric', ...
            ['%s returned non-finite reference NRE in trial %d at ' ...
             'rho=%.2f (first frame=%d; count=%d).'], ...
            algorithm_name, trial, observation_ratio, ...
            nonfinite(1), numel(nonfinite));
    end
    if any(reference_nre < 0)
        error('Exp_R3:NegativeMetric', ...
            '%s returned a negative reference NRE.', algorithm_name);
    end
end

function scalar_metrics = compute_r3_scalar_metrics( ...
    reference_nre, algorithm_status, evaluation_frames, final_window)

    [num_algorithms, num_trials, num_ratios, total_slices] = ...
        size(reference_nre);
    if isempty(evaluation_frames) || evaluation_frames(1) < 1 || ...
            evaluation_frames(end) > total_slices
        error('Exp_R3:InvalidEvaluationFrames', ...
            'evaluation_frames must be a nonempty subset of the sequence.');
    end
    scalar_metrics = struct();
    scalar_metrics.PostCalibrationMeanReferenceNRE = ...
        NaN(num_algorithms, num_trials, num_ratios);
    scalar_metrics.FinalFrameReferenceNRE = ...
        NaN(num_algorithms, num_trials, num_ratios);
    scalar_metrics.LateWindowMeanReferenceNRE = ...
        NaN(num_algorithms, num_trials, num_ratios);

    for algorithm_idx = 1:num_algorithms
        for trial = 1:num_trials
            for ratio_idx = 1:num_ratios
                if algorithm_status(algorithm_idx, trial, ratio_idx) ~= 1
                    continue;
                end
                curve = reshape(reference_nre( ...
                    algorithm_idx, trial, ratio_idx, :), ...
                    [1, total_slices]);
                scalar_metrics.PostCalibrationMeanReferenceNRE( ...
                    algorithm_idx, trial, ratio_idx) = ...
                    mean(curve(evaluation_frames));
                scalar_metrics.FinalFrameReferenceNRE( ...
                    algorithm_idx, trial, ratio_idx) = curve(end);
                scalar_metrics.LateWindowMeanReferenceNRE( ...
                    algorithm_idx, trial, ratio_idx) = ...
                    mean(curve(final_window));
            end
        end
    end
end

function output_table = build_r3_per_run_table(algorithm_names, ...
    observation_ratios, state, scalar_metrics, mask_seeds, ...
    spatter_seeds, tensor_init_seeds, matrix_init_seeds, execution_seeds)

    num_algorithms = numel(algorithm_names);
    num_trials = numel(mask_seeds);
    num_ratios = numel(observation_ratios);
    num_rows = num_algorithms * num_trials * num_ratios;

    Algorithm = cell(num_rows, 1);
    Trial = zeros(num_rows, 1);
    ObservationRatio = zeros(num_rows, 1);
    Status = cell(num_rows, 1);
    MaskSeed = zeros(num_rows, 1);
    SpatterSeed = zeros(num_rows, 1);
    TensorInitializationSeed = zeros(num_rows, 1);
    MatrixInitializationSeed = zeros(num_rows, 1);
    ExecutionSeed = zeros(num_rows, 1);
    ActualObservationRatio = NaN(num_rows, 1);
    RealizedSpatterDensity = NaN(num_rows, 1);
    EffectiveChangedPixelDensity = NaN(num_rows, 1);
    HuberDelta = NaN(num_rows, 1);
    AccuracyEvaluationRuntimeSeconds = NaN(num_rows, 1);
    PostCalibrationMeanReferenceNRE = NaN(num_rows, 1);
    FinalFrameReferenceNRE = NaN(num_rows, 1);
    LateWindowMeanReferenceNRE = NaN(num_rows, 1);

    row = 0;
    for ratio_idx = 1:num_ratios
        for trial = 1:num_trials
            for algorithm_idx = 1:num_algorithms
                row = row + 1;
                Algorithm{row} = algorithm_names{algorithm_idx};
                Trial(row) = trial;
                ObservationRatio(row) = observation_ratios(ratio_idx);
                Status{row} = status_label(state.algorithm_status( ...
                    algorithm_idx, trial, ratio_idx));
                MaskSeed(row) = mask_seeds(trial);
                SpatterSeed(row) = spatter_seeds(trial);
                TensorInitializationSeed(row) = tensor_init_seeds(trial);
                MatrixInitializationSeed(row) = matrix_init_seeds(trial);
                ExecutionSeed(row) = execution_seeds( ...
                    trial, algorithm_idx);
                ActualObservationRatio(row) = ...
                    state.actual_observation_ratio(trial, ratio_idx);
                RealizedSpatterDensity(row) = ...
                    state.realized_spatter_density(trial);
                EffectiveChangedPixelDensity(row) = ...
                    state.effective_changed_density(trial);
                HuberDelta(row) = state.huber_delta(trial, ratio_idx);
                AccuracyEvaluationRuntimeSeconds(row) = ...
                    state.algorithm_runtime_seconds( ...
                    algorithm_idx, trial, ratio_idx);
                PostCalibrationMeanReferenceNRE(row) = ...
                    scalar_metrics.PostCalibrationMeanReferenceNRE( ...
                    algorithm_idx, trial, ratio_idx);
                FinalFrameReferenceNRE(row) = ...
                    scalar_metrics.FinalFrameReferenceNRE( ...
                    algorithm_idx, trial, ratio_idx);
                LateWindowMeanReferenceNRE(row) = ...
                    scalar_metrics.LateWindowMeanReferenceNRE( ...
                    algorithm_idx, trial, ratio_idx);
            end
        end
    end

    output_table = table(Algorithm, Trial, ObservationRatio, Status, ...
        MaskSeed, SpatterSeed, TensorInitializationSeed, ...
        MatrixInitializationSeed, ExecutionSeed, ...
        ActualObservationRatio, RealizedSpatterDensity, ...
        EffectiveChangedPixelDensity, HuberDelta, ...
        AccuracyEvaluationRuntimeSeconds, ...
        PostCalibrationMeanReferenceNRE, FinalFrameReferenceNRE, ...
        LateWindowMeanReferenceNRE);
end

function summary_table = build_r3_summary_table(algorithm_names, ...
    observation_ratios, algorithm_status, scalar_metrics, metric_names, ...
    ci_level, num_resamples, bootstrap_seed, ci_method)

    num_rows = numel(algorithm_names) * numel(observation_ratios) * ...
        numel(metric_names);
    Algorithm = cell(num_rows, 1);
    ObservationRatio = zeros(num_rows, 1);
    Metric = cell(num_rows, 1);
    Mean = NaN(num_rows, 1);
    StandardDeviation = NaN(num_rows, 1);
    CILower = NaN(num_rows, 1);
    CIUpper = NaN(num_rows, 1);
    ValidTrials = zeros(num_rows, 1);
    FailedTrials = zeros(num_rows, 1);
    PendingTrials = zeros(num_rows, 1);
    TotalTrials = zeros(num_rows, 1);
    CILevel = repmat(ci_level, num_rows, 1);
    BootstrapResamples = repmat(num_resamples, num_rows, 1);
    BootstrapSeed = zeros(num_rows, 1);
    CIMethod = repmat({ci_method}, num_rows, 1);

    row = 0;
    for metric_idx = 1:numel(metric_names)
        metric_name = metric_names{metric_idx};
        metric_values = scalar_metrics.(metric_name);
        for ratio_idx = 1:numel(observation_ratios)
            for algorithm_idx = 1:numel(algorithm_names)
                row = row + 1;
                values = reshape(metric_values( ...
                    algorithm_idx, :, ratio_idx), [], 1);
                statuses = reshape(algorithm_status( ...
                    algorithm_idx, :, ratio_idx), [], 1);
                valid = statuses == 1 & isfinite(values);
                valid_values = values(valid);

                Algorithm{row} = algorithm_names{algorithm_idx};
                ObservationRatio(row) = observation_ratios(ratio_idx);
                Metric{row} = metric_name;
                ValidTrials(row) = nnz(valid);
                FailedTrials(row) = nnz(statuses == 2);
                PendingTrials(row) = nnz(statuses == 0);
                TotalTrials(row) = numel(statuses);
                BootstrapSeed(row) = bootstrap_seed + row - 1;

                if ~isempty(valid_values)
                    Mean(row) = mean(valid_values);
                    if numel(valid_values) >= 2
                        StandardDeviation(row) = std(valid_values);
                    end
                end
                [CILower(row), CIUpper(row)] = bootstrap_mean_ci( ...
                    valid_values, num_resamples, ci_level, ...
                    BootstrapSeed(row));
            end
        end
    end

    summary_table = table(Algorithm, ObservationRatio, Metric, Mean, ...
        StandardDeviation, CILower, CIUpper, ValidTrials, FailedTrials, ...
        PendingTrials, TotalTrials, CILevel, BootstrapResamples, ...
        BootstrapSeed, CIMethod);
end

function comparison_table = build_r3_paired_comparison(algorithm_names, ...
    observation_ratios, algorithm_status, scalar_metrics, metric_names, ...
    reference_algorithm_name, ci_level, num_resamples, bootstrap_seed, ...
    ci_method)

    reference_idx = find(strcmp(algorithm_names, ...
        reference_algorithm_name), 1);
    if isempty(reference_idx)
        error('Exp_R3:MissingReferenceAlgorithm', ...
            'Paired comparison requires %s.', reference_algorithm_name);
    end
    baseline_indices = find(~strcmp(algorithm_names, ...
        reference_algorithm_name));
    num_rows = numel(baseline_indices) * ...
        numel(observation_ratios) * numel(metric_names);

    BaselineAlgorithm = cell(num_rows, 1);
    ReferenceAlgorithm = repmat({reference_algorithm_name}, num_rows, 1);
    ObservationRatio = zeros(num_rows, 1);
    Metric = cell(num_rows, 1);
    TotalTrials = zeros(num_rows, 1);
    BaselineValidTrials = zeros(num_rows, 1);
    ReferenceValidTrials = zeros(num_rows, 1);
    ValidPairs = zeros(num_rows, 1);
    BaselineFailures = zeros(num_rows, 1);
    ReferenceFailures = zeros(num_rows, 1);
    MeanBaselineNRE = NaN(num_rows, 1);
    MeanReferenceNRE = NaN(num_rows, 1);
    MeanBaselineMinusReferenceNRE = NaN(num_rows, 1);
    MedianBaselineMinusReferenceNRE = NaN(num_rows, 1);
    BaselineMinusReferenceCILower = NaN(num_rows, 1);
    BaselineMinusReferenceCIUpper = NaN(num_rows, 1);
    ReferenceLowerNREFraction = NaN(num_rows, 1);
    CILevel = repmat(ci_level, num_rows, 1);
    BootstrapResamples = repmat(num_resamples, num_rows, 1);
    BootstrapSeed = zeros(num_rows, 1);
    CIMethod = repmat({ci_method}, num_rows, 1);

    row = 0;
    for metric_idx = 1:numel(metric_names)
        metric_name = metric_names{metric_idx};
        metric_values = scalar_metrics.(metric_name);
        for ratio_idx = 1:numel(observation_ratios)
            reference_values = reshape(metric_values( ...
                reference_idx, :, ratio_idx), [], 1);
            reference_status = reshape(algorithm_status( ...
                reference_idx, :, ratio_idx), [], 1);
            reference_valid = reference_status == 1 & ...
                isfinite(reference_values);

            for baseline_idx = baseline_indices
                row = row + 1;
                baseline_values = reshape(metric_values( ...
                    baseline_idx, :, ratio_idx), [], 1);
                baseline_status = reshape(algorithm_status( ...
                    baseline_idx, :, ratio_idx), [], 1);
                baseline_valid = baseline_status == 1 & ...
                    isfinite(baseline_values);
                paired_valid = baseline_valid & reference_valid;
                differences = baseline_values(paired_valid) - ...
                    reference_values(paired_valid);

                BaselineAlgorithm{row} = algorithm_names{baseline_idx};
                ObservationRatio(row) = observation_ratios(ratio_idx);
                Metric{row} = metric_name;
                TotalTrials(row) = numel(reference_values);
                BaselineValidTrials(row) = nnz(baseline_valid);
                ReferenceValidTrials(row) = nnz(reference_valid);
                ValidPairs(row) = nnz(paired_valid);
                BaselineFailures(row) = nnz(baseline_status == 2);
                ReferenceFailures(row) = nnz(reference_status == 2);
                BootstrapSeed(row) = bootstrap_seed + row - 1;

                if ~isempty(differences)
                    MeanBaselineNRE(row) = ...
                        mean(baseline_values(paired_valid));
                    MeanReferenceNRE(row) = ...
                        mean(reference_values(paired_valid));
                    MeanBaselineMinusReferenceNRE(row) = ...
                        mean(differences);
                    MedianBaselineMinusReferenceNRE(row) = ...
                        median(differences);
                    ReferenceLowerNREFraction(row) = ...
                        mean(differences > 0);
                end
                [BaselineMinusReferenceCILower(row), ...
                    BaselineMinusReferenceCIUpper(row)] = ...
                    bootstrap_mean_ci(differences, num_resamples, ...
                    ci_level, BootstrapSeed(row));
            end
        end
    end

    comparison_table = table(BaselineAlgorithm, ReferenceAlgorithm, ...
        ObservationRatio, Metric, TotalTrials, BaselineValidTrials, ...
        ReferenceValidTrials, ValidPairs, BaselineFailures, ...
        ReferenceFailures, MeanBaselineNRE, MeanReferenceNRE, ...
        MeanBaselineMinusReferenceNRE, ...
        MedianBaselineMinusReferenceNRE, ...
        BaselineMinusReferenceCILower, ...
        BaselineMinusReferenceCIUpper, ReferenceLowerNREFraction, ...
        CILevel, BootstrapResamples, BootstrapSeed, CIMethod);
end

function [per_trial_table, summary_table, overhead_table] = ...
    build_r3_timing_tables(method_names, observation_ratio, ratio_idx, ...
    state, total_slices, mask_seeds, spatter_seeds, tensor_init_seeds, ...
    ci_level, num_resamples, bootstrap_seed, ci_method)

    num_methods = numel(method_names);
    num_trials = numel(mask_seeds);
    olstec_idx = find(strcmp(method_names, 'OLSTEC_80'), 1);
    rsi_idx = find(strcmp(method_names, 'RSI_OLSTEC'), 1);
    if num_methods ~= 2 || isempty(olstec_idx) || isempty(rsi_idx)
        error('Exp_R3:InvalidTimingMethods', ...
            'Timing summaries require OLSTEC_80 and RSI_OLSTEC.');
    end

    wall_seconds = state.timing_wall_seconds;
    core_seconds = state.timing_core_seconds;
    if ~isequal(size(wall_seconds), [num_methods, num_trials]) || ...
            ~isequal(size(core_seconds), [num_methods, num_trials]) || ...
            any(~isfinite(wall_seconds(:))) || ...
            any(~isfinite(core_seconds(:))) || ...
            any(wall_seconds(:) <= 0) || any(core_seconds(:) <= 0)
        error('Exp_R3:InvalidTimingResult', ...
            'Timing tables require complete finite positive paired results.');
    end

    wall_fps = total_slices ./ wall_seconds;
    core_fps = total_slices ./ core_seconds;
    Trial = (1:num_trials)';
    ObservationRatio = repmat(observation_ratio, num_trials, 1);
    ExecutionOrder = cell(num_trials, 1);
    for trial = 1:num_trials
        if mod(trial, 2) == 1
            ExecutionOrder{trial} = 'OLSTEC_80_then_RSI_OLSTEC';
        else
            ExecutionOrder{trial} = 'RSI_OLSTEC_then_OLSTEC_80';
        end
    end
    MaskSeed = mask_seeds(:);
    SpatterSeed = spatter_seeds(:);
    TensorInitializationSeed = tensor_init_seeds(:);
    ActualObservationRatio = ...
        state.actual_observation_ratio(:, ratio_idx);
    HuberDelta = state.huber_delta(:, ratio_idx);
    OLSTECWallTimeSeconds = wall_seconds(olstec_idx, :)';
    RSIWallTimeSeconds = wall_seconds(rsi_idx, :)';
    OLSTECCoreTimeSeconds = core_seconds(olstec_idx, :)';
    RSICoreTimeSeconds = core_seconds(rsi_idx, :)';
    OLSTECWallFPS = wall_fps(olstec_idx, :)';
    RSIWallFPS = wall_fps(rsi_idx, :)';
    OLSTECCoreFPS = core_fps(olstec_idx, :)';
    RSICoreFPS = core_fps(rsi_idx, :)';
    RSIToOLSTECWallTimeRatio = ...
        RSIWallTimeSeconds ./ OLSTECWallTimeSeconds;
    RSIToOLSTECCoreTimeRatio = ...
        RSICoreTimeSeconds ./ OLSTECCoreTimeSeconds;
    RSIMeanPriorIRLSIterations = ...
        state.timing_prior_irls_iterations;
    RSIMeanPosteriorIRLSIterations = ...
        state.timing_posterior_irls_iterations;

    per_trial_table = table(Trial, ObservationRatio, ExecutionOrder, ...
        MaskSeed, SpatterSeed, TensorInitializationSeed, ...
        ActualObservationRatio, HuberDelta, OLSTECWallTimeSeconds, ...
        RSIWallTimeSeconds, OLSTECCoreTimeSeconds, ...
        RSICoreTimeSeconds, OLSTECWallFPS, RSIWallFPS, ...
        OLSTECCoreFPS, RSICoreFPS, RSIToOLSTECWallTimeRatio, ...
        RSIToOLSTECCoreTimeRatio, RSIMeanPriorIRLSIterations, ...
        RSIMeanPosteriorIRLSIterations);

    Algorithm = method_names(:);
    Trials = repmat(num_trials, num_methods, 1);
    WallTimeMedianSeconds = NaN(num_methods, 1);
    WallTimeQ1Seconds = NaN(num_methods, 1);
    WallTimeQ3Seconds = NaN(num_methods, 1);
    CoreTimeMedianSeconds = NaN(num_methods, 1);
    CoreTimeQ1Seconds = NaN(num_methods, 1);
    CoreTimeQ3Seconds = NaN(num_methods, 1);
    WallFPSMedian = NaN(num_methods, 1);
    WallFPSQ1 = NaN(num_methods, 1);
    WallFPSQ3 = NaN(num_methods, 1);
    CoreFPSMedian = NaN(num_methods, 1);
    CoreFPSQ1 = NaN(num_methods, 1);
    CoreFPSQ3 = NaN(num_methods, 1);

    for method_idx = 1:num_methods
        method_wall = wall_seconds(method_idx, :)';
        method_core = core_seconds(method_idx, :)';
        method_wall_fps = wall_fps(method_idx, :)';
        method_core_fps = core_fps(method_idx, :)';
        WallTimeMedianSeconds(method_idx) = median(method_wall);
        WallTimeQ1Seconds(method_idx) = linear_quantile(method_wall, 0.25);
        WallTimeQ3Seconds(method_idx) = linear_quantile(method_wall, 0.75);
        CoreTimeMedianSeconds(method_idx) = median(method_core);
        CoreTimeQ1Seconds(method_idx) = linear_quantile(method_core, 0.25);
        CoreTimeQ3Seconds(method_idx) = linear_quantile(method_core, 0.75);
        WallFPSMedian(method_idx) = median(method_wall_fps);
        WallFPSQ1(method_idx) = linear_quantile(method_wall_fps, 0.25);
        WallFPSQ3(method_idx) = linear_quantile(method_wall_fps, 0.75);
        CoreFPSMedian(method_idx) = median(method_core_fps);
        CoreFPSQ1(method_idx) = linear_quantile(method_core_fps, 0.25);
        CoreFPSQ3(method_idx) = linear_quantile(method_core_fps, 0.75);
    end

    summary_table = table(Algorithm, Trials, WallTimeMedianSeconds, ...
        WallTimeQ1Seconds, WallTimeQ3Seconds, CoreTimeMedianSeconds, ...
        CoreTimeQ1Seconds, CoreTimeQ3Seconds, WallFPSMedian, ...
        WallFPSQ1, WallFPSQ3, CoreFPSMedian, CoreFPSQ1, CoreFPSQ3);

    [wall_ratio, wall_ci_lower, wall_ci_upper] = ...
        paired_geometric_mean_ratio( ...
        wall_seconds(rsi_idx, :)', wall_seconds(olstec_idx, :)', ...
        num_resamples, ci_level, bootstrap_seed);
    [core_ratio, core_ci_lower, core_ci_upper] = ...
        paired_geometric_mean_ratio( ...
        core_seconds(rsi_idx, :)', core_seconds(olstec_idx, :)', ...
        num_resamples, ci_level, bootstrap_seed + 1);

    Comparison = {'RSI_OLSTEC / OLSTEC_80'};
    ValidPairs = num_trials;
    WallTimeGeometricMeanRatio = wall_ratio;
    WallTimeRatioCILower = wall_ci_lower;
    WallTimeRatioCIUpper = wall_ci_upper;
    CoreTimeGeometricMeanRatio = core_ratio;
    CoreTimeRatioCILower = core_ci_lower;
    CoreTimeRatioCIUpper = core_ci_upper;
    CILevel = ci_level;
    BootstrapResamples = num_resamples;
    WallTimeBootstrapSeed = bootstrap_seed;
    CoreTimeBootstrapSeed = bootstrap_seed + 1;
    CIMethod = {ci_method};
    overhead_table = table(Comparison, ValidPairs, ...
        WallTimeGeometricMeanRatio, WallTimeRatioCILower, ...
        WallTimeRatioCIUpper, CoreTimeGeometricMeanRatio, ...
        CoreTimeRatioCILower, CoreTimeRatioCIUpper, CILevel, ...
        BootstrapResamples, WallTimeBootstrapSeed, ...
        CoreTimeBootstrapSeed, CIMethod);
end

function [ci_lower, ci_upper] = bootstrap_mean_ci( ...
    values, num_resamples, ci_level, bootstrap_seed)

    values = values(:);
    if numel(values) < 2
        ci_lower = NaN;
        ci_upper = NaN;
        return;
    end

    stream = RandStream('mt19937ar', 'Seed', bootstrap_seed);
    sample_indices = randi(stream, numel(values), ...
        numel(values), num_resamples);
    bootstrap_means = mean(values(sample_indices), 1);
    tail_probability = (1 - ci_level) / 2;
    ci_lower = linear_quantile(bootstrap_means, tail_probability);
    ci_upper = linear_quantile(bootstrap_means, 1 - tail_probability);
end

function [estimate, ci_lower, ci_upper] = ...
    paired_geometric_mean_ratio(numerator, denominator, ...
    num_resamples, ci_level, bootstrap_seed)

    numerator = numerator(:);
    denominator = denominator(:);
    valid = isfinite(numerator) & numerator > 0 & ...
        isfinite(denominator) & denominator > 0;
    if ~all(valid) || ~any(valid)
        error('Exp_R3:InvalidPairedRuntime', ...
            'Paired runtime analysis requires finite positive pairs.');
    end

    log_ratios = log(numerator ./ denominator);
    estimate = exp(mean(log_ratios));
    if numel(log_ratios) < 2
        ci_lower = NaN;
        ci_upper = NaN;
        return;
    end

    stream = RandStream('mt19937ar', 'Seed', bootstrap_seed);
    sample_indices = randi(stream, numel(log_ratios), ...
        numel(log_ratios), num_resamples);
    bootstrap_estimates = exp(mean(log_ratios(sample_indices), 1));
    tail_probability = (1 - ci_level) / 2;
    ci_lower = linear_quantile(bootstrap_estimates, tail_probability);
    ci_upper = linear_quantile(bootstrap_estimates, ...
        1 - tail_probability);
end

function value = linear_quantile(samples, probability)
    sorted_samples = sort(samples(:));
    position = 1 + (numel(sorted_samples) - 1) * probability;
    lower_index = floor(position);
    upper_index = ceil(position);
    if lower_index == upper_index
        value = sorted_samples(lower_index);
    else
        interpolation_weight = position - lower_index;
        value = sorted_samples(lower_index) + interpolation_weight * ...
            (sorted_samples(upper_index) - sorted_samples(lower_index));
    end
end

function records = empty_failure_records()
    records = struct( ...
        'Algorithm', {}, ...
        'Trial', {}, ...
        'ObservationRatio', {}, ...
        'MaskSeed', {}, ...
        'SpatterSeed', {}, ...
        'TensorInitializationSeed', {}, ...
        'MatrixInitializationSeed', {}, ...
        'ExecutionSeed', {}, ...
        'Category', {}, ...
        'Identifier', {}, ...
        'Message', {}, ...
        'FirstNonfiniteFrame', {}, ...
        'NonfiniteCount', {});
end

function record = make_failure_record(algorithm_name, trial, ...
    observation_ratio, mask_seed, spatter_seed, tensor_init_seed, ...
    matrix_init_seed, execution_seed, exception)

    identifier = exception.identifier;
    if isempty(identifier)
        identifier = 'unidentified_exception';
    end
    message_text = regexprep(strtrim(exception.message), ...
        '[\r\n]+', ' | ');
    [first_nonfinite_frame, nonfinite_count] = ...
        parse_nonfinite_diagnostic(message_text);

    record = struct( ...
        'Algorithm', algorithm_name, ...
        'Trial', trial, ...
        'ObservationRatio', observation_ratio, ...
        'MaskSeed', mask_seed, ...
        'SpatterSeed', spatter_seed, ...
        'TensorInitializationSeed', tensor_init_seed, ...
        'MatrixInitializationSeed', matrix_init_seed, ...
        'ExecutionSeed', execution_seed, ...
        'Category', classify_failure(identifier, message_text), ...
        'Identifier', identifier, ...
        'Message', message_text, ...
        'FirstNonfiniteFrame', first_nonfinite_frame, ...
        'NonfiniteCount', nonfinite_count);
end

function [first_frame, count] = parse_nonfinite_diagnostic(message_text)
    first_frame = NaN;
    count = NaN;
    tokens = regexp(message_text, ...
        'first frame=(\d+); count=(\d+)', 'tokens', 'once');
    if ~isempty(tokens)
        first_frame = str2double(tokens{1});
        count = str2double(tokens{2});
    end
end

function category = classify_failure(identifier, message_text)
    diagnostic_text = lower([identifier, ' ', message_text]);
    if contains(diagnostic_text, 'nonfinitemetric') || ...
            contains(diagnostic_text, 'non-finite') || ...
            contains(diagnostic_text, 'invalidmetric') || ...
            contains(diagnostic_text, 'reconstructionsizemismatch') || ...
            contains(diagnostic_text, 'missingreconstruction')
        category = 'invalid_output_or_metric';
    elseif contains(diagnostic_text, 'nomem') || ...
            contains(diagnostic_text, 'out of memory') || ...
            contains(diagnostic_text, 'memory allocation')
        category = 'memory_or_resource';
    elseif contains(diagnostic_text, 'singular') || ...
            contains(diagnostic_text, 'rank deficient') || ...
            contains(diagnostic_text, 'ill-conditioned') || ...
            contains(diagnostic_text, 'positive definite')
        category = 'numerical_linear_algebra';
    else
        category = 'runtime_exception';
    end
end

function value = is_process_interruption(exception)
    diagnostic_text = lower([exception.identifier, ' ', exception.message]);
    value = contains(diagnostic_text, 'operationterminatedbyuser') || ...
        contains(diagnostic_text, 'terminated by user') || ...
        contains(diagnostic_text, 'operation terminated') || ...
        contains(diagnostic_text, 'nomem') || ...
        contains(diagnostic_text, 'outofmemory') || ...
        contains(diagnostic_text, 'out of memory') || ...
        contains(diagnostic_text, 'memory allocation') || ...
        contains(diagnostic_text, 'requested array exceeds') || ...
        contains(diagnostic_text, 'unable to allocate');
end

function failure_summary_table = build_r3_failure_summary( ...
    failure_records, algorithm_names, observation_ratios)

    Algorithm = cell(0, 1);
    ObservationRatio = zeros(0, 1);
    Category = cell(0, 1);
    Count = zeros(0, 1);

    for algorithm_idx = 1:numel(algorithm_names)
        for ratio_idx = 1:numel(observation_ratios)
            matching = strcmp({failure_records.Algorithm}, ...
                algorithm_names{algorithm_idx}) & ...
                abs([failure_records.ObservationRatio] - ...
                observation_ratios(ratio_idx)) < 1e-12;
            categories = unique({failure_records(matching).Category});
            for category_idx = 1:numel(categories)
                Algorithm{end+1, 1} = ...
                    algorithm_names{algorithm_idx}; %#ok<AGROW>
                ObservationRatio(end+1, 1) = ...
                    observation_ratios(ratio_idx); %#ok<AGROW>
                Category{end+1, 1} = categories{category_idx}; %#ok<AGROW>
                Count(end+1, 1) = nnz(matching & ...
                    strcmp({failure_records.Category}, ...
                    categories{category_idx})); %#ok<AGROW>
            end
        end
    end

    failure_summary_table = table(Algorithm, ObservationRatio, ...
        Category, Count);
end

function label = status_label(status_code)
    switch status_code
        case 0
            label = 'pending';
        case 1
            label = 'successful';
        case 2
            label = 'failed';
        otherwise
            error('Exp_R3:InvalidStatusCode', ...
                'Unknown algorithm status code: %d.', status_code);
    end
end

function print_primary_metric_summary(summary_table, primary_metric)
    fprintf('\nPrimary metric: %s\n', primary_metric);
    fprintf('%-15s %-8s %-14s %-14s %-9s %-9s\n', ...
        'Algorithm', 'rho', 'Mean', 'Std', 'Valid', 'Failed');
    rows = strcmp(summary_table.Metric, primary_metric);
    selected = summary_table(rows, :);
    for row = 1:height(selected)
        fprintf('%-15s %-8.2f %-14.6f %-14.6f %-9d %-9d\n', ...
            selected.Algorithm{row}, selected.ObservationRatio(row), ...
            selected.Mean(row), selected.StandardDeviation(row), ...
            selected.ValidTrials(row), selected.FailedTrials(row));
    end
end

function plot_r3_mean_trajectories(reference_nre, algorithm_status, ...
    algorithm_names, observation_ratios, selected_rho, ...
    evaluation_start_frame, use_running_average, figure_visibility, result_dir, ...
    export_results, output_filename)

    ratio_idx = find(abs(observation_ratios - selected_rho) < 1e-12, 1);
    if isempty(ratio_idx)
        error('Exp_R3:PlotRatioMissing', ...
            'Selected plotting ratio is not configured.');
    end

    total_slices = size(reference_nre, 4);
    num_trials = size(reference_nre, 2);
    if evaluation_start_frame < 1 || evaluation_start_frame > total_slices
        error('Exp_R3:InvalidPlotEvaluationStart', ...
            'evaluation_start_frame must lie within the sequence.');
    end
    if use_running_average
        x_axis = evaluation_start_frame:total_slices;
    else
        x_axis = 1:total_slices;
    end
    fig = figure('Visible', figure_visibility, ...
        'Name', 'R3 Monte Carlo Reference NRE');
    hold on;
    grid on;
    box on;
    line_handles = gobjects(0);
    legend_entries = {};

    for algorithm_idx = 1:numel(algorithm_names)
        trial_curves = reshape(reference_nre( ...
            algorithm_idx, :, ratio_idx, :), ...
            [num_trials, total_slices]);
        statuses = reshape(algorithm_status( ...
            algorithm_idx, :, ratio_idx), [], 1);
        trial_curves(statuses ~= 1, :) = NaN;

        if use_running_average
            trial_curves = trial_curves(:, evaluation_start_frame:end);
            running_denominator = 1:numel(x_axis);
            for trial = 1:num_trials
                if all(isfinite(trial_curves(trial, :)))
                    trial_curves(trial, :) = ...
                        cumsum(trial_curves(trial, :)) ./ ...
                        running_denominator;
                end
            end
        end

        mean_curve = finite_column_mean(trial_curves);
        std_curve = finite_column_std(trial_curves);
        if ~any(isfinite(mean_curve))
            continue;
        end

        color = algorithm_color(algorithm_names{algorithm_idx});
        positive_values = trial_curves( ...
            isfinite(trial_curves) & trial_curves > 0);
        if isempty(positive_values)
            plot_floor = eps;
        else
            plot_floor = max(eps, 0.5 * min(positive_values));
        end
        lower_curve = max(mean_curve - std_curve, plot_floor);
        upper_curve = max(mean_curve + std_curve, plot_floor);
        valid_band = isfinite(lower_curve) & isfinite(upper_curve);
        if any(valid_band)
            band_x = x_axis(valid_band);
            fill([band_x, fliplr(band_x)], ...
                [lower_curve(valid_band), fliplr(upper_curve(valid_band))], ...
                color, 'FaceAlpha', 0.08, 'EdgeColor', 'none', ...
                'HandleVisibility', 'off');
        end
        line_handle = semilogy(x_axis, max(mean_curve, plot_floor), ...
            'Color', color, 'LineWidth', algorithm_line_width( ...
            algorithm_names{algorithm_idx}), ...
            'LineStyle', algorithm_line_style( ...
            algorithm_names{algorithm_idx}));
        line_handles(end+1) = line_handle; %#ok<AGROW>
        legend_entries{end+1} = ...
            algorithm_display_name(algorithm_names{algorithm_idx}); %#ok<AGROW>
    end

    xlabel('Frame Index');
    if use_running_average
        ylabel('Post-Calibration Running-Average Reference NRE');
    else
        ylabel('Reference NRE');
    end
    title(sprintf('Observation Ratio: \\rho=%.2f', selected_rho));
    xlim([x_axis(1), x_axis(end)]);
    set(gca, 'FontSize', 12);
    if ~isempty(line_handles)
        legend(line_handles, legend_entries, 'Location', 'best', ...
            'FontSize', 9, 'Interpreter', 'tex');
    end

    if export_results
        save_reopenable_figure(fig, ...
            fullfile(result_dir, output_filename));
    end
    if strcmp(figure_visibility, 'off')
        close(fig);
    end
end

function plot_r3_representative_frames(input_frames, reconstructions, ...
    algorithm_names, visual_algorithm_names, frame_indices, ...
    representative_trial, representative_rho, figure_visibility, ...
    result_dir, export_results)

    num_columns = numel(frame_indices);
    num_rows = numel(visual_algorithm_names) + 1;
    fig = figure('Visible', figure_visibility, ...
        'Name', 'R3 Prespecified Representative Trial', ...
        'Position', [50, 50, 220 * num_columns, 190 * num_rows]);
    tiledlayout(num_rows, num_columns, ...
        'TileSpacing', 'none', 'Padding', 'compact');

    for column = 1:num_columns
        nexttile;
        imagesc(input_frames(:, :, column));
        local_set_clim([0, 1]);
        colormap(gray);
        axis image;
        set(gca, 'XTick', [], 'YTick', [], 'Box', 'off');
        title(sprintf('Frame %d', frame_indices(column)), ...
            'FontWeight', 'bold');
        if column == 1
            ylabel('Injected input', 'FontWeight', 'bold');
        end
    end

    for visual_idx = 1:numel(visual_algorithm_names)
        algorithm_name = visual_algorithm_names{visual_idx};
        algorithm_idx = find(strcmp(algorithm_names, algorithm_name), 1);
        for column = 1:num_columns
            nexttile;
            if isempty(algorithm_idx)
                imagesc(zeros(size(input_frames, 1), ...
                    size(input_frames, 2)));
                text(0.5, 0.5, 'Not configured', ...
                    'Units', 'normalized', 'HorizontalAlignment', 'center');
            else
                frame = reconstructions(:, :, column, algorithm_idx);
                if any(isfinite(frame(:)))
                    imagesc(frame);
                else
                    imagesc(zeros(size(input_frames, 1), ...
                        size(input_frames, 2)));
                    text(0.5, 0.5, 'Unavailable', ...
                        'Units', 'normalized', ...
                        'HorizontalAlignment', 'center');
                end
            end
            local_set_clim([0, 1]);
            colormap(gray);
            axis image;
            set(gca, 'XTick', [], 'YTick', [], 'Box', 'off');
            if column == 1
                ylabel(algorithm_display_name(algorithm_name), ...
                    'FontWeight', 'bold', 'Interpreter', 'tex');
            end
        end
    end
    sgtitle(sprintf(['Prespecified Trial %d, Observation Ratio ' ...
        '\\rho=%.2f'], representative_trial, representative_rho));

    if export_results
        save_reopenable_figure(fig, fullfile(result_dir, ...
            'Fig_MultiFrame_Matrix_R3.fig'));
    end
    if strcmp(figure_visibility, 'off')
        close(fig);
    end
end

function save_reopenable_figure(fig, filename)
    original_visibility = fig.Visible;
    fig.Visible = 'on';
    savefig(fig, filename);
    fig.Visible = original_visibility;
end

function value = finite_mean(values)
    values = values(isfinite(values));
    if isempty(values)
        value = NaN;
    else
        value = mean(values);
    end
end

function values = finite_column_mean(matrix)
    valid = isfinite(matrix);
    count = sum(valid, 1);
    matrix(~valid) = 0;
    values = sum(matrix, 1) ./ count;
    values(count == 0) = NaN;
end

function values = finite_column_std(matrix)
    means = finite_column_mean(matrix);
    valid = isfinite(matrix);
    count = sum(valid, 1);
    centered = matrix - means;
    centered(~valid) = 0;
    denominator = max(count - 1, 1);
    values = sqrt(sum(centered .^ 2, 1) ./ denominator);
    values(count < 2) = NaN;
end

function color = algorithm_color(algorithm_name)
    switch algorithm_name
        case 'CP_WOPT'
            color = [0.15, 0.15, 0.15];
        case 'Petrels'
            color = [0.55, 0.20, 0.65];
        case 'Grasta'
            color = [0.85, 0.55, 0.05];
        case 'Grouse'
            color = [0.20, 0.60, 0.25];
        case 'TeCPSGD'
            color = [0.10, 0.35, 0.80];
        case 'OLSTEC_70'
            color = [0.20, 0.70, 0.75];
        case 'OLSTEC_80'
            color = [0.05, 0.55, 0.65];
        case 'OLSTEC_90'
            color = [0.00, 0.40, 0.55];
        case 'OLSTEC_99'
            color = [0.00, 0.25, 0.40];
        case 'RSI_OLSTEC'
            color = [0.85, 0.10, 0.10];
        otherwise
            color = [0.40, 0.40, 0.40];
    end
end

function style = algorithm_line_style(algorithm_name)
    switch algorithm_name
        case 'CP_WOPT'
            style = '--';
        case 'OLSTEC_70'
            style = ':';
        case 'OLSTEC_80'
            style = '--';
        case 'OLSTEC_90'
            style = '-';
        case 'OLSTEC_99'
            style = '-.';
        otherwise
            style = '-';
    end
end

function width = algorithm_line_width(algorithm_name)
    if strcmp(algorithm_name, 'RSI_OLSTEC')
        width = 2.4;
    else
        width = 1.6;
    end
end

function name = algorithm_display_name(algorithm_name)
    if startsWith(algorithm_name, 'OLSTEC_')
        lambda = str2double(algorithm_name(8:end)) / 100;
        name = sprintf('OLSTEC (\\lambda=%.2f)', lambda);
    elseif strcmp(algorithm_name, 'RSI_OLSTEC')
        name = 'RSI-OLSTEC';
    elseif strcmp(algorithm_name, 'CP_WOPT')
        name = 'CP-WOPT (batch)';
    elseif strcmp(algorithm_name, 'Petrels')
        name = 'PETRELS';
    elseif strcmp(algorithm_name, 'Grasta')
        name = 'GRASTA';
    elseif strcmp(algorithm_name, 'Grouse')
        name = 'GROUSE';
    else
        name = algorithm_name;
    end
end

function local_set_clim(limits)
    if exist('clim', 'file') == 2 || exist('clim', 'builtin') == 5
        clim(limits);
    else
        set(gca, 'CLim', limits);
    end
end
