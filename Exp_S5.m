%% Experiment S5: Side-information reliability
% Paired Monte Carlo evaluation of RSI-OLSTEC under controlled
% side-information degradations.
clear; clc; close all;

repo_root = fileparts(mfilename('fullpath'));
addpath(repo_root);
addpath(genpath(fullfile(repo_root, 'tool')));
addpath(genpath(fullfile(repo_root, 'auxiliary')));
addpath(genpath(fullfile(repo_root, 'benchmark')));

%% 1. Configuration
num_trials = 50;
trial_seed_ids = (1001:(1000 + num_trials))';

rows = 100;
cols = 100;
total_frames = 1500;
tensor_dims = [rows, cols, total_frames];
rank_r = 5;
mutation_scale = 0.50;
transition_boundary_range = [400, 800];
snr_db = 25;
spatter_density = 0.05;
spatter_base_magnitude = 0.50;
observation_ratio = 0.50;

lambda_min = 0.10;
lambda_max = 0.80;
fixed_lambda = 0.80;
mu = 0.01;
grad_ema_alpha = 0.999;
irls_max_iters = 3;
irls_tolerance = 1e-3;
normalization_epsilon = 1e-3;

huber_burn_in_frames = 30;
huber_scale_multiplier = 3;
huber_delta_lower_bound = 0.01;
huber_delta_fallback = 0.05;

side_config.baseline = 10.0;
side_config.event_amplitude = 10.0 * mutation_scale;
side_config.event_decay = 0.80;
side_config.burn_in_frames = 30;
side_config.threshold_sigma_multiplier = 3;
side_config.threshold_floor = 0.05;
% Event offsets are referenced to the first post-mutation frame.

window_pre = 100;
window_post = 300;
evaluation_length = window_pre + window_post + 1;
relative_frames = (-window_pre:window_post)';
zero_index = window_pre + 1;

ci_level = 0.95;
bootstrap_resamples = 10000;
export_results = true;

conditions = define_s5_conditions();
num_conditions = numel(conditions);
num_runs = num_conditions + 1;
run_ids = [{'fixed_huber'}, {conditions.id}];
run_labels = [{'Huber fixed lambda 0.80'}, {conditions.label}];

metric_names = {'PreNRE50', 'PostPeakNRE50', ...
    'RecoveryIntegral50', 'PostMeanNRE100', ...
    'FirstLambdaReductionLag0To50', ...
    'MinimumLambda0To50', ...
    'CumulativeLambdaReduction0To50', ...
    'PreMutationReducedLambdaFraction'};

result_dir = fullfile(repo_root, 'result', 'S5');
checkpoint_file = fullfile(result_dir, 'S5_checkpoint.mat');
if export_results && ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

settings = struct( ...
    'num_trials', num_trials, ...
    'trial_seed_ids', trial_seed_ids, ...
    'tensor_dims', tensor_dims, ...
    'rank_r', rank_r, ...
    'mutation_scale', mutation_scale, ...
    'transition_boundary_range', transition_boundary_range, ...
    'snr_db', snr_db, ...
    'spatter_density', spatter_density, ...
    'spatter_base_magnitude', spatter_base_magnitude, ...
    'observation_ratio', observation_ratio, ...
    'lambda_min', lambda_min, ...
    'lambda_max', lambda_max, ...
    'fixed_lambda', fixed_lambda, ...
    'mu', mu, ...
    'grad_ema_alpha', grad_ema_alpha, ...
    'irls_max_iters', irls_max_iters, ...
    'irls_tolerance', irls_tolerance, ...
    'normalization_epsilon', normalization_epsilon, ...
    'huber_burn_in_frames', huber_burn_in_frames, ...
    'huber_scale_multiplier', huber_scale_multiplier, ...
    'huber_delta_lower_bound', huber_delta_lower_bound, ...
    'huber_delta_fallback', huber_delta_fallback, ...
    'side_config', side_config, ...
    'window_pre', window_pre, ...
    'window_post', window_post, ...
    'ci_level', ci_level, ...
    'bootstrap_resamples', bootstrap_resamples, ...
    'conditions', conditions);

%% 2. Storage and checkpoint
state.completed = false(num_trials, 1);
state.aligned_nre = NaN(num_runs, num_trials, evaluation_length);
state.aligned_lambda = NaN(num_runs, num_trials, evaluation_length);
state.aligned_side = NaN(num_conditions, num_trials, evaluation_length);
state.metrics = NaN(num_runs, num_trials, numel(metric_names));
state.runtime_seconds = NaN(num_runs, num_trials);
state.transition_boundary = NaN(num_trials, 1);
state.huber_delta = NaN(num_trials, 1);
state.actual_observation_ratio = NaN(num_trials, 1);
state.actual_spatter_density = NaN(num_trials, 1);
state.side_threshold = NaN(num_conditions, num_trials);
state.side_event_delay = NaN(num_conditions, num_trials);
state.side_missing_rate = NaN(num_conditions, num_trials);
state.side_onset_gradient_to_threshold = NaN(num_conditions, num_trials);

if exist(checkpoint_file, 'file')
    loaded = load(checkpoint_file, 'checkpoint');
    if ~isfield(loaded, 'checkpoint') || ...
            ~isequaln(loaded.checkpoint.settings, settings)
        error('Exp_S5:CheckpointMismatch', ...
            'The existing S5 checkpoint uses different settings.');
    end
    state = loaded.checkpoint.state;
    fprintf('Resuming S5 from %d/%d completed trials.\n', ...
        nnz(state.completed), num_trials);
end

%% 3. Paired Monte Carlo experiment
fprintf('Starting S5 with %d paired trials and %d tracking runs per trial.\n', ...
    num_trials, num_runs);

for trial = 1:num_trials
    if state.completed(trial)
        continue;
    end

    fprintf('Trial %d/%d\n', trial, num_trials);
    trial_seed_id = trial_seed_ids(trial);
    visual_seeds = make_visual_seeds(trial_seed_id);
    visual = generate_s5_visual_trial(rows, cols, total_frames, ...
        rank_r, mutation_scale, transition_boundary_range, snr_db, ...
        spatter_density, spatter_base_magnitude, observation_ratio, ...
        huber_burn_in_frames, huber_scale_multiplier, ...
        huber_delta_lower_bound, huber_delta_fallback, visual_seeds);

    start_frame = visual.transition_boundary - window_pre;
    end_frame = visual.transition_boundary + window_post;
    state.transition_boundary(trial) = visual.transition_boundary;
    state.huber_delta(trial) = visual.huber_delta;
    state.actual_observation_ratio(trial) = ...
        nnz(visual.observation_mask) / numel(visual.observation_mask);
    state.actual_spatter_density(trial) = visual.spatter_density;

    % Fixed-memory Huber reference.
    fixed_side = side_config.baseline * ones(total_frames, 1);
    [nre_curve, lambda_curve, pre_mutation_reduced_fraction, elapsed] = ...
        run_s5_tracking_case(visual, tensor_dims, rank_r, fixed_side, Inf, ...
        fixed_lambda, fixed_lambda, mu, grad_ema_alpha, ...
        irls_max_iters, irls_tolerance, normalization_epsilon, ...
        start_frame, end_frame);
    state.aligned_nre(1, trial, :) = reshape(nre_curve, 1, 1, []);
    state.aligned_lambda(1, trial, :) = reshape(lambda_curve, 1, 1, []);
    state.metrics(1, trial, :) = reshape(compute_s5_metrics( ...
        nre_curve, lambda_curve, zero_index, lambda_max, ...
        pre_mutation_reduced_fraction), 1, 1, []);
    state.runtime_seconds(1, trial) = elapsed;

    % RSI-OLSTEC with one side-information condition at a time.
    side_seed = 80000 + trial_seed_id;
    for condition_index = 1:num_conditions
        condition = conditions(condition_index);
        [side_signal, effective_side, side_threshold, side_event_delay, ...
            side_missing_rate, onset_gradient_to_threshold] = ...
            build_s5_side_signal(condition, visual.transition_boundary, ...
            total_frames, side_seed, side_config);

        run_index = condition_index + 1;
        [nre_curve, lambda_curve, ...
            pre_mutation_reduced_fraction, elapsed] = ...
            run_s5_tracking_case(visual, tensor_dims, rank_r, ...
            side_signal, side_threshold, ...
            lambda_min, lambda_max, mu, grad_ema_alpha, ...
            irls_max_iters, irls_tolerance, normalization_epsilon, ...
            start_frame, end_frame);

        state.aligned_nre(run_index, trial, :) = ...
            reshape(nre_curve, 1, 1, []);
        state.aligned_lambda(run_index, trial, :) = ...
            reshape(lambda_curve, 1, 1, []);
        state.aligned_side(condition_index, trial, :) = reshape( ...
            effective_side(start_frame:end_frame), 1, 1, []);
        state.metrics(run_index, trial, :) = reshape( ...
            compute_s5_metrics(nre_curve, lambda_curve, ...
            zero_index, lambda_max, pre_mutation_reduced_fraction), ...
            1, 1, []);
        state.runtime_seconds(run_index, trial) = elapsed;
        state.side_threshold(condition_index, trial) = side_threshold;
        state.side_event_delay(condition_index, trial) = side_event_delay;
        state.side_missing_rate(condition_index, trial) = ...
            side_missing_rate;
        state.side_onset_gradient_to_threshold(condition_index, trial) = ...
            onset_gradient_to_threshold;

        fprintf('  %-28s %.2f s\n', condition.id, elapsed);
    end

    constant_index = find(strcmp({conditions.id}, 'constant'), 1) + 1;
    fixed_nre = reshape(state.aligned_nre(1, trial, :), 1, []);
    constant_nre = reshape( ...
        state.aligned_nre(constant_index, trial, :), 1, []);
    constant_lambda = reshape( ...
        state.aligned_lambda(constant_index, trial, :), 1, []);
    if max(abs(fixed_nre - constant_nre)) > 1e-10 || ...
            max(abs(constant_lambda - fixed_lambda)) > 1e-10
        error('Exp_S5:ConstantControlMismatch', ...
            'The constant side signal must reproduce the fixed-lambda result.');
    end

    state.completed(trial) = true;
    if export_results
        checkpoint = struct('settings', settings, 'state', state);
        save_checkpoint_atomic(checkpoint_file, checkpoint);
        clear checkpoint;
    end
    clear visual;
end

%% 4. Statistical summaries
[per_trial_table, summary_table] = build_s5_tables( ...
    state, conditions, run_ids, run_labels, metric_names, ...
    ci_level, bootstrap_resamples);

fprintf('\nS5 recovery integral over the first 50 post-mutation frames\n');
fprintf('%-30s %12s %12s\n', 'Condition', 'Mean', 'Std');
for run_index = 1:num_runs
    fprintf('%-30s %12.4f %12.4f\n', ...
        summary_table.Label{run_index}, ...
        summary_table.RecoveryIntegral50Mean(run_index), ...
        summary_table.RecoveryIntegral50Std(run_index));
end

if export_results
    writetable(per_trial_table, ...
        fullfile(result_dir, 'S5_per_trial_metrics.csv'));
    writetable(summary_table, ...
        fullfile(result_dir, 'S5_summary_statistics.csv'));
    save(fullfile(result_dir, 'S5_stats.mat'), ...
        'settings', 'conditions', 'run_ids', 'run_labels', ...
        'metric_names', 'state', 'per_trial_table', 'summary_table', ...
        'relative_frames', '-v7');
end

%% 5. Figures
plot_s5_quality_summary(summary_table, conditions, result_dir, ...
    export_results);
plot_s5_selected_trajectories(state, run_ids, ...
    run_labels, relative_frames, result_dir, export_results);

fprintf('S5 outputs written to: %s\n', result_dir);

%% Local functions
function conditions = define_s5_conditions()
    template = new_condition('', '', '');
    conditions = repmat(template, 1, 19);

    conditions(1) = new_condition( ...
        'noise_free_mutation_aligned', ...
        'Noise-free mutation-aligned cue', 'reference');
    conditions(1).noise_sigma = 0;

    conditions(2) = new_condition( ...
        'noisy_mutation_aligned', ...
        'Noisy mutation-aligned cue', 'reference');

    conditions(3) = new_condition( ...
        'delay_random_2_5', 'Random delay 2-5 frames', 'reference');
    conditions(3).random_delay_range = [2, 5];

    conditions(4) = new_condition( ...
        'noise_0p50', 'Noise sigma 0.50', 'noise');
    conditions(4).noise_sigma = 0.50;

    conditions(5) = new_condition( ...
        'noise_1p00', 'Noise sigma 1.00', 'noise');
    conditions(5).noise_sigma = 1.00;

    conditions(6) = new_condition( ...
        'delay_10', 'Delay 10 frames', 'delay');
    conditions(6).fixed_event_offset = 10;

    conditions(7) = new_condition( ...
        'delay_20', 'Delay 20 frames', 'delay');
    conditions(7).fixed_event_offset = 20;

    conditions(8) = new_condition( ...
        'gain_0p25', 'Moderate attenuation (gain 0.25)', ...
        'attenuation');
    conditions(8).event_gain = 0.25;

    conditions(9) = new_condition( ...
        'gain_0p10', 'Severe attenuation (gain 0.10)', ...
        'attenuation');
    conditions(9).event_gain = 0.10;

    conditions(10) = new_condition( ...
        'random_noise_only', 'Random noise only', 'null');
    conditions(10).include_true_event = false;
    conditions(10).event_gain = 0;

    conditions(11) = new_condition( ...
        'constant', 'Constant signal', 'null');
    conditions(11).include_true_event = false;
    conditions(11).event_gain = 0;
    conditions(11).noise_sigma = 0;

    conditions(12) = new_condition( ...
        'missing_random_0p30', ...
        'Random missing 30% with causal hold', 'missing');
    conditions(12).missing_mode = 'random';
    conditions(12).missing_rate = 0.30;

    conditions(13) = new_condition( ...
        'missing_random_0p50', ...
        'Random missing 50% with causal hold', 'missing');
    conditions(13).missing_mode = 'random';
    conditions(13).missing_rate = 0.50;

    conditions(14) = new_condition( ...
        'missing_burst_20', 'Event burst missing 20 frames', 'missing');
    conditions(14).missing_mode = 'burst';
    conditions(14).missing_burst_length = 20;

    conditions(15) = new_condition( ...
        'smooth_beta_0p90', 'Causal EWMA beta 0.90', 'smoothing');
    conditions(15).smoothing_beta = 0.90;

    conditions(16) = new_condition( ...
        'smooth_beta_0p99', 'Causal EWMA beta 0.99', 'smoothing');
    conditions(16).smoothing_beta = 0.99;

    conditions(17) = new_condition( ...
        'clock_lead_20', 'Clock lead 20 frames', 'misaligned');
    conditions(17).fixed_event_offset = -20;

    conditions(18) = new_condition( ...
        'false_event_minus50', 'Additional false event at -50', ...
        'misleading');
    conditions(18).false_event_offset = -50;
    conditions(18).false_event_gain = 1;

    conditions(19) = new_condition( ...
        'compound_stress', 'Compound side-information stress', ...
        'compound');
    conditions(19).event_gain = 0.50;
    conditions(19).noise_sigma = 0.50;
    conditions(19).fixed_event_offset = 10;
    conditions(19).missing_mode = 'random';
    conditions(19).missing_rate = 0.30;
end

function condition = new_condition(id, label, family)
    condition.id = id;
    condition.label = label;
    condition.family = family;
    condition.include_true_event = true;
    condition.event_gain = 1.0;
    condition.noise_sigma = 0.20;
    condition.fixed_event_offset = 0;
    condition.random_delay_range = [];
    condition.false_event_offset = 0;
    condition.false_event_gain = 0;
    condition.missing_mode = 'none';
    condition.missing_rate = 0;
    condition.missing_burst_length = 0;
    condition.smoothing_beta = 0;
end

function seeds = make_visual_seeds(trial)
    seeds.event = 10000 + trial;
    seeds.subspace = 20000 + trial;
    seeds.temporal = 30000 + trial;
    seeds.gaussian = 40000 + trial;
    seeds.spatter = 50000 + trial;
    seeds.observation = 60000 + trial;
    seeds.initialization = 70000 + trial;
end

function visual = generate_s5_visual_trial(rows, cols, total_frames, ...
    rank_r, mutation_scale, transition_boundary_range, snr_db, ...
    spatter_density, spatter_base_magnitude, observation_ratio, ...
    burn_in_frames, scale_multiplier, delta_lower_bound, ...
    delta_fallback, seeds)

    stream = RandStream('mt19937ar', 'Seed', seeds.event);
    transition_boundary = randi( ...
        stream, transition_boundary_range, 1, 1);

    stream = RandStream('mt19937ar', 'Seed', seeds.subspace);
    basis_a_before = orthonormalize_with_sign( ...
        randn(stream, rows, rank_r));
    basis_b_before = orthonormalize_with_sign( ...
        randn(stream, cols, rank_r));
    basis_a_after = orthonormalize_with_sign( ...
        basis_a_before + mutation_scale * randn(stream, rows, rank_r));
    basis_b_after = orthonormalize_with_sign( ...
        basis_b_before + mutation_scale * randn(stream, cols, rank_r));

    stream = RandStream('mt19937ar', 'Seed', seeds.temporal);
    frame_axis = (1:total_frames)';
    temporal_coefficients = zeros(total_frames, rank_r);
    for component = 1:rank_r
        temporal_coefficients(:, component) = 10.0 + ...
            2.0 * sin(2 * pi * frame_axis / ...
            (100 + 10 * component)) + ...
            0.1 * randn(stream, total_frames, 1);
    end

    clean_tensor = zeros(rows, cols, total_frames);
    % The transition boundary is the first frame using the post-mutation bases.
    for frame = 1:total_frames
        if frame < transition_boundary
            basis_a = basis_a_before;
            basis_b = basis_b_before;
        else
            basis_a = basis_a_after;
            basis_b = basis_b_after;
        end
        clean_tensor(:, :, frame) = basis_a * ...
            diag(temporal_coefficients(frame, :)) * basis_b';
    end

    signal_power = norm(clean_tensor(:))^2 / numel(clean_tensor);
    gaussian_sigma = sqrt(signal_power / 10^(snr_db / 10));
    stream = RandStream('mt19937ar', 'Seed', seeds.gaussian);
    observed_full = clean_tensor + gaussian_sigma * ...
        randn(stream, rows, cols, total_frames);

    stream = RandStream('mt19937ar', 'Seed', seeds.spatter);
    spatter_mask = rand(stream, rows, cols, total_frames) < ...
        spatter_density;
    num_spatter = nnz(spatter_mask);
    observed_full(spatter_mask) = observed_full(spatter_mask) + ...
        spatter_base_magnitude * ...
        (1 + abs(randn(stream, num_spatter, 1)));

    stream = RandStream('mt19937ar', 'Seed', seeds.observation);
    observation_mask = rand(stream, rows, cols, total_frames) < ...
        observation_ratio;
    observed_tensor = observed_full .* observation_mask;

    stream = RandStream('mt19937ar', 'Seed', seeds.initialization);
    initialization.A = orthonormalize_with_sign( ...
        randn(stream, rows, rank_r));
    initialization.B = orthonormalize_with_sign( ...
        randn(stream, cols, rank_r));
    initialization.C = randn(stream, total_frames, rank_r);

    huber_delta = estimate_s5_huber_delta(observed_tensor, ...
        observation_mask, burn_in_frames, scale_multiplier, ...
        delta_lower_bound, delta_fallback);

    visual.clean_tensor = clean_tensor;
    visual.observed_tensor = observed_tensor;
    visual.observation_mask = observation_mask;
    visual.initialization = initialization;
    visual.transition_boundary = transition_boundary;
    visual.huber_delta = huber_delta;
    visual.spatter_density = num_spatter / numel(spatter_mask);
end

function basis = orthonormalize_with_sign(input_matrix)
    [q_matrix, r_matrix] = qr(input_matrix, 0);
    signs = sign(diag(r_matrix) + 1e-10);
    basis = q_matrix * diag(signs);
end

function delta = estimate_s5_huber_delta(observed_tensor, ...
    observation_mask, burn_in_frames, scale_multiplier, ...
    lower_bound, fallback)

    differences = cell(burn_in_frames - 1, 1);
    for frame = 2:burn_in_frames
        common_mask = observation_mask(:, :, frame) & ...
            observation_mask(:, :, frame - 1);
        frame_difference = observed_tensor(:, :, frame) - ...
            observed_tensor(:, :, frame - 1);
        differences{frame - 1} = frame_difference(common_mask);
    end
    differences = vertcat(differences{:});

    if isempty(differences)
        delta = fallback;
    else
        center = median(differences);
        mad_value = median(abs(differences - center));
        sigma = (1.4826 * mad_value) / sqrt(2);
        delta = max(lower_bound, scale_multiplier * sigma);
    end
end

function [aligned_nre, aligned_lambda, ...
    pre_mutation_reduced_fraction, elapsed] = ...
    run_s5_tracking_case(visual, tensor_dims, rank_r, side_signal, ...
    side_threshold, lambda_min, lambda_max, mu, grad_ema_alpha, ...
    irls_max_iters, irls_tolerance, ...
    normalization_epsilon, start_frame, end_frame)

    options = struct( ...
        'lambda_min', lambda_min, ...
        'lambda_max', lambda_max, ...
        'huber_delta', visual.huber_delta, ...
        'min_grad_threshold', side_threshold, ...
        'mu', mu, ...
        'verbose', 0, ...
        'early_stop_on', 'none', ...
        'store_matrix', true, ...
        'store_subinfo', false, ...
        'grad_ema_alpha', grad_ema_alpha, ...
        'irls_max_iters', irls_max_iters, ...
        'irls_tolerance', irls_tolerance, ...
        'normalization_epsilon', normalization_epsilon);

    timer = tic;
    [~, ~, info] = rsi_olstec(visual.observed_tensor, ...
        visual.observation_mask, [], tensor_dims, rank_r, ...
        visual.initialization, options, side_signal);
    elapsed = toc(timer);

    full_nre = validate_complete_nre( ...
        compute_true_nre_tensor(visual.clean_tensor, info.L), ...
        tensor_dims(3), 'RSI-OLSTEC', NaN);
    full_lambda = info.lambda_history(:)';
    if numel(full_lambda) ~= tensor_dims(3)
        error('Exp_S5:InvalidTrackingOutput', ...
            'RSI-OLSTEC returned %d forgetting factors; %d were expected.', ...
            numel(full_lambda), tensor_dims(3));
    end
    aligned_nre = full_nre(start_frame:end_frame);
    aligned_lambda = full_lambda(start_frame:end_frame);

    tolerance = 1e-12;
    pre_mutation_frames = 2:(visual.transition_boundary - 1);
    pre_mutation_reduced_fraction = mean( ...
        full_lambda(pre_mutation_frames) < lambda_max - tolerance);

    if any(~isfinite(full_lambda)) || ...
            any(full_lambda < lambda_min - tolerance) || ...
            any(full_lambda > lambda_max + tolerance)
        error('Exp_S5:InvalidTrackingOutput', ...
            'RSI-OLSTEC returned an invalid NRE or forgetting-factor curve.');
    end
end

function metrics = compute_s5_metrics( ...
    nre_curve, lambda_curve, zero_index, lambda_max, ...
    pre_mutation_reduced_fraction)

    pre_50 = (zero_index - 50):(zero_index - 1);
    post_50 = zero_index:(zero_index + 49);
    post_100 = zero_index:(zero_index + 99);
    window_0_to_50 = zero_index:(zero_index + 50);

    lambda_reduced = lambda_curve < lambda_max - 1e-12;
    first_reduction = find(lambda_reduced(window_0_to_50), 1);
    if isempty(first_reduction)
        first_lag = NaN;
    else
        first_lag = first_reduction - 1;
    end

    metrics = [ ...
        mean(nre_curve(pre_50)), ...
        max(nre_curve(post_50)), ...
        sum(nre_curve(post_50)), ...
        mean(nre_curve(post_100)), ...
        first_lag, ...
        min(lambda_curve(window_0_to_50)), ...
        sum(lambda_max - lambda_curve(window_0_to_50)), ...
        pre_mutation_reduced_fraction];
end

function [per_trial_table, summary_table] = build_s5_tables( ...
    state, conditions, run_ids, run_labels, metric_names, ...
    ci_level, bootstrap_resamples)

    num_runs = numel(run_ids);
    num_trials = size(state.metrics, 2);
    num_rows = num_runs * num_trials;
    metric_index = @(name) find(strcmp(metric_names, name), 1);

    Trial = zeros(num_rows, 1);
    ConditionID = cell(num_rows, 1);
    Label = cell(num_rows, 1);
    RuntimeSeconds = NaN(num_rows, 1);
    PreNRE50 = NaN(num_rows, 1);
    PostPeakNRE50 = NaN(num_rows, 1);
    RecoveryIntegral50 = NaN(num_rows, 1);
    PostMeanNRE100 = NaN(num_rows, 1);
    FirstLambdaReductionLag0To50 = NaN(num_rows, 1);
    MinimumLambda0To50 = NaN(num_rows, 1);
    CumulativeLambdaReduction0To50 = NaN(num_rows, 1);
    PreMutationReducedLambdaFraction = NaN(num_rows, 1);
    SideThreshold = NaN(num_rows, 1);
    SideEventDelay = NaN(num_rows, 1);
    SideMissingRate = NaN(num_rows, 1);
    SideOnsetGradientToThresholdRatio = NaN(num_rows, 1);

    row = 0;
    for trial = 1:num_trials
        for run_index = 1:num_runs
            row = row + 1;
            Trial(row) = trial;
            ConditionID{row} = run_ids{run_index};
            Label{row} = run_labels{run_index};
            RuntimeSeconds(row) = state.runtime_seconds(run_index, trial);
            PreNRE50(row) = state.metrics(run_index, trial, ...
                metric_index('PreNRE50'));
            PostPeakNRE50(row) = state.metrics(run_index, trial, ...
                metric_index('PostPeakNRE50'));
            RecoveryIntegral50(row) = state.metrics(run_index, trial, ...
                metric_index('RecoveryIntegral50'));
            PostMeanNRE100(row) = state.metrics(run_index, trial, ...
                metric_index('PostMeanNRE100'));
            FirstLambdaReductionLag0To50(row) = ...
                state.metrics(run_index, trial, ...
                metric_index('FirstLambdaReductionLag0To50'));
            MinimumLambda0To50(row) = state.metrics(run_index, trial, ...
                metric_index('MinimumLambda0To50'));
            CumulativeLambdaReduction0To50(row) = ...
                state.metrics(run_index, trial, ...
                metric_index('CumulativeLambdaReduction0To50'));
            PreMutationReducedLambdaFraction(row) = ...
                state.metrics(run_index, trial, ...
                metric_index('PreMutationReducedLambdaFraction'));

            if run_index > 1
                condition_index = run_index - 1;
                SideThreshold(row) = ...
                    state.side_threshold(condition_index, trial);
                SideEventDelay(row) = ...
                    state.side_event_delay(condition_index, trial);
                SideMissingRate(row) = ...
                    state.side_missing_rate(condition_index, trial);
                SideOnsetGradientToThresholdRatio(row) = ...
                    state.side_onset_gradient_to_threshold( ...
                    condition_index, trial);
            end
        end
    end

    per_trial_table = table(Trial, ConditionID, Label, RuntimeSeconds, ...
        PreNRE50, PostPeakNRE50, RecoveryIntegral50, PostMeanNRE100, ...
        FirstLambdaReductionLag0To50, MinimumLambda0To50, ...
        CumulativeLambdaReduction0To50, ...
        PreMutationReducedLambdaFraction, SideThreshold, SideEventDelay, ...
        SideMissingRate, SideOnsetGradientToThresholdRatio);

    RecoveryIntegral50Mean = NaN(num_runs, 1);
    RecoveryIntegral50Std = NaN(num_runs, 1);
    RecoveryIntegral50CILower = NaN(num_runs, 1);
    RecoveryIntegral50CIUpper = NaN(num_runs, 1);
    FixedMinusConditionMean = NaN(num_runs, 1);
    FixedMinusConditionCILower = NaN(num_runs, 1);
    FixedMinusConditionCIUpper = NaN(num_runs, 1);
    ConditionMinusNoisyMutationAlignedMean = NaN(num_runs, 1);
    ConditionMinusNoisyMutationAlignedCILower = NaN(num_runs, 1);
    ConditionMinusNoisyMutationAlignedCIUpper = NaN(num_runs, 1);
    PostMeanNRE100Mean = NaN(num_runs, 1);
    PreNRE50Mean = NaN(num_runs, 1);
    PreMutationReducedLambdaFractionMean = NaN(num_runs, 1);

    recovery_index = metric_index('RecoveryIntegral50');
    post_mean_index = metric_index('PostMeanNRE100');
    pre_nre_index = metric_index('PreNRE50');
    pre_reduction_index = ...
        metric_index('PreMutationReducedLambdaFraction');
    fixed_values = reshape(state.metrics(1, :, recovery_index), [], 1);
    noisy_mutation_condition = find( ...
        strcmp({conditions.id}, 'noisy_mutation_aligned'), 1);
    noisy_mutation_values = reshape(state.metrics( ...
        noisy_mutation_condition + 1, :, recovery_index), [], 1);

    for run_index = 1:num_runs
        values = reshape( ...
            state.metrics(run_index, :, recovery_index), [], 1);
        RecoveryIntegral50Mean(run_index) = mean(values);
        RecoveryIntegral50Std(run_index) = std(values);
        [RecoveryIntegral50CILower(run_index), ...
            RecoveryIntegral50CIUpper(run_index)] = bootstrap_mean_ci( ...
            values, bootstrap_resamples, ci_level, 200000 + run_index);

        fixed_difference = fixed_values - values;
        FixedMinusConditionMean(run_index) = mean(fixed_difference);
        [FixedMinusConditionCILower(run_index), ...
            FixedMinusConditionCIUpper(run_index)] = bootstrap_mean_ci( ...
            fixed_difference, bootstrap_resamples, ci_level, ...
            210000 + run_index);

        noisy_mutation_difference = values - noisy_mutation_values;
        ConditionMinusNoisyMutationAlignedMean(run_index) = ...
            mean(noisy_mutation_difference);
        [ConditionMinusNoisyMutationAlignedCILower(run_index), ...
            ConditionMinusNoisyMutationAlignedCIUpper(run_index)] = ...
            bootstrap_mean_ci(noisy_mutation_difference, ...
            bootstrap_resamples, ci_level, 220000 + run_index);

        PostMeanNRE100Mean(run_index) = mean(reshape( ...
            state.metrics(run_index, :, post_mean_index), [], 1));
        PreNRE50Mean(run_index) = mean(reshape( ...
            state.metrics(run_index, :, pre_nre_index), [], 1));
        PreMutationReducedLambdaFractionMean(run_index) = ...
            mean(reshape(state.metrics(run_index, :, ...
            pre_reduction_index), [], 1));
    end

    ConditionID = run_ids(:);
    Label = run_labels(:);
    summary_table = table(ConditionID, Label, ...
        RecoveryIntegral50Mean, RecoveryIntegral50Std, ...
        RecoveryIntegral50CILower, RecoveryIntegral50CIUpper, ...
        FixedMinusConditionMean, FixedMinusConditionCILower, ...
        FixedMinusConditionCIUpper, ...
        ConditionMinusNoisyMutationAlignedMean, ...
        ConditionMinusNoisyMutationAlignedCILower, ...
        ConditionMinusNoisyMutationAlignedCIUpper, ...
        PostMeanNRE100Mean, PreNRE50Mean, ...
        PreMutationReducedLambdaFractionMean);
end

function [lower, upper] = bootstrap_mean_ci( ...
    values, num_resamples, ci_level, seed)

    values = values(:);
    if numel(values) < 2
        lower = NaN;
        upper = NaN;
        return;
    end

    stream = RandStream('mt19937ar', 'Seed', seed);
    indices = randi(stream, numel(values), numel(values), num_resamples);
    bootstrap_means = mean(values(indices), 1);
    bootstrap_means = sort(bootstrap_means);
    tail = (1 - ci_level) / 2;
    lower = linear_quantile(bootstrap_means, tail);
    upper = linear_quantile(bootstrap_means, 1 - tail);
end

function value = linear_quantile(sorted_values, probability)
    position = 1 + (numel(sorted_values) - 1) * probability;
    lower_index = floor(position);
    upper_index = ceil(position);
    if lower_index == upper_index
        value = sorted_values(lower_index);
    else
        weight = position - lower_index;
        value = sorted_values(lower_index) + weight * ...
            (sorted_values(upper_index) - sorted_values(lower_index));
    end
end

function plot_s5_quality_summary(summary_table, conditions, ...
    result_dir, export_results)

    figure_handle = figure('Color', 'w', 'Visible', 'off', ...
        'Position', [80, 80, 1450, 650]);
    axes_handle = axes('Parent', figure_handle);
    hold(axes_handle, 'on');

    for condition_index = 1:numel(conditions)
        row = condition_index + 1;
        mean_value = summary_table.RecoveryIntegral50Mean(row);
        lower_error = mean_value - ...
            summary_table.RecoveryIntegral50CILower(row);
        upper_error = ...
            summary_table.RecoveryIntegral50CIUpper(row) - mean_value;
        color = family_color(conditions(condition_index).family);
        errorbar(axes_handle, condition_index, mean_value, ...
            lower_error, upper_error, 'o', 'Color', color, ...
            'MarkerFaceColor', color, 'MarkerSize', 6, ...
            'LineWidth', 1.3, 'CapSize', 7);
    end

    fixed_mean = summary_table.RecoveryIntegral50Mean(1);
    fixed_line = plot(axes_handle, [0.5, numel(conditions) + 0.5], ...
        [fixed_mean, fixed_mean], 'k--', 'LineWidth', 1.5);
    set(axes_handle, 'XTick', 1:numel(conditions), ...
        'XTickLabel', {conditions.label}, ...
        'FontName', 'Times New Roman', 'FontSize', 10);
    xtickangle(axes_handle, 45);
    xlim(axes_handle, [0.5, numel(conditions) + 0.5]);
    ylabel(axes_handle, 'Recovery integral over 50 frames');
    title(axes_handle, 'Side-information reliability');
    legend(axes_handle, fixed_line, {'Huber fixed lambda 0.80'}, ...
        'Location', 'northwest', 'Box', 'off');
    grid(axes_handle, 'on');
    box(axes_handle, 'on');

    export_s5_figure(figure_handle, result_dir, ...
        'S5_side_quality_summary', export_results);
end

function plot_s5_selected_trajectories(state, run_ids, ...
    run_labels, relative_frames, result_dir, export_results)

    selected_ids = {'fixed_huber', 'noisy_mutation_aligned', 'delay_20', ...
        'missing_burst_20', 'random_noise_only', ...
        'false_event_minus50'};
    colors = lines(numel(selected_ids));

    figure_handle = figure('Color', 'w', 'Visible', 'off', ...
        'Position', [100, 60, 1050, 900]);
    nre_axes = subplot(3, 1, 1, 'Parent', figure_handle);
    lambda_axes = subplot(3, 1, 2, 'Parent', figure_handle);
    side_axes = subplot(3, 1, 3, 'Parent', figure_handle);
    hold(nre_axes, 'on');
    hold(lambda_axes, 'on');
    hold(side_axes, 'on');

    nre_handles = gobjects(numel(selected_ids), 1);
    lambda_handles = gobjects(numel(selected_ids), 1);
    side_handles = gobjects(numel(selected_ids) - 1, 1);
    selected_labels = cell(numel(selected_ids), 1);

    for selected_index = 1:numel(selected_ids)
        run_index = find(strcmp(run_ids, selected_ids{selected_index}), 1);
        nre_curves = reshape(state.aligned_nre(run_index, :, :), ...
            size(state.aligned_nre, 2), size(state.aligned_nre, 3));
        lambda_curves = reshape( ...
            state.aligned_lambda(run_index, :, :), ...
            size(state.aligned_lambda, 2), ...
            size(state.aligned_lambda, 3));

        nre_handles(selected_index) = semilogy(nre_axes, ...
            relative_frames, mean(nre_curves, 1), ...
            'Color', colors(selected_index, :), 'LineWidth', 1.6);
        lambda_handles(selected_index) = plot(lambda_axes, ...
            relative_frames, mean(lambda_curves, 1), ...
            'Color', colors(selected_index, :), 'LineWidth', 1.6);
        selected_labels{selected_index} = run_labels{run_index};

        if run_index > 1
            condition_index = run_index - 1;
            side_curves = reshape( ...
                state.aligned_side(condition_index, :, :), ...
                size(state.aligned_side, 2), ...
                size(state.aligned_side, 3));
            side_handles(selected_index - 1) = plot(side_axes, ...
                relative_frames, mean(side_curves, 1), ...
                'Color', colors(selected_index, :), 'LineWidth', 1.4);
        end
    end

    add_event_marker(nre_axes);
    add_event_marker(lambda_axes);
    add_event_marker(side_axes);
    set(nre_axes, 'YScale', 'log');
    ylabel(nre_axes, 'NRE');
    ylabel(lambda_axes, 'Forgetting factor');
    ylabel(side_axes, 'Effective side information');
    xlabel(side_axes, 'Frames relative to structural mutation');
    title(nre_axes, 'Mean mutation-aligned trajectories');
    ylim(lambda_axes, [0.08, 0.82]);
    legend(nre_axes, nre_handles, selected_labels, ...
        'Location', 'eastoutside', 'Box', 'off');
    legend(lambda_axes, lambda_handles, selected_labels, ...
        'Location', 'eastoutside', 'Box', 'off');
    legend(side_axes, side_handles, selected_labels(2:end), ...
        'Location', 'eastoutside', 'Box', 'off');
    grid(nre_axes, 'on'); box(nre_axes, 'on');
    grid(lambda_axes, 'on'); box(lambda_axes, 'on');
    grid(side_axes, 'on'); box(side_axes, 'on');
    set([nre_axes, lambda_axes, side_axes], ...
        'FontName', 'Times New Roman', 'FontSize', 10);

    export_s5_figure(figure_handle, result_dir, ...
        'S5_selected_trajectories', export_results);
end

function add_event_marker(axes_handle)
    limits = ylim(axes_handle);
    line(axes_handle, [0, 0], limits, ...
        'Color', 'k', 'LineStyle', ':', 'LineWidth', 1.0);
    ylim(axes_handle, limits);
end

function color = family_color(family)
    switch family
        case 'reference'
            color = [0.10, 0.45, 0.75];
        case 'noise'
            color = [0.85, 0.33, 0.10];
        case 'delay'
            color = [0.49, 0.18, 0.56];
        case 'attenuation'
            color = [0.47, 0.67, 0.19];
        case 'null'
            color = [0.35, 0.35, 0.35];
        case 'missing'
            color = [0.93, 0.69, 0.13];
        case 'smoothing'
            color = [0.30, 0.75, 0.93];
        case 'misaligned'
            color = [0.64, 0.08, 0.18];
        case 'misleading'
            color = [0.75, 0.15, 0.15];
        otherwise
            color = [0, 0, 0];
    end
end

function export_s5_figure(figure_handle, result_dir, ...
    base_filename, export_results)

    if export_results
        original_visibility = get(figure_handle, 'Visible');
        set(figure_handle, 'Visible', 'on');
        savefig(figure_handle, ...
            fullfile(result_dir, [base_filename, '.fig']));
        set(figure_handle, 'Visible', original_visibility);
        set(figure_handle, 'Units', 'inches');
        position = get(figure_handle, 'Position');
        set(figure_handle, 'PaperUnits', 'inches', ...
            'PaperSize', position(3:4), ...
            'PaperPosition', [0, 0, position(3:4)], ...
            'PaperPositionMode', 'manual');
        print(figure_handle, ...
            fullfile(result_dir, [base_filename, '.pdf']), ...
            '-dpdf', '-vector');
        print(figure_handle, ...
            fullfile(result_dir, [base_filename, '.eps']), ...
            '-depsc', '-vector');
    end
    close(figure_handle);
end
