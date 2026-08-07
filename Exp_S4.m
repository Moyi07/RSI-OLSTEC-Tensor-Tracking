%% Experiment S4
% =========================================================================
% Prerequisites: Run `run_me_first.m` once to add all dependencies to path.
%
% Objective:
%   Compare tracking methods across observation ratios and tensor ranks
%   under smooth dynamics, Gaussian noise, and sparse impulsive corruption.
%   The main figure reports steady-state NRE and trajectories at rho = 0.10.
%   Matrix baselines use the same rank as the tensor methods, and the plotted
%   OLSTEC configuration uses lambda = 0.80.
% =========================================================================
clear; clc; close all;

%% 1. Core Experimental Variables (Dual Independent Variables)
% -------------------------------------------------------------------------
test_fractions = [0.70, 0.50, 0.30, 0.10]; % Independent Variable 1: Observation Ratio (70%, 50%, 30%, 10%)
test_ranks     = [5, 10, 15];              % Independent Variable 2: Background complexity / Tensor Rank
target_ts_fraction = 0.10;                 % Observation ratio for the time-series curves
num_trials   = 50;                        % Number of Monte Carlo trials
tensor_dims  = [50, 50, 500];
sparse_ratio = 0.05;
SNR_dB       = 25;
tolcost      = 1e-8;
export_results = true;
aux_noise_sigma = 0.20;
result_dir = fullfile(fileparts(mfilename('fullpath')), 'result', 'S4');
if export_results && ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

alg_names = {'PETRELS', 'GRASTA', 'GROUSE', 'TeCPSGD', 'OLSTEC', 'RSI-OLSTEC'};
num_algs = length(alg_names);

num_fracs = length(test_fractions);
num_ranks = length(test_ranks);
matrix_rank_by_rank = test_ranks;

lambda_list_olstec = [0.70, 0.80, 0.90, 0.99];
num_olstec_lams = length(lambda_list_olstec);
fixed_lam_idx = find(abs(lambda_list_olstec - 0.80) < 1e-12, 1);
T = tensor_dims(3);
steady_state_window = max(1, T - 99):T;

% Raw per-trial results are retained so every reported statistic can be
% reconstructed from complete trajectories.
mean_errors_3D = NaN(num_algs, num_fracs, num_ranks);
std_errors_3D = NaN(num_algs, num_fracs, num_ranks);
time_series_3D = NaN(num_algs, T, num_ranks);
valid_counts_3D = zeros(num_algs, num_fracs, num_ranks);
failure_counts_3D = zeros(num_algs, num_fracs, num_ranks);
nre_trials = NaN(num_algs, num_fracs, num_ranks, num_trials, T);
steady_state_errors = NaN(num_algs, num_fracs, num_ranks, num_trials);
algorithm_status = zeros(num_algs, num_fracs, num_ranks, num_trials, 'uint8');
rsi_lambda_mean_trials = NaN(num_fracs, num_ranks, num_trials);
rsi_lambda_below_max_pct_trials = NaN(num_fracs, num_ranks, num_trials);

olstec_nre_trials = NaN(num_olstec_lams, num_fracs, num_ranks, num_trials, T);
olstec_status = zeros(num_olstec_lams, num_fracs, num_ranks, num_trials, 'uint8');
olstec_sens_ts_cum    = zeros(num_olstec_lams, tensor_dims(3), num_ranks);
olstec_sens_ts_counts = zeros(num_olstec_lams, tensor_dims(3), num_ranks);

trial_seed_matrix = zeros(num_ranks, num_trials);
clean_seed_matrix = zeros(num_ranks, num_trials);
auxiliary_seed_matrix = zeros(num_ranks, num_trials);
spatter_seed_matrix = zeros(num_ranks, num_trials);
gaussian_seed_matrix = zeros(num_ranks, num_trials);
mask_seed_matrix = zeros(num_ranks, num_trials);
initialization_seed_matrix = zeros(num_ranks, num_trials);
matrix_initialization_seed_matrix = zeros(num_ranks, num_trials);
algorithm_seed_4D = zeros(num_algs, num_fracs, num_ranks, num_trials);
olstec_seed_4D = zeros(num_olstec_lams, num_fracs, num_ranks, num_trials);
empty_failure_record = struct( ...
    'Algorithm', '', 'Rank', NaN, 'ObservationRatio', NaN, ...
    'Lambda', NaN, 'Trial', NaN, 'TrialSeed', NaN, ...
    'ExecutionSeed', NaN, 'Category', '', 'Identifier', '', ...
    'Message', '', 'OutputLength', NaN, 'FirstInvalidFrame', NaN);
max_failure_records = (num_algs - 1 + num_olstec_lams) * ...
    num_fracs * num_ranks * num_trials;
failure_records = repmat(empty_failure_record, max_failure_records, 1);
failure_record_count = 0;

fprintf('Starting cross-ablation study (%d Monte Carlo trials)...\n', num_trials);
total_start = tic;

%% 2. Paired Monte Carlo Simulation
% -------------------------------------------------------------------------
for r_idx = 1:num_ranks
    rank_r = test_ranks(r_idx);
    fprintf('\n======================================================\n');
    fprintf('Main Group %d/%d: Target Rank = %d, Matrix Rank = %d\n', ...
        r_idx, num_ranks, rank_r, matrix_rank_by_rank(r_idx));
    fprintf('======================================================\n');

    rows = tensor_dims(1);
    cols = tensor_dims(2);
    numr = rows * cols;
    numc = T;
    matrix_rank = rank_r;

    for trial = 1:num_trials
        fprintf('  - Trial %d/%d... ', trial, num_trials);
        failures_before_trial = failure_record_count;

        trial_seed = 42 + (r_idx - 1) * 1000 + (trial - 1);
        trial_seed_matrix(r_idx, trial) = trial_seed;
        clean_seed_matrix(r_idx, trial) = 100000 + trial_seed;
        auxiliary_seed_matrix(r_idx, trial) = 200000 + trial_seed;
        spatter_seed_matrix(r_idx, trial) = 300000 + trial_seed;
        gaussian_seed_matrix(r_idx, trial) = 400000 + trial_seed;
        mask_seed_matrix(r_idx, trial) = 500000 + trial_seed;
        initialization_seed_matrix(r_idx, trial) = 600000 + trial_seed;
        matrix_initialization_seed_matrix(r_idx, trial) = 650000 + trial_seed;

        rng(clean_seed_matrix(r_idx, trial), 'twister');
        A_true = s4_orthonormalize(randn(rows, rank_r));
        B_true = s4_orthonormalize(randn(cols, rank_r));
        C_true = zeros(T, rank_r);
        frame_index = (1:T)';
        for r = 1:rank_r
            C_true(:, r) = 10.0 + 2.0 * sin(2 * pi * frame_index / (100 + r * 10)) + ...
                0.1 * randn(T, 1);
        end

        Tensor_Y_Clean = zeros(rows, cols, T);
        for f = 1:T
            Tensor_Y_Clean(:, :, f) = A_true * diag(C_true(f, :)) * B_true';
        end

        rng(auxiliary_seed_matrix(r_idx, trial), 'twister');
        aux_info = C_true(:, 1) + aux_noise_sigma * randn(T, 1);
        burn_in_aux = min(30, length(aux_info));
        diff_aux = diff(aux_info(1:burn_in_aux));
        mad_aux = median(abs(diff_aux - median(diff_aux)));
        est_aux_sigma = (1.4826 * mad_aux) / sqrt(2);
        adaptive_min_grad = max(0.05, 3 * sqrt(2) * est_aux_sigma);

        rng(spatter_seed_matrix(r_idx, trial), 'twister');
        spike_magnitude = max(abs(Tensor_Y_Clean(:))) * 1.5;
        S_mask = rand(rows, cols, T) < sparse_ratio;
        Sparse_Noise = zeros(rows, cols, T);
        Sparse_Noise(S_mask) = spike_magnitude * (1 + abs(randn(sum(S_mask(:)), 1)));

        sig_pow = norm(Tensor_Y_Clean(:))^2 / numel(Tensor_Y_Clean);
        noise_sigma = sqrt(sig_pow / 10^(SNR_dB / 10));
        rng(gaussian_seed_matrix(r_idx, trial), 'twister');
        Gaussian_Noise = noise_sigma * randn(rows, cols, T);
        Tensor_Y_Noisy = Tensor_Y_Clean + Gaussian_Noise + Sparse_Noise;
        Matrix_Y_Noisy = reshape(Tensor_Y_Noisy, [numr, numc]);

        rng(mask_seed_matrix(r_idx, trial), 'twister');
        uniform_mask = rand(rows, cols, T);

        rng(initialization_seed_matrix(r_idx, trial), 'twister');
        Xinit.A = randn(rows, rank_r);
        Xinit.B = randn(cols, rank_r);
        Xinit.C = randn(T, rank_r);

        % Matrix methods share one initial column space within each trial.
        matrix_init_stream = RandStream('mt19937ar', ...
            'Seed', matrix_initialization_seed_matrix(r_idx, trial));
        matrix_init = struct();
        matrix_init.U = randn(matrix_init_stream, numr, matrix_rank);
        matrix_init.Weight = randn(matrix_init_stream, matrix_rank, numc);

        calc_true_nre = @(L_out) compute_true_nre_tensor(Tensor_Y_Clean, L_out);

        for exp_idx = 1:num_fracs
            fraction = test_fractions(exp_idx);
            OmegaTensor = uniform_mask < fraction;
            OmegaMatrix = reshape(OmegaTensor, [numr, numc]);

            burn_in = min(30, T);
            diff_pixels = cell(burn_in - 1, 1);
            for frame = 2:burn_in
                common_mask = OmegaTensor(:, :, frame) & OmegaTensor(:, :, frame - 1);
                diff_frame = Tensor_Y_Noisy(:, :, frame) - Tensor_Y_Noisy(:, :, frame - 1);
                diff_pixels{frame - 1} = diff_frame(common_mask);
            end
            diff_pixels = vertcat(diff_pixels{:});
            if ~isempty(diff_pixels)
                mad_val = median(abs(diff_pixels - median(diff_pixels)));
                est_sigma = (1.4826 * mad_val) / sqrt(2);
                huber_delta = max(0.01, 3 * est_sigma);
            else
                huber_delta = 0.05;
            end

            for a = 1:num_algs
                algo_name = alg_names{a};

                if strcmp(algo_name, 'OLSTEC')
                    for lam_idx = 1:num_olstec_lams
                        lam = lambda_list_olstec(lam_idx);
                        err_curve = [];
                        execution_seed = 1700000 + 100000 * r_idx + ...
                            1000 * trial + lam_idx;
                        olstec_seed_4D(lam_idx, exp_idx, r_idx, trial) = execution_seed;
                        if lam_idx == fixed_lam_idx
                            algorithm_seed_4D(a, exp_idx, r_idx, trial) = execution_seed;
                        end
                        try
                            rng(execution_seed, 'twister');
                            opts = struct('maxepochs', 1, 'tolcost', tolcost, ...
                                'permute_on', 0, 'lambda', lam, 'mu', 0.01, ...
                                'tw_flag', 0, 'tw_len', 10, 'verbose', 0, ...
                                'store_matrix', 1, 'store_subinfo', 1);
                            [~, ~, info] = olstec(Tensor_Y_Noisy, OmegaTensor, [], ...
                                tensor_dims, rank_r, Xinit, opts);
                            err_curve = calc_true_nre(info.L);
                            label = sprintf('OLSTEC (lambda=%.2f)', lam);
                            err_curve = validate_complete_nre(err_curve, T, label, trial);
                            olstec_nre_trials(lam_idx, exp_idx, r_idx, trial, :) = ...
                                reshape(err_curve, 1, 1, 1, 1, []);
                            olstec_status(lam_idx, exp_idx, r_idx, trial) = 1;

                            if lam_idx == fixed_lam_idx
                                nre_trials(a, exp_idx, r_idx, trial, :) = ...
                                    reshape(err_curve, 1, 1, 1, 1, []);
                                steady_state_errors(a, exp_idx, r_idx, trial) = ...
                                    mean(err_curve(steady_state_window));
                                algorithm_status(a, exp_idx, r_idx, trial) = 1;
                            end
                        catch ME
                            if s4_is_process_interruption(ME)
                                rethrow(ME);
                            end
                            olstec_status(lam_idx, exp_idx, r_idx, trial) = 2;
                            if lam_idx == fixed_lam_idx
                                algorithm_status(a, exp_idx, r_idx, trial) = 2;
                            end
                            label = sprintf('OLSTEC (lambda=%.2f)', lam);
                            failure_record_count = failure_record_count + 1;
                            failure_records(failure_record_count, 1) = s4_failure_record( ...
                                label, rank_r, fraction, lam, trial, trial_seed, ...
                                execution_seed, ME, err_curve);
                        end
                    end
                    continue;
                end

                err_curve = [];
                execution_seed = 700000 + 100000 * r_idx + ...
                    1000 * trial + a;
                if strcmp(algo_name, 'GRASTA')
                    execution_seed = matrix_initialization_seed_matrix(r_idx, trial);
                end
                algorithm_seed_4D(a, exp_idx, r_idx, trial) = execution_seed;
                lambda_mean_value = NaN;
                lambda_below_max_value = NaN;
                try
                    rng(execution_seed, 'twister');
                    switch algo_name
                        case 'PETRELS'
                            opts = struct('maxepochs', 1, 'tolcost', tolcost, ...
                                'permute_on', 0, 'rank', matrix_rank, 'lambda', 0.98, ...
                                'verbose', 0, 'store_matrix', 1, 'store_subinfo', 1);
                            [~, ~, info] = petrels_mod(matrix_init, Matrix_Y_Noisy, ...
                                OmegaMatrix, [], numr, numc, opts);
                            err_curve = calc_true_nre(info.L);
                        case 'GRASTA'
                            opts = struct('maxepochs', 1, 'tolcost', tolcost, ...
                                'permute_on', 0, 'RANK', matrix_rank, 'rho', 1.8, ...
                                'ITER_MAX', 20, 'MAX_MU', 10000, 'MIN_MU', 1, ...
                                'DIM_M', numr, 'USE_MEX', 0, 'verbose', 0, ...
                                'store_matrix', 1, 'store_subinfo', 1);
                            [~, ~, info] = grasta_mod(matrix_init, Matrix_Y_Noisy, ...
                                OmegaMatrix, [], numr, numc, opts);
                            err_curve = calc_true_nre(info.L);
                        case 'GROUSE'
                            opts = struct('maxepochs', 1, 'tolcost', tolcost, ...
                                'permute_on', 0, 'maxrank', matrix_rank, ...
                                'step_size', 0.0001, 'verbose', 0, ...
                                'store_matrix', 1, 'store_subinfo', 1);
                            [~, ~, info] = grouse_mod(matrix_init, Matrix_Y_Noisy, ...
                                OmegaMatrix, [], numr, numc, opts);
                            err_curve = calc_true_nre(info.L);
                        case 'TeCPSGD'
                            opts = struct('maxepochs', 1, 'tolcost', tolcost, ...
                                'permute_on', 0, 'lambda', 0.99, 'stepsize', 0.1, ...
                                'mu', 0.01, 'verbose', 0, 'store_matrix', 1, ...
                                'store_subinfo', 1);
                            [~, ~, info] = TeCPSGD(Tensor_Y_Noisy, OmegaTensor, [], ...
                                tensor_dims, rank_r, Xinit, opts);
                            err_curve = calc_true_nre(reshape(info.L, [numr, numc]));
                        case 'RSI-OLSTEC'
                            opts = struct('maxepochs', 1, 'tolcost', tolcost, ...
                                'permute_on', 0, 'lambda_max', 0.80, ...
                                'lambda_min', 0.70, 'huber_delta', huber_delta, ...
                                'min_grad_threshold', adaptive_min_grad, 'mu', 0.01, ...
                                'verbose', 0, 'store_matrix', 1, 'store_subinfo', 1);
                            [~, ~, info] = rsi_olstec(Tensor_Y_Noisy, OmegaTensor, [], ...
                                tensor_dims, rank_r, Xinit, opts, aux_info);
                            err_curve = calc_true_nre(info.L);
                            lambda_vals = info.lambda_history(:);
                            lambda_vals = lambda_vals(isfinite(lambda_vals));
                            if ~isempty(lambda_vals)
                                lambda_mean_value = mean(lambda_vals);
                                lambda_below_max_value = 100 * mean(lambda_vals < opts.lambda_max);
                            end
                    end

                    err_curve = validate_complete_nre(err_curve, T, algo_name, trial);
                    nre_trials(a, exp_idx, r_idx, trial, :) = ...
                        reshape(err_curve, 1, 1, 1, 1, []);
                    steady_state_errors(a, exp_idx, r_idx, trial) = ...
                        mean(err_curve(steady_state_window));
                    algorithm_status(a, exp_idx, r_idx, trial) = 1;
                    if strcmp(algo_name, 'RSI-OLSTEC')
                        rsi_lambda_mean_trials(exp_idx, r_idx, trial) = lambda_mean_value;
                        rsi_lambda_below_max_pct_trials(exp_idx, r_idx, trial) = ...
                            lambda_below_max_value;
                    end
                catch ME
                    if s4_is_process_interruption(ME)
                        rethrow(ME);
                    end
                    algorithm_status(a, exp_idx, r_idx, trial) = 2;
                    failure_record_count = failure_record_count + 1;
                    failure_records(failure_record_count, 1) = s4_failure_record( ...
                        algo_name, rank_r, fraction, NaN, trial, trial_seed, ...
                        execution_seed, ME, err_curve);
                end
            end
        end

        trial_failures = failure_record_count - failures_before_trial;
        fprintf('completed (%d recorded failures).\n', trial_failures);
    end
end

failure_records = failure_records(1:failure_record_count);

if any(algorithm_status(:) == 0) || any(olstec_status(:) == 0)
    error('Exp_S4:IncompleteExecutionState', ...
        'At least one configured algorithm run was neither completed nor recorded as failed.');
end

for r_idx = 1:num_ranks
    for exp_idx = 1:num_fracs
        for a = 1:num_algs
            status = reshape(algorithm_status(a, exp_idx, r_idx, :), num_trials, 1);
            success = (status == 1);
            valid_counts_3D(a, exp_idx, r_idx) = sum(success);
            failure_counts_3D(a, exp_idx, r_idx) = sum(status == 2);

            values = reshape(steady_state_errors(a, exp_idx, r_idx, :), num_trials, 1);
            values = values(success);
            if ~isempty(values)
                mean_errors_3D(a, exp_idx, r_idx) = mean(values);
            end
            if numel(values) > 1
                std_errors_3D(a, exp_idx, r_idx) = std(values);
            end

            if abs(test_fractions(exp_idx) - target_ts_fraction) < 1e-12 && any(success)
                curves = reshape(nre_trials(a, exp_idx, r_idx, :, :), num_trials, T);
                time_series_3D(a, :, r_idx) = mean(curves(success, :), 1);
            end
        end
    end

    target_idx = find(abs(test_fractions - target_ts_fraction) < 1e-12, 1);
    if ~isempty(target_idx)
        for lam_idx = 1:num_olstec_lams
            status = reshape(olstec_status(lam_idx, target_idx, r_idx, :), num_trials, 1);
            success = (status == 1);
            if any(success)
                curves = reshape(olstec_nre_trials(lam_idx, target_idx, r_idx, :, :), ...
                    num_trials, T);
                olstec_sens_ts_cum(lam_idx, :, r_idx) = sum(curves(success, :), 1);
                olstec_sens_ts_counts(lam_idx, :, r_idx) = repmat(sum(success), 1, T);
            end
        end
    end
end

fprintf('\nAll Cross-Ablation Experiments completed in %.1f seconds.\n', toc(total_start));
fprintf('Smooth auxiliary noise sigma = %.2f.\n', aux_noise_sigma);

%% 3. Visualization: Matrix Subplots
% -------------------------------------------------------------------------
fig_height = 350 * num_ranks;
figure('Name', 'Robustness against Sparsity & Rank', 'Position', [100, 50, 1200, fig_height], 'Color', 'w');

hex_colors = containers.Map(...
    {'RSI-OLSTEC', 'OLSTEC', 'GRASTA', 'GROUSE', 'PETRELS', 'TeCPSGD'}, ...
    {'#D95319',    '#0072BD', '#77AC30', '#EDB120', '#7E2F8E',  '#4DBEEE'} ...
);
colors = zeros(num_algs, 3);
for i = 1:num_algs
    current_hex = hex_colors(alg_names{i});
    colors(i, :) = sscanf(current_hex(2:end), '%2x%2x%2x', [1 3]) / 255;
end

x_axis = 1:tensor_dims(3);
alphabet = 'abcdefghijklmnopqrstuvwxyz';

for r_idx = 1:num_ranks
    rank_r = test_ranks(r_idx);

    % --- Left Column: Bar Charts (Macro Trend) ---
    subplot(num_ranks, 2, 2 * r_idx - 1); hold on; box on; grid on;

    cur_mean = mean_errors_3D(:,:,r_idx);
    cur_std  = std_errors_3D(:,:,r_idx);
    hb = bar(cur_mean');

    for k = 1:num_algs
        hb(k).FaceColor = colors(k,:); hb(k).EdgeColor = 'k'; hb(k).LineWidth = 1.0;
        if isprop(hb(k), 'XEndPoints'), x_pos = hb(k).XEndPoints; else, x_pos = hb(k).XData + hb(k).XOffset; end
        errorbar(x_pos, cur_mean(k,:), cur_std(k,:), 'k', 'LineStyle', 'none', 'LineWidth', 1.2, 'CapSize', 6);
    end
    set(gca, 'YScale', 'log', 'FontName', 'Times New Roman', 'FontSize', 12);
    set(gca, 'XTick', 1:num_fracs, 'XTickLabel', arrayfun(@(x) sprintf('%.0f%%', x*100), test_fractions, 'UniformOutput', false));
    ylabel('Steady-State NRE (log scale)', 'Interpreter', 'latex', 'FontSize', 13);

    title_str = sprintf('(%s) Macro Trend (Rank = %d)', alphabet(2 * r_idx - 1), rank_r);
    title(title_str, 'Interpreter', 'latex', 'FontSize', 13, 'FontWeight', 'bold');

    % Display legend only on the first row to avoid clutter
    if r_idx == 1, legend(hb, alg_names, 'Location', 'northwest', 'FontSize', 10); end
    if r_idx == num_ranks, xlabel('Observation Ratio ($\rho$)', 'Interpreter', 'latex', 'FontSize', 13); end
    positive_mean = cur_mean(cur_mean > 0);
    if ~isempty(positive_mean)
        ylim([min(positive_mean) * 0.5, max(cur_mean(:), [], 'omitnan') * 1.5]);
    end

    % --- Right Column: Time-Series Curves (Micro Convergence) ---
    subplot(num_ranks, 2, 2 * r_idx); hold on; box on; grid on;

    cur_ts = time_series_3D(:,:,r_idx);
    for a = 1:num_algs
        % Explicitly label OLSTEC as lambda = 0.80 in the legend
        if strcmp(alg_names{a}, 'OLSTEC')
            leg_name = 'OLSTEC (\lambda=0.80)';
        else
            leg_name = alg_names{a};
        end
        plot(x_axis, cur_ts(a, :), 'Color', colors(a,:), 'LineWidth', 1.5, 'DisplayName', leg_name);
    end

    set(gca, 'YScale', 'log', 'FontName', 'Times New Roman', 'FontSize', 12);
    ylabel('Mean NRE (log scale)', 'Interpreter', 'latex', 'FontSize', 13);

    title_str = sprintf('(%s) Micro Convergence at $\\rho=%.0f\\%%$ (Rank = %d)', alphabet(2 * r_idx), target_ts_fraction*100, rank_r);
    title(title_str, 'Interpreter', 'latex', 'FontSize', 13, 'FontWeight', 'bold');

    if r_idx == 1, legend('Location', 'northeast', 'FontSize', 10); end
    if r_idx == num_ranks, xlabel('Time Index (Frames)', 'Interpreter', 'latex', 'FontSize', 13); end
    xlim([1, tensor_dims(3)]);
    positive_ts = cur_ts(cur_ts > 0);
    if ~isempty(positive_ts)
        ylim([min(positive_ts) * 0.5, max(cur_ts(:), [], 'omitnan') * 1.5]);
    end
end
fprintf('Cross-ablation figure generated.\n');

fprintf('\n=================================================================================\n');
fprintf('   EXP S4 RUN SUMMARY: VALID TRIAL COUNTS / FAILURE COUNTS\n');
fprintf('=================================================================================\n');
for r_idx = 1:num_ranks
    fprintf('Rank = %d\n', test_ranks(r_idx));
    for exp_idx = 1:num_fracs
        fprintf('  Observation %.0f%%: ', test_fractions(exp_idx) * 100);
        for a = 1:num_algs
            fprintf('%s=%d/%d fail=%d; ', alg_names{a}, valid_counts_3D(a, exp_idx, r_idx), num_trials, failure_counts_3D(a, exp_idx, r_idx));
        end
        lambda_mean_vals = squeeze(rsi_lambda_mean_trials(exp_idx, r_idx, :));
        lambda_low_vals = squeeze(rsi_lambda_below_max_pct_trials(exp_idx, r_idx, :));
        fprintf('RSI lambda mean=%.4f, lambda<0.80%%=%.2f; ', ...
            mean(lambda_mean_vals, 'omitnan'), mean(lambda_low_vals, 'omitnan'));
        fprintf('\n');
    end
end

num_summary_rows = num_algs * num_fracs * num_ranks;
summary_algorithm = cell(num_summary_rows, 1);
summary_rank = zeros(num_summary_rows, 1);
summary_fraction = zeros(num_summary_rows, 1);
summary_valid = zeros(num_summary_rows, 1);
summary_failed = zeros(num_summary_rows, 1);
row = 0;
for r_idx = 1:num_ranks
    for exp_idx = 1:num_fracs
        for a = 1:num_algs
            row = row + 1;
            summary_algorithm{row} = alg_names{a};
            summary_rank(row) = test_ranks(r_idx);
            summary_fraction(row) = test_fractions(exp_idx);
            summary_valid(row) = valid_counts_3D(a, exp_idx, r_idx);
            summary_failed(row) = failure_counts_3D(a, exp_idx, r_idx);
        end
    end
end
failure_summary = table(summary_algorithm, summary_rank, summary_fraction, ...
    summary_valid, summary_failed, 'VariableNames', ...
    {'Algorithm', 'Rank', 'ObservationRatio', 'ValidRuns', 'FailedRuns'});

num_olstec_summary_rows = num_olstec_lams * num_fracs * num_ranks;
olstec_summary_lambda = zeros(num_olstec_summary_rows, 1);
olstec_summary_rank = zeros(num_olstec_summary_rows, 1);
olstec_summary_fraction = zeros(num_olstec_summary_rows, 1);
olstec_summary_valid = zeros(num_olstec_summary_rows, 1);
olstec_summary_failed = zeros(num_olstec_summary_rows, 1);
row = 0;
for r_idx = 1:num_ranks
    for exp_idx = 1:num_fracs
        for lam_idx = 1:num_olstec_lams
            row = row + 1;
            status = reshape(olstec_status(lam_idx, exp_idx, r_idx, :), num_trials, 1);
            olstec_summary_lambda(row) = lambda_list_olstec(lam_idx);
            olstec_summary_rank(row) = test_ranks(r_idx);
            olstec_summary_fraction(row) = test_fractions(exp_idx);
            olstec_summary_valid(row) = sum(status == 1);
            olstec_summary_failed(row) = sum(status == 2);
        end
    end
end
olstec_failure_summary = table(olstec_summary_lambda, olstec_summary_rank, ...
    olstec_summary_fraction, olstec_summary_valid, olstec_summary_failed, ...
    'VariableNames', {'Lambda', 'Rank', 'ObservationRatio', 'ValidRuns', 'FailedRuns'});

if export_results
    failure_log = struct2table(failure_records);
    save(fullfile(result_dir, 'S4_stats.mat'), ...
        'aux_noise_sigma', 'matrix_rank_by_rank', 'tensor_dims', 'sparse_ratio', ...
        'SNR_dB', 'tolcost', 'steady_state_window', ...
        'test_fractions', 'test_ranks', 'target_ts_fraction', 'num_trials', ...
        'alg_names', 'mean_errors_3D', 'std_errors_3D', 'time_series_3D', ...
        'valid_counts_3D', 'failure_counts_3D', 'nre_trials', ...
        'steady_state_errors', 'algorithm_status', 'lambda_list_olstec', ...
        'olstec_nre_trials', 'olstec_status', ...
        'olstec_sens_ts_cum', 'olstec_sens_ts_counts', ...
        'rsi_lambda_mean_trials', 'rsi_lambda_below_max_pct_trials', ...
        'trial_seed_matrix', 'clean_seed_matrix', 'auxiliary_seed_matrix', ...
        'spatter_seed_matrix', 'gaussian_seed_matrix', 'mask_seed_matrix', ...
        'initialization_seed_matrix', 'matrix_initialization_seed_matrix', ...
        'algorithm_seed_4D', 'olstec_seed_4D', ...
        'failure_records', 'failure_summary', 'olstec_failure_summary');
    writetable(failure_log, fullfile(result_dir, 'S4_failure_log.csv'));
    writetable(failure_summary, fullfile(result_dir, 'S4_failure_summary.csv'));
    writetable(olstec_failure_summary, fullfile(result_dir, 'S4_olstec_failure_summary.csv'));
    savefig(gcf, fullfile(result_dir, 'S4.fig'));
    set(gcf, 'Units', 'Inches');
    pos = get(gcf, 'Position');
    set(gcf, 'PaperPositionMode', 'auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
    print(gcf, fullfile(result_dir, 'S4.pdf'), '-dpdf', '-vector');
    print(gcf, fullfile(result_dir, 'S4.eps'), '-depsc', '-vector');
end

function record = s4_failure_record(algorithm, rank_r, fraction, lambda, trial, ...
    trial_seed, execution_seed, ME, err_curve)
    first_invalid = find(~isfinite(err_curve) | err_curve < 0, 1, 'first');
    if isempty(first_invalid)
        first_invalid = NaN;
    end
    record = struct( ...
        'Algorithm', algorithm, ...
        'Rank', rank_r, ...
        'ObservationRatio', fraction, ...
        'Lambda', lambda, ...
        'Trial', trial, ...
        'TrialSeed', trial_seed, ...
        'ExecutionSeed', execution_seed, ...
        'Category', s4_failure_category(ME), ...
        'Identifier', ME.identifier, ...
        'Message', ME.message, ...
        'OutputLength', numel(err_curve), ...
        'FirstInvalidFrame', first_invalid);
end

function tf = s4_is_process_interruption(ME)
    identifier = lower(ME.identifier);
    message = lower(ME.message);
    tf = contains(identifier, 'operationterminatedbyuser') || ...
        contains(identifier, 'nomem') || ...
        contains(identifier, 'outofmemory') || ...
        contains(message, 'operation terminated by user') || ...
        contains(message, 'out of memory') || ...
        contains(message, 'memory allocation') || ...
        contains(message, 'requested array exceeds') || ...
        contains(message, 'unable to allocate');
end

function category = s4_failure_category(ME)
    if startsWith(ME.identifier, 'validate_complete_nre:')
        category = 'invalid_nre_output';
    else
        text = lower([ME.identifier, ' ', ME.message]);
        if contains(text, 'singular') || contains(text, 'ill-conditioned') || ...
                contains(text, 'not positive definite')
            category = 'numerical_failure';
        else
            category = 'algorithm_error';
        end
    end
end

function Q = s4_orthonormalize(M)
    [Q_raw, R_raw] = qr(M, 0);
    signs = sign(diag(R_raw) + 1e-10)';
    Q = Q_raw * diag(signs);
end
