%% Experiment S3 LN Gaussian-Only
% =========================================================================
% Prerequisites: Run `run_me_first.m` once to add all dependencies to path.
%
% Description:
%   This script is a no-spatter ablation of the event-driven S3 benchmark.
%   It removes the sparse impulsive corruption term E_t while preserving the
%   smooth dynamics, event-driven subspace changes, Gaussian noise, sampling
%   ratio, auxiliary signal, and algorithm settings.
% =========================================================================
clear; clc; close all;

%% 1. Global Configuration
% -------------------------------------------------------------------------
fprintf('Starting Synthetic Experiment S3 LN Gaussian-Only...\n');

% Experiment Parameters
tensor_dims     = [50, 50, 500];  % [rows, cols, frames]
rank_r          = 15;             % True tensor CP-rank
fraction        = 0.10;           % Observation ratio (10%)
sparse_ratio    = 0.00;           % Sparse impulsive noise ratio removed for this ablation
SNR_dB          = 25;
tolcost         = 1e-8;
maxepochs       = 1;
verbose         = 0;
n_monte_carlo   = 50;           % Number of Monte Carlo trials.
export_results  = true;
result_dir      = fullfile(fileparts(mfilename('fullpath')), 'result', 'S3_LN_GaussianOnly');
if export_results && ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

% Algorithm Execution Switches
run_cpwopt      = true;
run_petrels     = true;
run_grasta      = true;
run_grouse      = true;
run_tecpsgd     = true;
run_olstec      = true;
run_rsi_olstec  = true;

run_s3_monte_carlo(tensor_dims, rank_r, fraction, sparse_ratio, ...
    SNR_dB, tolcost, maxepochs, verbose, n_monte_carlo, export_results, ...
    result_dir, run_cpwopt, run_petrels, run_grasta, run_grouse, ...
    run_tecpsgd, run_olstec, run_rsi_olstec);

function run_s3_monte_carlo(tensor_dims, rank_r, fraction, sparse_ratio, ...
    SNR_dB, tolcost, maxepochs, verbose, n_monte_carlo, export_results, result_dir, ...
    run_cpwopt, run_petrels, run_grasta, run_grouse, run_tecpsgd, run_olstec, run_rsi_olstec)

    fprintf('=== [S3 Monte Carlo Protocol] Trials=%d, Rank=%d, Obs=%.1f%% ===\n', ...
        n_monte_carlo, rank_r, fraction * 100);

    rows = tensor_dims(1);
    cols = tensor_dims(2);
    total_slices = tensor_dims(3);
    numr = rows * cols;
    numc = total_slices;
    matrix_rank = rank_r;
    lambda_list = [0.70, 0.80, 0.90, 0.99];
    aux_noise_sigma = 0.2;
    aux_step_gain = 10.0;
    aux_event_decay = 20;
    condition_name = 'Gaussian-only';
    fprintf('Matrix baseline rank set to %d.\n', matrix_rank);

    max_num_algs = double(run_cpwopt) + double(run_grouse) + double(run_grasta) + ...
        double(run_petrels) + double(run_tecpsgd) + double(run_olstec) * numel(lambda_list) + ...
        double(run_rsi_olstec);
    alg_names = cell(1, max_num_algs);
    algorithm_seed_offsets = zeros(1, max_num_algs);
    alg_count = 0;
    idx_cpwopt = [];
    idx_petrels = [];
    idx_grasta = [];
    idx_grouse = [];
    idx_tecpsgd = [];
    idx_olstec = zeros(1, numel(lambda_list));
    idx_rsi = [];

    if run_cpwopt
        alg_count = alg_count + 1;
        alg_names{alg_count} = 'CP-WOPT (batch)';
        algorithm_seed_offsets(alg_count) = 1;
        idx_cpwopt = alg_count;
    end
    if run_grouse
        alg_count = alg_count + 1;
        alg_names{alg_count} = 'GROUSE';
        algorithm_seed_offsets(alg_count) = 2;
        idx_grouse = alg_count;
    end
    if run_grasta
        alg_count = alg_count + 1;
        alg_names{alg_count} = 'GRASTA';
        algorithm_seed_offsets(alg_count) = 3;
        idx_grasta = alg_count;
    end
    if run_petrels
        alg_count = alg_count + 1;
        alg_names{alg_count} = 'PETRELS';
        algorithm_seed_offsets(alg_count) = 4;
        idx_petrels = alg_count;
    end
    if run_tecpsgd
        alg_count = alg_count + 1;
        alg_names{alg_count} = 'TeCPSGD';
        algorithm_seed_offsets(alg_count) = 5;
        idx_tecpsgd = alg_count;
    end
    if run_olstec
        for i = 1:numel(lambda_list)
            alg_count = alg_count + 1;
            alg_names{alg_count} = sprintf('OLSTEC (lambda=%.2f)', lambda_list(i));
            algorithm_seed_offsets(alg_count) = 5 + i;
            idx_olstec(i) = alg_count;
        end
    end
    if run_rsi_olstec
        alg_count = alg_count + 1;
        alg_names{alg_count} = 'RSI-OLSTEC (full)';
        algorithm_seed_offsets(alg_count) = 10;
        idx_rsi = alg_count;
    end
    alg_names = alg_names(1:alg_count);
    algorithm_seed_offsets = algorithm_seed_offsets(1:alg_count);

    num_algs = numel(alg_names);
    err_sum = zeros(num_algs, total_slices);
    err_count = zeros(num_algs, total_slices);
    final_errors = NaN(num_algs, n_monte_carlo);
    trial_mean_errors = NaN(num_algs, n_monte_carlo);
    trial_auc_errors = NaN(num_algs, n_monte_carlo);
    rsi_lambda_mean = NaN(1, n_monte_carlo);
    rsi_lambda_below_max_pct = NaN(1, n_monte_carlo);
    nre_trials = NaN(num_algs, n_monte_carlo, total_slices);
    algorithm_status = zeros(num_algs, n_monte_carlo, 'uint8');
    failure_records = repmat(struct( ...
        'Condition', '', 'Algorithm', '', 'Trial', NaN, ...
        'TrialSeed', NaN, 'ExecutionSeed', NaN, 'Category', '', ...
        'Identifier', '', 'Message', '', 'OutputLength', NaN, ...
        'FirstInvalidFrame', NaN), 0, 1);
    seed_list = 42 + (0:n_monte_carlo-1);
    clean_seed_list = 100000 + seed_list;
    auxiliary_seed_list = 200000 + seed_list;
    gaussian_seed_list = 400000 + seed_list;
    mask_seed_list = 500000 + seed_list;
    initialization_seed_list = 600000 + seed_list;
    matrix_initialization_seed_list = 650000 + seed_list;
    algorithm_seed_list = bsxfun(@plus, 700000 + 100 * seed_list(:), algorithm_seed_offsets);
    if run_grasta
        algorithm_seed_list(:, idx_grasta) = matrix_initialization_seed_list(:);
    end

    for trial = 1:n_monte_carlo
        fprintf('  Trial %d/%d (seed=%d)\n', trial, n_monte_carlo, seed_list(trial));
        rng(clean_seed_list(trial), 'twister');

        A_true = s3_orthonormalize(randn(rows, rank_r));
        B_true = s3_orthonormalize(randn(cols, rank_r));
        C_true = zeros(total_slices, rank_r);
        t = (1:total_slices)';
        for r = 1:rank_r
            C_true(:, r) = 10.0 + 2.0 * sin(2 * pi * t / (100 + r*10)) + 0.1 * randn(total_slices, 1);
        end

        num_events = 2;
        event_frames = sort([randi([140, 220]), randi([300, 390])]);
        event_strengths = 0.50 + 0.50 * rand(num_events, 1);
        event_delays = randi([0, 3], num_events, 1);

        Tensor_Y_Clean = zeros(rows, cols, total_slices);
        next_event = 1;
        for f = 1:total_slices
            if next_event <= num_events && f == event_frames(next_event)
                A_true = s3_orthonormalize(A_true + event_strengths(next_event) * randn(rows, rank_r));
                B_true = s3_orthonormalize(B_true + event_strengths(next_event) * randn(cols, rank_r));
                next_event = next_event + 1;
            end
            Tensor_Y_Clean(:, :, f) = A_true * diag(C_true(f, :)) * B_true';
        end
        rng(auxiliary_seed_list(trial), 'twister');
        aux_info = s3_event_aux_signal(total_slices, event_frames, event_strengths, ...
            event_delays, aux_step_gain, aux_event_decay, aux_noise_sigma);
        burn_in_aux = min(30, length(aux_info));
        diff_aux = diff(aux_info(1:burn_in_aux));
        mad_aux = median(abs(diff_aux - median(diff_aux)));
        est_aux_sigma = (1.4826 * mad_aux) / sqrt(2);
        adaptive_min_grad = max(0.05, 3 * sqrt(2) * est_aux_sigma);

        sig_pow = norm(Tensor_Y_Clean(:))^2 / numel(Tensor_Y_Clean);
        noise_sigma = sqrt(sig_pow / 10^(SNR_dB/10));
        rng(gaussian_seed_list(trial), 'twister');
        Gaussian_Noise = noise_sigma * randn(rows, cols, total_slices);
        Tensor_Y_Noisy = Tensor_Y_Clean + Gaussian_Noise;

        rng(mask_seed_list(trial), 'twister');
        OmegaTensor = rand(rows, cols, total_slices) < fraction;
        Matrix_Y_Noisy = reshape(Tensor_Y_Noisy, [numr, numc]);
        OmegaMatrix = reshape(OmegaTensor, [numr, numc]);

        burn_in = min(30, total_slices);
        diff_pixels = cell(burn_in - 1, 1);
        for t_idx = 2:burn_in
            common_mask = OmegaTensor(:, :, t_idx) & OmegaTensor(:, :, t_idx-1);
            diff_frame = Tensor_Y_Noisy(:, :, t_idx) - Tensor_Y_Noisy(:, :, t_idx-1);
            diff_pixels{t_idx-1} = diff_frame(common_mask);
        end
        diff_pixels = vertcat(diff_pixels{:});
        if ~isempty(diff_pixels)
            mad_val = median(abs(diff_pixels - median(diff_pixels)));
            est_sigma = (1.4826 * mad_val) / sqrt(2);
            huber_delta = max(0.01, 3 * est_sigma);
        else
            huber_delta = 0.05;
        end

        rng(initialization_seed_list(trial), 'twister');
        Xinit.A = randn(rows, rank_r);
        Xinit.B = randn(cols, rank_r);
        Xinit.C = randn(total_slices, rank_r);

        % Matrix methods share one initial column space within each trial.
        matrix_init_stream = RandStream('mt19937ar', ...
            'Seed', matrix_initialization_seed_list(trial));
        matrix_init = struct();
        matrix_init.U = randn(matrix_init_stream, numr, matrix_rank);
        matrix_init.Weight = randn(matrix_init_stream, matrix_rank, numc);

        if run_cpwopt
            err_curve = [];
            try
                rng(algorithm_seed_list(trial, idx_cpwopt), 'twister');
                opts = struct('maxepochs', 30, 'display_iters', 1, 'tolcost', tolcost, ...
                    'verbose', verbose, 'store_matrix', false, 'store_subinfo', true);
                [Xsol_cp, ~, ~] = cp_wopt_mod(Tensor_Y_Noisy, OmegaTensor, [], tensor_dims, rank_r, Xinit, opts);
                L_cp_full = zeros(rows, cols, total_slices);
                for f = 1:total_slices
                    L_cp_full(:, :, f) = Xsol_cp.A * diag(Xsol_cp.C(f, :)) * Xsol_cp.B';
                end
                err_curve = compute_true_nre_tensor(Tensor_Y_Clean, L_cp_full);
                s3_record_curve(idx_cpwopt, err_curve);
            catch ME
                s3_record_failure(idx_cpwopt, ME, err_curve);
            end
        end

        if run_grouse
            err_curve = [];
            try
                rng(algorithm_seed_list(trial, idx_grouse), 'twister');
                opts = struct('maxepochs', maxepochs, 'tolcost', tolcost, 'permute_on', false, ...
                    'maxrank', matrix_rank, 'step_size', 0.0001, 'verbose', 0, ...
                    'store_matrix', true, 'store_subinfo', true);
                [~, ~, info] = grouse_mod(matrix_init, Matrix_Y_Noisy, ...
                    OmegaMatrix, [], numr, numc, opts);
                err_curve = compute_true_nre_tensor(Tensor_Y_Clean, info.L);
                s3_record_curve(idx_grouse, err_curve);
            catch ME
                s3_record_failure(idx_grouse, ME, err_curve);
            end
        end

        if run_grasta
            err_curve = [];
            try
                % GRASTA reconstructs the shared column space internally.
                rng(algorithm_seed_list(trial, idx_grasta), 'twister');
                opts = struct('maxepochs', maxepochs, 'tolcost', tolcost, 'permute_on', false, ...
                    'RANK', matrix_rank, 'rho', 1.8, 'ITER_MAX', 20, 'MAX_MU', 10000, ...
                    'MIN_MU', 1, 'DIM_M', numr, 'USE_MEX', 0, 'verbose', 0, ...
                    'store_matrix', true, 'store_subinfo', true);
                [~, ~, info] = grasta_mod(matrix_init, Matrix_Y_Noisy, ...
                    OmegaMatrix, [], numr, numc, opts);
                err_curve = compute_true_nre_tensor(Tensor_Y_Clean, info.L);
                s3_record_curve(idx_grasta, err_curve);
            catch ME
                s3_record_failure(idx_grasta, ME, err_curve);
            end
        end

        if run_petrels
            err_curve = [];
            try
                rng(algorithm_seed_list(trial, idx_petrels), 'twister');
                opts = struct('maxepochs', maxepochs, 'tolcost', tolcost, 'permute_on', false, ...
                    'rank', matrix_rank, 'lambda', 0.98, 'verbose', 0, ...
                    'store_matrix', true, 'store_subinfo', true);
                [~, ~, info] = petrels_mod(matrix_init, Matrix_Y_Noisy, ...
                    OmegaMatrix, [], numr, numc, opts);
                err_curve = compute_true_nre_tensor(Tensor_Y_Clean, info.L);
                s3_record_curve(idx_petrels, err_curve);
            catch ME
                s3_record_failure(idx_petrels, ME, err_curve);
            end
        end

        if run_tecpsgd
            err_curve = [];
            try
                rng(algorithm_seed_list(trial, idx_tecpsgd), 'twister');
                opts = struct('maxepochs', maxepochs, 'tolcost', tolcost, 'permute_on', false, ...
                    'lambda', 0.99, 'stepsize', 0.1, 'mu', 0.01, 'verbose', 0, ...
                    'store_matrix', true, 'store_subinfo', true);
                [~, ~, info] = TeCPSGD(Tensor_Y_Noisy, OmegaTensor, [], tensor_dims, rank_r, Xinit, opts);
                if ndims(info.L) == 3
                    L_mat = reshape(info.L, [numr, numc]);
                else
                    L_mat = info.L;
                end
                err_curve = compute_true_nre_tensor(Tensor_Y_Clean, L_mat);
                s3_record_curve(idx_tecpsgd, err_curve);
            catch ME
                s3_record_failure(idx_tecpsgd, ME, err_curve);
            end
        end

        if run_olstec
            for i = 1:numel(lambda_list)
                err_curve = [];
                try
                    rng(algorithm_seed_list(trial, idx_olstec(i)), 'twister');
                    lam = lambda_list(i);
                    opts = struct('maxepochs', maxepochs, 'tolcost', tolcost, 'permute_on', false, ...
                        'lambda', lam, 'mu', 0.01, 'tw_flag', 0, 'tw_len', 10, ...
                        'verbose', 0, 'store_matrix', true, 'store_subinfo', true);
                    [~, ~, info] = olstec(Tensor_Y_Noisy, OmegaTensor, [], tensor_dims, rank_r, Xinit, opts);
                    err_curve = compute_true_nre_tensor(Tensor_Y_Clean, info.L);
                    s3_record_curve(idx_olstec(i), err_curve);
                catch ME
                    s3_record_failure(idx_olstec(i), ME, err_curve);
                end
            end
        end

        if run_rsi_olstec
            err_curve = [];
            try
                rng(algorithm_seed_list(trial, idx_rsi), 'twister');
                opts = struct('maxepochs', maxepochs, 'tolcost', tolcost, 'permute_on', false, ...
                    'lambda_max', 0.80, 'lambda_min', 0.70, 'huber_delta', huber_delta, ...
                    'min_grad_threshold', adaptive_min_grad, 'mu', 0.01, 'verbose', 0, ...
                    'store_matrix', true, 'store_subinfo', true);
                [~, ~, info] = rsi_olstec(Tensor_Y_Noisy, OmegaTensor, [], tensor_dims, rank_r, Xinit, opts, aux_info);
                err_curve = compute_true_nre_tensor(Tensor_Y_Clean, info.L);
                lambda_vals = info.lambda_history(:);
                lambda_vals = lambda_vals(isfinite(lambda_vals));
                if ~isempty(lambda_vals)
                    rsi_lambda_mean(trial) = mean(lambda_vals);
                    rsi_lambda_below_max_pct(trial) = 100 * mean(lambda_vals < opts.lambda_max);
                end
                s3_record_curve(idx_rsi, err_curve);
            catch ME
                s3_record_failure(idx_rsi, ME, err_curve);
            end
        end
    end

    if any(algorithm_status(:) == 0)
        error('Exp_S3_LN:IncompleteExecutionState', ...
            'At least one configured algorithm run has no success or failure status.');
    end
    valid_counts = sum(algorithm_status == 1, 2);
    failure_counts = sum(algorithm_status == 2, 2);
    pending_counts = sum(algorithm_status == 0, 2);
    failure_summary = table(alg_names(:), valid_counts, failure_counts, pending_counts, ...
        'VariableNames', {'Algorithm', 'ValidRuns', 'FailedRuns', 'PendingRuns'});

    mean_err = err_sum ./ max(1, err_count);
    mean_err(err_count == 0) = NaN;
    std_err = reshape(std(nre_trials, 0, 2, 'omitnan'), ...
        num_algs, total_slices);
    std_err(err_count <= 1) = NaN;

    x_axis = 1:total_slices;
    fig = figure('Name', 'Synthetic: Monte Carlo True Residual Error', ...
        'Position', [100, 100, 900, 560], 'Color', 'w');
    hold on; grid on; box on;

    base_colors = lines(max(num_algs, 7));
    olstec_styles = {':', '--', '-', '-.'};
    p_handles = gobjects(num_algs, 1);
    leg_str = cell(num_algs, 1);
    for a = 1:num_algs
        name = alg_names{a};
        color = base_colors(a, :);
        line_style = '-';
        line_width = 1.5;
        if contains(name, 'CP-WOPT')
            color = [0 0 0];
            line_style = '--';
            line_width = 2.0;
        elseif contains(name, 'RSI-OLSTEC')
            color = s3_hex_to_rgb('#D95319');
            line_width = 2.2;
        elseif contains(name, 'OLSTEC')
            color = s3_hex_to_rgb('#0072BD');
            line_style = olstec_styles{min(sum(idx_olstec <= a & idx_olstec > 0), numel(olstec_styles))};
        elseif contains(name, 'GRASTA')
            color = s3_hex_to_rgb('#77AC30');
        elseif contains(name, 'PETRELS')
            color = s3_hex_to_rgb('#7E2F8E');
        elseif contains(name, 'TeCPSGD')
            color = s3_hex_to_rgb('#4DBEEE');
        elseif contains(name, 'GROUSE')
            color = s3_hex_to_rgb('#EDB120');
        end
        h = semilogy(x_axis, mean_err(a, :), 'LineStyle', line_style, ...
            'Color', color, 'LineWidth', line_width);
        p_handles(a) = h;
        leg_str{a} = name;
    end

    set(gca, 'YScale', 'log', 'FontName', 'Times New Roman', 'FontSize', 12, 'TickLabelInterpreter', 'latex');
    xlabel('Time Index (Frames)', 'Interpreter', 'latex', 'FontSize', 14);
    ylabel('Mean NRE (log scale)', 'Interpreter', 'latex', 'FontSize', 14);
    title(sprintf('S3 LN Gaussian-Only Benchmark with Smooth Dynamics and Event Changes (n=%d)', n_monte_carlo), ...
        'Interpreter', 'latex', 'FontSize', 14, 'FontWeight', 'bold');
    lgd = legend(p_handles, leg_str, 'Location', 'southwest', 'FontSize', 10);
    lgd.Box = 'off';
    positive_vals = mean_err(mean_err > 0 & isfinite(mean_err));
    if ~isempty(positive_vals)
        steady_data = mean_err(:, 50:end);
        steady_max = max(steady_data(isfinite(steady_data)));

        ylim([min(positive_vals) * 0.8, steady_max * 1.5]);
    end
    xlim([1, total_slices]);
    event_window_ranges = [140, 220; 300, 390];
    s3_mark_event_windows(gca, event_window_ranges);
    hold off;

    fprintf('\n===========================================================\n');
    fprintf('SYNTHETIC MONTE CARLO RESULTS: TRUE ERROR SUMMARY\n');
    fprintf('Auxiliary noise sigma: %.3f, event gain: %.3f, decay: %.1f frames\n', ...
        aux_noise_sigma, aux_step_gain, aux_event_decay);
    fprintf('Sparse impulsive spatter E_t removed (reported sparse ratio %.1f%%).\n', ...
        sparse_ratio * 100);
    fprintf('Matrix baseline rank: %d\n', matrix_rank);
    fprintf('RSI lambda mean: %.4f +/- %.4f, lambda<0.80%%: %.2f +/- %.2f\n', ...
        mean(rsi_lambda_mean, 'omitnan'), std(rsi_lambda_mean, 'omitnan'), ...
        mean(rsi_lambda_below_max_pct, 'omitnan'), std(rsi_lambda_below_max_pct, 'omitnan'));
    fprintf('===========================================================\n');
    fprintf('%-34s | %-21s | %-21s | %-21s | %-13s\n', ...
        'Algorithm', 'Final NRE', 'Mean NRE', 'AUC NRE', 'valid/fail');
    fprintf('---------------------------------------------------------------------------------------------------------------\n');
    for a = 1:num_algs
        final_vals = final_errors(a, :);
        mean_vals = trial_mean_errors(a, :);
        auc_vals = trial_auc_errors(a, :);
        final_vals = final_vals(~isnan(final_vals));
        mean_vals = mean_vals(~isnan(mean_vals));
        auc_vals = auc_vals(~isnan(auc_vals));
        valid_count = valid_counts(a);
        fprintf('%-34s | %.6e +/- %.6e | %.6e +/- %.6e | %.6e +/- %.6e | %3d/%-3d fail=%d\n', ...
            alg_names{a}, mean(final_vals), std(final_vals), ...
            mean(mean_vals), std(mean_vals), ...
            mean(auc_vals), std(auc_vals), ...
            valid_count, n_monte_carlo, failure_counts(a));
    end
    fprintf('===========================================================\n');

    if export_results
        failure_log = struct2table(failure_records);
        save(fullfile(result_dir, 'S3_LN_GaussianOnly_stats.mat'), ...
            'matrix_rank', 'tensor_dims', 'rank_r', 'fraction', 'sparse_ratio', 'SNR_dB', ...
            'tolcost', 'maxepochs', 'n_monte_carlo', 'seed_list', ...
            'clean_seed_list', 'auxiliary_seed_list', 'gaussian_seed_list', ...
            'mask_seed_list', 'initialization_seed_list', ...
            'matrix_initialization_seed_list', ...
            'algorithm_seed_offsets', 'algorithm_seed_list', ...
            'aux_noise_sigma', 'aux_step_gain', 'aux_event_decay', ...
            'lambda_list', 'alg_names', 'rsi_lambda_mean', 'rsi_lambda_below_max_pct', ...
            'mean_err', 'std_err', 'err_count', 'final_errors', ...
            'trial_mean_errors', 'trial_auc_errors', 'nre_trials', ...
            'algorithm_status', 'failure_records', 'failure_summary', 'failure_counts');
        writetable(failure_log, fullfile(result_dir, 'S3_LN_GaussianOnly_failure_log.csv'));
        writetable(failure_summary, fullfile(result_dir, 'S3_LN_GaussianOnly_failure_summary.csv'));
        savefig(fig, fullfile(result_dir, 'S3_LN_GaussianOnly.fig'));
        set(fig, 'Units', 'Inches');
        pos = get(fig, 'Position');
        set(fig, 'PaperPositionMode', 'auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
        print(fig, fullfile(result_dir, 'S3_LN_GaussianOnly.pdf'), '-dpdf', '-vector');
        print(fig, fullfile(result_dir, 'S3_LN_GaussianOnly.eps'), '-depsc', '-vector');
    end

    function s3_record_curve(alg_idx, err_curve)
        err_curve = validate_complete_nre(err_curve, total_slices, alg_names{alg_idx}, trial);
        nre_trials(alg_idx, trial, :) = reshape(err_curve, 1, 1, []);
        err_sum(alg_idx, :) = err_sum(alg_idx, :) + err_curve;
        err_count(alg_idx, :) = err_count(alg_idx, :) + 1;
        final_errors(alg_idx, trial) = err_curve(end);
        trial_mean_errors(alg_idx, trial) = mean(err_curve);
        if total_slices > 1
            trial_auc_errors(alg_idx, trial) = trapz(1:total_slices, err_curve) / (total_slices - 1);
        else
            trial_auc_errors(alg_idx, trial) = err_curve;
        end
        algorithm_status(alg_idx, trial) = 1;
    end

    function s3_record_failure(alg_idx, ME, err_curve)
        if s3_is_process_interruption(ME)
            rethrow(ME);
        end
        if ~isempty(idx_rsi) && alg_idx == idx_rsi
            rsi_lambda_mean(trial) = NaN;
            rsi_lambda_below_max_pct(trial) = NaN;
        end
        algorithm_status(alg_idx, trial) = 2;
        output_length = numel(err_curve);
        first_invalid = find(~isfinite(err_curve) | err_curve < 0, 1, 'first');
        if isempty(first_invalid)
            first_invalid = NaN;
        end
        failure_records(end + 1, 1) = struct( ...
            'Condition', condition_name, ...
            'Algorithm', alg_names{alg_idx}, ...
            'Trial', trial, ...
            'TrialSeed', seed_list(trial), ...
            'ExecutionSeed', algorithm_seed_list(trial, alg_idx), ...
            'Category', s3_failure_category(ME), ...
            'Identifier', ME.identifier, ...
            'Message', ME.message, ...
            'OutputLength', output_length, ...
            'FirstInvalidFrame', first_invalid);
        fprintf('    %s failed in trial %d: %s\n', ...
            alg_names{alg_idx}, trial, ME.message);
    end
end

function tf = s3_is_process_interruption(ME)
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

function category = s3_failure_category(ME)
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

function Q = s3_orthonormalize(M)
    [Q_raw, R_raw] = qr(M, 0);
    signs = sign(diag(R_raw) + 1e-10)';
    Q = Q_raw * diag(signs);
end

function aux_signal = s3_event_aux_signal(T, event_frames, event_strengths, event_delays, step_gain, decay, noise_sigma)
    t = (1:T)';
    aux_clean = 10.0 + 2.0 * sin(2 * pi * t / 110);
    for event_idx = 1:numel(event_frames)
        start_frame = min(T, event_frames(event_idx) + event_delays(event_idx));
        tail = (start_frame:T)';
        aux_clean(tail) = aux_clean(tail) + ...
            step_gain * event_strengths(event_idx) .* exp(-(tail - start_frame) / decay);
    end
    aux_signal = aux_clean + noise_sigma * randn(T, 1);
end

function rgb = s3_hex_to_rgb(hex_str)
    rgb = sscanf(hex_str(2:end), '%2x%2x%2x', [1 3]) / 255;
end

function s3_mark_event_windows(ax, event_windows)
    axes(ax);
    yl = ylim(ax);
    if ~all(isfinite(yl)) || yl(1) <= 0 || yl(2) <= yl(1)
        return;
    end

    label_y = 10^(log10(yl(1)) + 0.12 * (log10(yl(2)) - log10(yl(1))));
    for event_idx = 1:size(event_windows, 1)
        x0 = event_windows(event_idx, 1);
        x1 = event_windows(event_idx, 2);

        h_patch = patch(ax, [x0 x1 x1 x0], [yl(1) yl(1) yl(2) yl(2)], ...
            [0.95 0.90 0.55], 'EdgeColor', 'none', 'FaceAlpha', 0.30, ...
            'HandleVisibility', 'off');
        if exist('uistack', 'file') == 2
            uistack(h_patch, 'bottom');
        end

        xline(ax, x0, ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 0.8, ...
            'HandleVisibility', 'off');
        xline(ax, x1, ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 0.8, ...
            'HandleVisibility', 'off');

        text(ax, mean([x0, x1]), label_y, sprintf('Event window %d', event_idx), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'FontName', 'Times New Roman', 'FontSize', 16, ...
            'Color', [0.25 0.25 0.25], 'HandleVisibility', 'off');
    end
    ylim(ax, yl);
end
