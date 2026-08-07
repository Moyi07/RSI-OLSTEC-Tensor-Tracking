%% Experiment S2
% =========================================================================
% Prerequisites: Run `run_me_first.m` once to add all dependencies to path.
%
% Objective:
%   Evaluate robust residual weighting under Gaussian noise and sparse
%   impulsive corruption.
%
% Experimental Design (Ablation Study):
%   This script compares two fixed-memory variants to isolate the effect of
%   robust residual weighting:
%   1. Huber-IRLS variant: Huber weighting enabled.
%   2. Squared-loss control: Huber weighting disabled.
%   Paired wall-clock and core-loop timings quantify the incremental cost of
%   robust reweighting within the same recursive implementation.
% =========================================================================
clear; close all; clc;

%% 1. Monte Carlo Configuration
% -------------------------------------------------------------------------
n_monte_carlo = 50;
disp(['Starting Monte Carlo simulation with ', num2str(n_monte_carlo), ' runs...']);

% Tensor Dimensions & Common Parameters
I = 100; J = 100; T = 500;
dims = [I, J, T];
true_rank = 5;

% Data Generation Parameters
SNR_dB = 25;
sparse_ratio = 0.05;
spatter_base_mag = 0.50;
observation_ratio = 0.50;
drift_rate = 1e-4;

% Prespecified computational analysis
timing_warmup_runs_per_variant = 1;
bootstrap_ci_level = 0.95;
bootstrap_resamples = 10000;
bootstrap_seed = 20260729;

% Prespecified residual-map visualization
residual_color_percentile = 99;

% Storage for Statistics
history_err_robust = zeros(T, n_monte_carlo);
history_err_l2     = zeros(T, n_monte_carlo);
history_f1         = zeros(n_monte_carlo, 1);
history_f1_l2      = zeros(n_monte_carlo, 1);
history_wall_time_robust = zeros(n_monte_carlo, 1);
history_wall_time_l2 = zeros(n_monte_carlo, 1);
history_core_time_robust = zeros(n_monte_carlo, 1);
history_core_time_l2 = zeros(n_monte_carlo, 1);
history_prior_irls_robust = zeros(n_monte_carlo, 1);
history_prior_irls_l2 = zeros(n_monte_carlo, 1);
history_posterior_irls_robust = zeros(n_monte_carlo, 1);
history_posterior_irls_l2 = zeros(n_monte_carlo, 1);
huber_thresholds = zeros(n_monte_carlo, 1);
execution_order = cell(n_monte_carlo, 1);
snapshot_idx = T - 50;
snapshot_gt = zeros(I, J, n_monte_carlo);
snapshot_res_rob = zeros(I, J, n_monte_carlo);
snapshot_res_l2 = zeros(I, J, n_monte_carlo);
snapshot_nre_rob = zeros(n_monte_carlo, 1);
snapshot_nre_l2 = zeros(n_monte_carlo, 1);
export_results = true;
result_dir = fullfile(fileparts(mfilename('fullpath')), 'result', 'S2');
if export_results && ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

%% 2. Main Monte Carlo Loop
% -------------------------------------------------------------------------
for mc = 1:n_monte_carlo
    rng(mc, 'twister');

    % --- A. Physics-Informed Synthetic Data Generation ---
    % Force sign alignment during initialization
    A_curr = randn(I, true_rank); [Q_A, R_A] = qr(A_curr, 0);
    A_curr = Q_A * diag(sign(diag(R_A) + 1e-10));

    B_curr = randn(J, true_rank); [Q_B, R_B] = qr(B_curr, 0);
    B_curr = Q_B * diag(sign(diag(R_B) + 1e-10));

    % Pre-generate temporal sine dynamics
    C_true = zeros(T, true_rank); t_idx = (1:T)';
    for r = 1:true_rank
        % Use sinusoidal dynamics consistent with S1/S3/S4, plus a 10.0 offset for stable SNR.
        C_true(:, r) = 10.0 + 2.0 * sin(2 * pi * t_idx / (100 + r*10)) + 0.1 * randn(T, 1);
    end

    X_true = zeros(I, J, T);

    for t = 1:T
        % 1. Spatial drift: QR decomposition with sign protection for manifold continuity
        [Q_A, R_A] = qr(A_curr + drift_rate * randn(I, true_rank), 0);
        A_curr = Q_A * diag(sign(diag(R_A) + 1e-10));

        [Q_B, R_B] = qr(B_curr + drift_rate * randn(J, true_rank), 0);
        B_curr = Q_B * diag(sign(diag(R_B) + 1e-10));

        % 2. Temporal evolution: Merge spatial drift with sine dynamics
        X_true(:, :, t) = A_curr * diag(C_true(t, :)) * B_curr';
    end

    % Calculate dynamic background noise based on Signal Power
    sig_pow = norm(X_true(:))^2 / numel(X_true);
    noise_sigma = sqrt(sig_pow / 10^(SNR_dB/10));

    % Background Sensor Noise
    Gaussian_Noise = noise_sigma * randn(I, J, T);

    % Sparse Impulsive Noise (Simulating WAAM Spatter)
    S_mask = rand(I, J, T) < sparse_ratio;
    burn_in_frames = 30;

    Sparse_Noise = zeros(I, J, T);
    num_outliers = sum(S_mask(:));

    Sparse_Noise(S_mask) = spatter_base_mag * (1 + abs(randn(num_outliers, 1)));

    % Final Observation Tensor
    Y_full = X_true + Gaussian_Noise + Sparse_Noise;
    Omega = rand(I, J, T) < observation_ratio;
    Y_observed = Y_full .* Omega;

    Aux_Signal = 10.0 * ones(T, 1);

    % --- B. Algorithm Initialization ---
    [X_init.A, ~] = qr(randn(I, true_rank), 0);
    [X_init.B, ~] = qr(randn(J, true_rank), 0);
    X_init.C = randn(T, true_rank);

    % Unsupervised Temporal-Difference MAD Estimation (from observed data only)
    % Use Y_observed to strictly avoid clean-ground-truth leakage.
    diff_pixels = cell(burn_in_frames - 1, 1);
    for t_idx = 2:burn_in_frames
        common_mask = Omega(:,:,t_idx) & Omega(:,:,t_idx-1);
        diff_val = Y_observed(:,:,t_idx) - Y_observed(:,:,t_idx-1);
        diff_pixels{t_idx-1} = diff_val(common_mask);
    end
    diff_pixels = vertcat(diff_pixels{:});

    if ~isempty(diff_pixels)
        mad_val = median(abs(diff_pixels - median(diff_pixels)));
        est_sigma = (1.4826 * mad_val) / sqrt(2);
        huber_threshold_est = max(0.01, 3 * est_sigma);
    else
        huber_threshold_est = 0.05;
    end

    % Fix the forgetting factor to isolate robust residual weighting.
    opts_common = struct('mu', 0.01, 'verbose', 0, ...
                         'lambda_max', 0.80, 'lambda_min', 0.80, ...
                         'min_grad_threshold', Inf, ...
                         'irls_max_iters', 3, 'irls_tolerance', 1e-3, ...
                         'store_matrix', true, 'store_subinfo', false, ...
                         'early_stop_on', 'none');

    % Config 1: Huber-IRLS variant
    opts_robust = opts_common;
    opts_robust.huber_delta = huber_threshold_est;

    % Config 2: implementation-matched squared-loss control
    opts_l2 = opts_common;
    opts_l2.huber_delta = Inf;

    % --- C. Evaluation Data ---
    Omega_vec = reshape(Omega, [I*J, T]);
    Significant_Spatter = abs(Sparse_Noise) > (3 * noise_sigma);
    True_Outliers = reshape(Significant_Spatter, [I*J, T]) & Omega_vec;
    Mat_Y = reshape(Y_observed, [I*J, T]);
    eval_mask = false(1, T); eval_mask((burn_in_frames + 1):T) = true;
    evaluation = struct( ...
        'reference_tensor', X_true, ...
        'observed_matrix', Mat_Y, ...
        'observation_mask', Omega_vec, ...
        'true_outliers', True_Outliers, ...
        'evaluation_frames', eval_mask, ...
        'outlier_threshold', huber_threshold_est, ...
        'snapshot_index', snapshot_idx, ...
        'trial', mc);

    % Warm both computational paths once before collecting formal timings.
    if mc == 1
        fprintf('Warming Huber-IRLS and squared-loss execution paths...\n');
        for warmup_run = 1:timing_warmup_runs_per_variant
            [~, ~, ~] = rsi_olstec(Y_observed, Omega, [], dims, ...
                true_rank, X_init, opts_robust, Aux_Signal);
            [~, ~, ~] = rsi_olstec(Y_observed, Omega, [], dims, ...
                true_rank, X_init, opts_l2, Aux_Signal);
        end
    end

    % --- D. Counterbalanced Execution and Performance Evaluation ---
    if mod(mc, 2) == 1
        execution_order{mc} = 'HuberFirst';
        robust_result = run_s2_variant(Y_observed, Omega, dims, ...
            true_rank, X_init, opts_robust, Aux_Signal, evaluation, ...
            'Huber-IRLS variant');
        l2_result = run_s2_variant(Y_observed, Omega, dims, ...
            true_rank, X_init, opts_l2, Aux_Signal, evaluation, ...
            'Squared-loss control');
    else
        execution_order{mc} = 'SquaredLossFirst';
        l2_result = run_s2_variant(Y_observed, Omega, dims, ...
            true_rank, X_init, opts_l2, Aux_Signal, evaluation, ...
            'Squared-loss control');
        robust_result = run_s2_variant(Y_observed, Omega, dims, ...
            true_rank, X_init, opts_robust, Aux_Signal, evaluation, ...
            'Huber-IRLS variant');
    end

    history_err_robust(:, mc) = robust_result.nre;
    history_err_l2(:, mc) = l2_result.nre;
    history_f1(mc) = robust_result.f1;
    history_f1_l2(mc) = l2_result.f1;
    history_wall_time_robust(mc) = robust_result.wall_time;
    history_wall_time_l2(mc) = l2_result.wall_time;
    history_core_time_robust(mc) = robust_result.core_time;
    history_core_time_l2(mc) = l2_result.core_time;
    history_prior_irls_robust(mc) = robust_result.mean_prior_irls;
    history_prior_irls_l2(mc) = l2_result.mean_prior_irls;
    history_posterior_irls_robust(mc) = ...
        robust_result.mean_posterior_irls;
    history_posterior_irls_l2(mc) = l2_result.mean_posterior_irls;
    huber_thresholds(mc) = huber_threshold_est;
    snapshot_gt(:, :, mc) = X_true(:, :, snapshot_idx);
    snapshot_res_l2(:, :, mc) = l2_result.snapshot_residual;
    snapshot_res_rob(:, :, mc) = robust_result.snapshot_residual;
    snapshot_nre_l2(mc) = l2_result.snapshot_nre;
    snapshot_nre_rob(mc) = robust_result.snapshot_nre;

    if mod(mc, 2) == 0 || mc == 1, fprintf('Completed Trial %d/%d\n', mc, n_monte_carlo); end
end

%% 3. Statistical Analysis & Visualization
% -------------------------------------------------------------------------
mean_err_rob = mean(history_err_robust, 2); std_err_rob  = std(history_err_robust, 0, 2);
mean_err_l2  = mean(history_err_l2, 2);     std_err_l2   = std(history_err_l2, 0, 2);
history_wall_fps_robust = T ./ history_wall_time_robust;
history_wall_fps_l2 = T ./ history_wall_time_l2;
history_core_fps_robust = T ./ history_core_time_robust;
history_core_fps_l2 = T ./ history_core_time_l2;
wall_time_ratio = history_wall_time_robust ./ history_wall_time_l2;
core_time_ratio = history_core_time_robust ./ history_core_time_l2;

figure('Name', 'Exp S2: Spatter Robustness', 'Color', 'w', 'Position', [100, 100, 1200, 600]);

% Subplot 1: Convergence Curves
subplot(1, 2, 1); x_axis = (1:T)'; hold on;

% Color palette for consistency
color_rsi = [0.8500 0.3250 0.0980];
color_l2  = [0.0000 0.4470 0.7410];

% Plot Baseline L2
fill([x_axis; flipud(x_axis)], max(1e-10, [mean_err_l2 + std_err_l2; flipud(mean_err_l2 - std_err_l2)]), ...
    color_l2, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
semilogy(x_axis, mean_err_l2, '-', 'Color', color_l2, ...
    'LineWidth', 1.5, 'DisplayName', 'Squared-Loss Control');

% Plot RSI-OLSTEC
fill([x_axis; flipud(x_axis)], max(1e-10, [mean_err_rob + std_err_rob; flipud(mean_err_rob - std_err_rob)]), ...
    color_rsi, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
semilogy(x_axis, mean_err_rob, '-', 'Color', color_rsi, ...
    'LineWidth', 1.5, 'DisplayName', 'Huber-IRLS Variant');

grid on; set(gca, 'YScale', 'log');
xlabel('Time Index (Frame)'); ylabel('Mean NRE (log scale)');
title('(a) Convergence under Spatter Noise'); legend('Location', 'northeast');
all_positive_errs = [mean_err_l2(mean_err_l2 > 0); mean_err_rob(mean_err_rob > 0)];
if ~isempty(all_positive_errs)
    y_max = max(all_positive_errs) * 2.0;
    y_min = max(min(all_positive_errs) * 0.5, 1e-4);

    ylim([y_min, y_max]);
end
xlim([1, T]);

% Subplot 2: Steady-state Error Distribution
subplot(1, 2, 2);
steady_rob = mean(history_err_robust(max(1,T-99):T, :), 1)';
steady_l2  = mean(history_err_l2(max(1,T-99):T, :), 1)';
combined_steady = mean([steady_l2, steady_rob], 2);
median_combined_steady = median(combined_steady);
[~, representative_trial] = min(abs(combined_steady - median_combined_steady));
boxplot([steady_l2, steady_rob], ...
    'Labels', {'Squared-Loss Control', 'Huber-IRLS Variant'});
ylabel('Steady-State NRE');
title('(b) Error Distribution Stability'); grid on;

if export_results
    savefig(gcf, fullfile(result_dir, 'S2_1.fig'));
    set(gcf, 'Units', 'Inches');
    pos = get(gcf, 'Position');
    set(gcf, 'PaperPositionMode', 'auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
    print(gcf, fullfile(result_dir, 'S2_1.pdf'), '-dpdf', '-vector');
    print(gcf, fullfile(result_dir, 'S2_1.eps'), '-depsc', '-vector');
end

% --- Subplot 3: Spatial Residual Snapshot ---
figure('Name', 'Spatial Residual', 'Color', 'w', 'Position', [150, 150, 1000, 350]);

subplot(1, 3, 1);
imagesc(snapshot_gt(:, :, representative_trial)); axis off image; colorbar;
title(sprintf('Ground Truth [Trial %d]', representative_trial));

% L2 Residual
subplot(1, 3, 2);
Res_l2 = snapshot_res_l2(:, :, representative_trial);
Res_rob_for_scale = snapshot_res_rob(:, :, representative_trial);
imagesc(Res_l2); axis off image; colorbar;
pooled_residuals = sort([Res_l2(:); Res_rob_for_scale(:)]);
percentile_index = min(numel(pooled_residuals), max(1, ...
    round((residual_color_percentile / 100) * numel(pooled_residuals))));
c_max = pooled_residuals(percentile_index);
if c_max <= 0, c_max = max(pooled_residuals); end
if c_max <= 0, c_max = 1; end
residual_color_limits = [0, c_max];
local_set_clim(residual_color_limits);
title(sprintf('L2 Error [Median Combined Trial] (NRE: %.3f)', snapshot_nre_l2(representative_trial)));

% Values above the pooled percentile limit are color-saturated in both maps.
subplot(1, 3, 3);
Res_rob = Res_rob_for_scale;
imagesc(Res_rob); axis off image; colorbar;
local_set_clim(residual_color_limits);
title(sprintf('Huber Error [Median Combined Trial] (NRE: %.3f)', snapshot_nre_rob(representative_trial)));

if export_results
    savefig(gcf, fullfile(result_dir, 'S2_2.fig'));
    set(gcf, 'Units', 'Inches');
    pos = get(gcf, 'Position');
    set(gcf, 'PaperPositionMode', 'auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
    print(gcf, fullfile(result_dir, 'S2_2.pdf'), '-dpdf', '-vector');
    print(gcf, fullfile(result_dir, 'S2_2.eps'), '-depsc', '-vector');
end

%% 4. Quantitative Results
steady_robust_mean = mean(steady_rob);
steady_robust_std = std(steady_rob);
steady_l2_mean = mean(steady_l2);
steady_l2_std = std(steady_l2);
f1_robust_mean = mean(history_f1);
f1_robust_std = std(history_f1);
f1_l2_mean = mean(history_f1_l2);
f1_l2_std = std(history_f1_l2);

algorithm = {'Squared-loss control'; 'Huber-IRLS variant'};
trials = repmat(n_monte_carlo, 2, 1);
wall_time_median_seconds = [median(history_wall_time_l2); ...
    median(history_wall_time_robust)];
wall_time_q1_seconds = [linear_quantile(history_wall_time_l2, 0.25); ...
    linear_quantile(history_wall_time_robust, 0.25)];
wall_time_q3_seconds = [linear_quantile(history_wall_time_l2, 0.75); ...
    linear_quantile(history_wall_time_robust, 0.75)];
core_time_median_seconds = [median(history_core_time_l2); ...
    median(history_core_time_robust)];
core_time_q1_seconds = [linear_quantile(history_core_time_l2, 0.25); ...
    linear_quantile(history_core_time_robust, 0.25)];
core_time_q3_seconds = [linear_quantile(history_core_time_l2, 0.75); ...
    linear_quantile(history_core_time_robust, 0.75)];
wall_fps_median = [median(history_wall_fps_l2); ...
    median(history_wall_fps_robust)];
wall_fps_q1 = [linear_quantile(history_wall_fps_l2, 0.25); ...
    linear_quantile(history_wall_fps_robust, 0.25)];
wall_fps_q3 = [linear_quantile(history_wall_fps_l2, 0.75); ...
    linear_quantile(history_wall_fps_robust, 0.75)];
core_fps_median = [median(history_core_fps_l2); ...
    median(history_core_fps_robust)];
core_fps_q1 = [linear_quantile(history_core_fps_l2, 0.25); ...
    linear_quantile(history_core_fps_robust, 0.25)];
core_fps_q3 = [linear_quantile(history_core_fps_l2, 0.75); ...
    linear_quantile(history_core_fps_robust, 0.75)];
mean_prior_irls_iterations = [mean(history_prior_irls_l2); ...
    mean(history_prior_irls_robust)];
mean_posterior_irls_iterations = [mean(history_posterior_irls_l2); ...
    mean(history_posterior_irls_robust)];

runtime_summary_table = table(algorithm, trials, ...
    wall_time_median_seconds, wall_time_q1_seconds, ...
    wall_time_q3_seconds, core_time_median_seconds, ...
    core_time_q1_seconds, core_time_q3_seconds, wall_fps_median, ...
    wall_fps_q1, wall_fps_q3, core_fps_median, core_fps_q1, ...
    core_fps_q3, mean_prior_irls_iterations, ...
    mean_posterior_irls_iterations, ...
    'VariableNames', {'Algorithm', 'Trials', 'WallTimeMedianSeconds', ...
    'WallTimeQ1Seconds', 'WallTimeQ3Seconds', ...
    'CoreTimeMedianSeconds', 'CoreTimeQ1Seconds', ...
    'CoreTimeQ3Seconds', 'WallFPSMedian', 'WallFPSQ1', 'WallFPSQ3', ...
    'CoreFPSMedian', 'CoreFPSQ1', 'CoreFPSQ3', ...
    'MeanPriorIRLSIterations', 'MeanPosteriorIRLSIterations'});

[wall_ratio_estimate, wall_ratio_ci_lower, wall_ratio_ci_upper] = ...
    paired_geometric_mean_ratio(history_wall_time_robust, ...
    history_wall_time_l2, bootstrap_resamples, bootstrap_ci_level, ...
    bootstrap_seed);
[core_ratio_estimate, core_ratio_ci_lower, core_ratio_ci_upper] = ...
    paired_geometric_mean_ratio(history_core_time_robust, ...
    history_core_time_l2, bootstrap_resamples, bootstrap_ci_level, ...
    bootstrap_seed + 1);

comparison = {'Huber-IRLS / squared-loss control'};
valid_pairs = n_monte_carlo;
ratio_ci_level = bootstrap_ci_level;
ratio_bootstrap_resamples = bootstrap_resamples;
wall_ratio_bootstrap_seed = bootstrap_seed;
core_ratio_bootstrap_seed = bootstrap_seed + 1;
ratio_ci_method = {'Percentile bootstrap of paired geometric mean ratio'};
paired_overhead_table = table(comparison, valid_pairs, ...
    wall_ratio_estimate, wall_ratio_ci_lower, wall_ratio_ci_upper, ...
    core_ratio_estimate, core_ratio_ci_lower, core_ratio_ci_upper, ...
    ratio_ci_level, ratio_bootstrap_resamples, ...
    wall_ratio_bootstrap_seed, core_ratio_bootstrap_seed, ...
    ratio_ci_method, ...
    'VariableNames', {'Comparison', 'ValidPairs', ...
    'WallTimeGeometricMeanRatio', 'WallTimeRatioCILower', ...
    'WallTimeRatioCIUpper', 'CoreTimeGeometricMeanRatio', ...
    'CoreTimeRatioCILower', 'CoreTimeRatioCIUpper', 'CILevel', ...
    'BootstrapResamples', 'WallTimeBootstrapSeed', ...
    'CoreTimeBootstrapSeed', 'CIMethod'});

data_seed = (1:n_monte_carlo)';
trial = data_seed;
per_trial_table = table(trial, data_seed, execution_order, ...
    huber_thresholds, steady_l2, steady_rob, history_f1_l2, ...
    history_f1, history_wall_time_l2, history_wall_time_robust, ...
    history_core_time_l2, history_core_time_robust, ...
    history_wall_fps_l2, history_wall_fps_robust, ...
    history_core_fps_l2, history_core_fps_robust, ...
    wall_time_ratio, core_time_ratio, history_prior_irls_l2, ...
    history_prior_irls_robust, history_posterior_irls_l2, ...
    history_posterior_irls_robust, ...
    'VariableNames', {'Trial', 'DataSeed', 'ExecutionOrder', ...
    'HuberDelta', 'L2SteadyStateNRE', 'HuberSteadyStateNRE', ...
    'L2F1Score', 'HuberF1Score', 'L2WallTimeSeconds', ...
    'HuberWallTimeSeconds', 'L2CoreTimeSeconds', ...
    'HuberCoreTimeSeconds', 'L2WallFPS', 'HuberWallFPS', ...
    'L2CoreFPS', 'HuberCoreFPS', 'WallTimeRatio', ...
    'CoreTimeRatio', 'L2MeanPriorIRLSIterations', ...
    'HuberMeanPriorIRLSIterations', ...
    'L2MeanPosteriorIRLSIterations', ...
    'HuberMeanPosteriorIRLSIterations'});

fprintf('\n=================================================================================\n');
fprintf('   EXP S2 SUMMARY: Squared Loss vs. Huber-IRLS (Spatter Noise)\n');
fprintf('=================================================================================\n');
fprintf('%-28s | %-20s | %-20s\n', ...
    'Metric', 'Squared-Loss Control', 'Huber-IRLS Variant');
fprintf('---------------------------------------------------------------------------------\n');
fprintf('%-28s | %.4f +/- %.4f  | %.4f +/- %.4f\n', ...
    'Steady-State NRE', steady_l2_mean, steady_l2_std, ...
    steady_robust_mean, steady_robust_std);
fprintf('%-28s | %.4f +/- %.4f  | %.4f +/- %.4f\n', ...
    'Outlier Det. F1-Score', f1_l2_mean, f1_l2_std, ...
    f1_robust_mean, f1_robust_std);
fprintf('%-28s | %-20d | %-20d\n', 'Snapshot Trial', representative_trial, representative_trial);
fprintf('=================================================================================\n');
fprintf('Median wall time (s): L2 %.3f, Huber %.3f\n', ...
    wall_time_median_seconds(1), wall_time_median_seconds(2));
fprintf('Median core FPS:      L2 %.3f, Huber %.3f\n', ...
    core_fps_median(1), core_fps_median(2));
fprintf(['Paired Huber/L2 wall-time ratio: %.3f ' ...
    '(%.0f%% CI %.3f--%.3f)\n'], wall_ratio_estimate, ...
    100 * bootstrap_ci_level, wall_ratio_ci_lower, wall_ratio_ci_upper);
fprintf(['Paired Huber/L2 core-time ratio: %.3f ' ...
    '(%.0f%% CI %.3f--%.3f)\n'], core_ratio_estimate, ...
    100 * bootstrap_ci_level, core_ratio_ci_lower, core_ratio_ci_upper);

if export_results
    writetable(per_trial_table, ...
        fullfile(result_dir, 'S2_per_trial_metrics.csv'));
    writetable(runtime_summary_table, ...
        fullfile(result_dir, 'S2_runtime_summary.csv'));
    writetable(paired_overhead_table, ...
        fullfile(result_dir, 'S2_paired_overhead_ci.csv'));

    matlab_version = version;
    computer_architecture = computer;
    save(fullfile(result_dir, 'S2_stats.mat'), ...
        'n_monte_carlo', 'dims', 'true_rank', 'SNR_dB', 'sparse_ratio', ...
        'spatter_base_mag', 'observation_ratio', 'drift_rate', ...
        'history_err_robust', 'history_err_l2', 'history_f1', 'history_f1_l2', ...
        'history_wall_time_robust', 'history_wall_time_l2', ...
        'history_core_time_robust', 'history_core_time_l2', ...
        'history_wall_fps_robust', 'history_wall_fps_l2', ...
        'history_core_fps_robust', 'history_core_fps_l2', ...
        'history_prior_irls_robust', 'history_prior_irls_l2', ...
        'history_posterior_irls_robust', 'history_posterior_irls_l2', ...
        'wall_time_ratio', 'core_time_ratio', 'huber_thresholds', ...
        'execution_order', 'steady_rob', 'steady_l2', ...
        'combined_steady', 'median_combined_steady', ...
        'snapshot_idx', 'representative_trial', 'snapshot_gt', ...
        'snapshot_res_rob', 'snapshot_res_l2', 'snapshot_nre_rob', ...
        'snapshot_nre_l2', 'residual_color_percentile', ...
        'residual_color_limits', 'timing_warmup_runs_per_variant', ...
        'bootstrap_ci_level', 'bootstrap_resamples', 'bootstrap_seed', ...
        'runtime_summary_table', 'paired_overhead_table', ...
        'per_trial_table', 'matlab_version', 'computer_architecture');
end

function result = run_s2_variant(observed_tensor, observation_mask, ...
    tensor_dims, rank_r, initialization, options, auxiliary_signal, ...
    evaluation, method_label)

    timer = tic;
    [~, infos, sub_infos] = rsi_olstec(observed_tensor, ...
        observation_mask, [], tensor_dims, rank_r, initialization, ...
        options, auxiliary_signal);
    wall_time = toc(timer);

    core_time = infos.time(end);
    if ~(isfinite(wall_time) && wall_time > 0 && ...
            isfinite(core_time) && core_time > 0)
        error('Exp_S2:InvalidRuntime', ...
            '%s returned a nonpositive or nonfinite runtime.', method_label);
    end
    if ~isfield(sub_infos, 'L') || isempty(sub_infos.L)
        error('Exp_S2:MissingReconstruction', ...
            '%s did not return the required online reconstruction.', ...
            method_label);
    end

    online_reconstruction = sub_infos.L;
    prior_iterations = sub_infos.prior_irls_iterations;
    posterior_iterations = sub_infos.posterior_irls_iterations;
    clear sub_infos;

    total_frames = tensor_dims(3);
    nre = validate_complete_nre( ...
        compute_true_nre_tensor(evaluation.reference_tensor, ...
        online_reconstruction), total_frames, method_label, ...
        evaluation.trial)';

    residuals = abs(evaluation.observed_matrix - ...
        online_reconstruction) .* evaluation.observation_mask;
    detected_outliers = ...
        (residuals > evaluation.outlier_threshold) & ...
        evaluation.observation_mask;
    frames = evaluation.evaluation_frames;
    true_outliers = evaluation.true_outliers;

    true_positive = sum(sum(detected_outliers(:, frames) & ...
        true_outliers(:, frames)));
    false_positive = sum(sum(detected_outliers(:, frames) & ...
        ~true_outliers(:, frames)));
    false_negative = sum(sum(~detected_outliers(:, frames) & ...
        true_outliers(:, frames)));
    precision = true_positive / (true_positive + false_positive + 1e-10);
    recall = true_positive / (true_positive + false_negative + 1e-10);
    f1_score = 2 * precision * recall / (precision + recall + 1e-10);

    snapshot_index = evaluation.snapshot_index;
    rows = tensor_dims(1);
    cols = tensor_dims(2);
    reconstructed_snapshot = reshape( ...
        online_reconstruction(:, snapshot_index), rows, cols);
    snapshot_residual = abs( ...
        evaluation.reference_tensor(:, :, snapshot_index) - ...
        reconstructed_snapshot);

    result = struct( ...
        'nre', nre, ...
        'f1', f1_score, ...
        'snapshot_residual', snapshot_residual, ...
        'snapshot_nre', nre(snapshot_index), ...
        'wall_time', wall_time, ...
        'core_time', core_time, ...
        'mean_prior_irls', mean_finite(prior_iterations), ...
        'mean_posterior_irls', mean_finite(posterior_iterations));
end

function value = mean_finite(values)
    values = values(isfinite(values));
    if isempty(values)
        value = NaN;
    else
        value = mean(values);
    end
end

function [estimate, ci_lower, ci_upper] = paired_geometric_mean_ratio( ...
    numerator, denominator, num_resamples, ci_level, bootstrap_seed)

    numerator = numerator(:);
    denominator = denominator(:);
    valid = isfinite(numerator) & numerator > 0 & ...
        isfinite(denominator) & denominator > 0;
    if ~all(valid) || ~any(valid)
        error('Exp_S2:InvalidPairedRuntime', ...
            'Paired runtime analysis requires finite positive pairs.');
    end

    log_ratios = log(numerator(valid) ./ denominator(valid));
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

function local_set_clim(limits)
    if exist('clim', 'file') == 2 || exist('clim', 'builtin') == 5
        clim(limits);
    else
        set(gca, 'CLim', limits);
    end
end
