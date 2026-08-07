%% Experiment S1
% =========================================================================
% Prerequisites: Run `run_me_first.m` once to add all dependencies to path.
%
% Objective:
%   Evaluate post-mutation recovery across subspace mutation strengths
%   using paired Monte Carlo simulations.
% =========================================================================
clear; clc; close all;

%% 1. Configuration
% -------------------------------------------------------------------------
n_monte_carlo = 50;
I = 100; J = 100; T = 1500;
dims = [I, J, T];
true_rank = 5;
SNR_dB = 25;
sparse_ratio = 0.05;
spatter_base_mag = 0.50;
observation_ratio = 0.50;
aux_noise_sigma = 0.20;
aux_step_gain = 10.0;       % Side-information event gain
export_results = true;
result_dir = fullfile(fileparts(mfilename('fullpath')), 'result', 'S1');
if export_results && ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

scales_to_test = [0.1, 0.5, 1.0];
num_scales = length(scales_to_test);

% Alignment Windows (For Statistical Step Response)
window_pre = 100;
window_post = 300;
eval_window = window_pre + window_post + 1;

% Storage Matrices (3D)
Aligned_Error_Std  = zeros(eval_window, n_monte_carlo, num_scales);
Aligned_Error_RobustFixed = zeros(eval_window, n_monte_carlo, num_scales);
Aligned_Error_RSI  = zeros(eval_window, n_monte_carlo, num_scales);
Aligned_Error_Fast = zeros(eval_window, n_monte_carlo, num_scales);
Aligned_Aux        = zeros(eval_window, n_monte_carlo, num_scales);
mutation_frames    = zeros(n_monte_carlo, 1);
sensor_delays      = zeros(n_monte_carlo, 1);

% Hyperparameters
lambda_long = 0.80;
mu = 0.01;

fprintf('Starting multi-scale Monte Carlo simulation (%d trials)...\n', n_monte_carlo);

%% 2. Monte Carlo Loop
for mc = 1:n_monte_carlo
    rng(mc, 'twister');

    % --- 1. Randomization of Events ---
    % The mutation frame is the first frame generated from the mutated bases.
    mutation_frame = randi([401, 801]);
    sensor_delay = randi([2, 5]);
    aux_start_frame = mutation_frame + sensor_delay;
    mutation_frames(mc) = mutation_frame;
    sensor_delays(mc) = sensor_delay;

    % --- 2. Base Subspace Generation ---
    A1 = randn(I, true_rank); B1 = randn(J, true_rank);
    [Q_A, R_A] = qr(A1, 0); A1 = Q_A * diag(sign(diag(R_A) + 1e-10));
    [Q_B, R_B] = qr(B1, 0); B1 = Q_B * diag(sign(diag(R_B) + 1e-10));

    % Pre-generate fixed perturbation directions for controlled variables
    perturbation_dir_A = randn(I, true_rank);
    perturbation_dir_B = randn(J, true_rank);

    % =====================================================================
    % --- 3. Inner Loop: Iterate through different Mutation Scales ---
    % =====================================================================
    for s_idx = 1:num_scales
        perturbation_scale = scales_to_test(s_idx);

        % Couple auxiliary sensor signal with mutation strength in a controlled
        % side-information setting. The coupling strength is exported so the
        % controlled assumption is recorded with the experiment outputs.
        rng(mc * 10000 + 1, 'twister');
        clean_aux = 10.0 * ones(T, 1);
        for t = aux_start_frame:T
            if t == aux_start_frame
                clean_aux(t) = 10.0 + aux_step_gain * perturbation_scale;
            else
                clean_aux(t) = 10.0 + 0.8 * (clean_aux(t-1) - 10.0);
            end
        end
        Aux_Signal = clean_aux + aux_noise_sigma * randn(T, 1);

        start_idx = mutation_frame - window_pre;
        end_idx   = mutation_frame + window_post;
        Aligned_Aux(:, mc, s_idx) = Aux_Signal(start_idx:end_idx);

        % Apply scaled tensor mutations
        [Q_A2, R_A2] = qr(A1 + perturbation_scale * perturbation_dir_A, 0);
        A2 = Q_A2 * diag(sign(diag(R_A2) + 1e-10));
        [Q_B2, R_B2] = qr(B1 + perturbation_scale * perturbation_dir_B, 0);
        B2 = Q_B2 * diag(sign(diag(R_B2) + 1e-10));

        % Generate continuous sine dynamics for temporal factor. Common random
        % numbers are reused across mutation scales within each Monte Carlo
        % trial so kappa is the only controlled variable changing this block.
        rng(mc * 10000 + 2, 'twister');
        C_true = zeros(T, true_rank); t_idx = (1:T)';
        for r = 1:true_rank
            C_true(:, r) = 10.0 + 2.0 * sin(2 * pi * t_idx / (100 + r*10)) + 0.1 * randn(T, 1);
        end

        % Generate dynamic ground-truth tensor
        X_true = zeros(I, J, T);

        for t = 1:T
            if t < mutation_frame
                slice = A1 * diag(C_true(t, :)) * B1';
            else
                slice = A2 * diag(C_true(t, :)) * B2';
            end
            X_true(:, :, t) = slice;
        end

        sig_pow = norm(X_true(:))^2 / numel(X_true);
        noise_sigma = sqrt(sig_pow / 10^(SNR_dB/10));
        rng(mc * 10000 + 3, 'twister');
        Gaussian_Noise = noise_sigma * randn(I, J, T);

        rng(mc * 10000 + 4, 'twister');
        S_mask = rand(I, J, T) < sparse_ratio;
        Sparse_Noise = zeros(I, J, T);
        Sparse_Noise(S_mask) = spatter_base_mag * (1 + abs(randn(sum(S_mask(:)), 1)));

        Y_full = X_true + Gaussian_Noise + Sparse_Noise;

        rng(mc * 10000 + 5, 'twister');
        Omega = rand(I, J, T) < observation_ratio;
        Y_observed = Y_full .* Omega;

        % Initialization (shared starting point)
        rng(mc * 10000 + 6, 'twister');
        [X_init.A, ~] = qr(randn(I, true_rank), 0);
        [X_init.B, ~] = qr(randn(J, true_rank), 0);
        X_init.C = randn(T, true_rank);

        burn_in_frames = 30;
        diff_pixels = cell(burn_in_frames - 1, 1);
        for t_idx = 2:burn_in_frames
            common_mask = Omega(:,:,t_idx) & Omega(:,:,t_idx-1);
            diff_frame = Y_observed(:,:,t_idx) - Y_observed(:,:,t_idx-1);
            diff_pixels{t_idx-1} = diff_frame(common_mask);
        end
        diff_pixels = vertcat(diff_pixels{:});
        if ~isempty(diff_pixels)
            mad_val = median(abs(diff_pixels - median(diff_pixels)));
            est_sigma = (1.4826 * mad_val) / sqrt(2);
            huber_delta_est = max(0.01, 3 * est_sigma);
        else
            huber_delta_est = 0.05;
        end

        % Algorithm configurations
        opts_std = struct('lambda', lambda_long, 'mu', mu, 'verbose', 0, ...
                          'maxepochs', 1, 'tolcost', 1e-8, ...
                          'store_matrix', true, 'store_subinfo', true);

        burn_in_aux = min(30, length(Aux_Signal));
        diff_aux = diff(Aux_Signal(1:burn_in_aux));
        mad_aux = median(abs(diff_aux - median(diff_aux)));
        est_aux_sigma = (1.4826 * mad_aux) / sqrt(2);

        opts_rsi = struct('lambda_max', lambda_long, 'lambda_min', 0.10, ...
                          'huber_delta', huber_delta_est, 'mu', mu, 'verbose', 0, ...
                          'maxepochs', 1, 'tolcost', 1e-8, ...
                          'store_matrix', true, 'store_subinfo', true, ...
                          'min_grad_threshold', max(0.05, 3 * sqrt(2) * est_aux_sigma));
        opts_robust_fixed = opts_rsi;
        opts_robust_fixed.lambda_min = lambda_long;
        opts_robust_fixed.lambda_max = lambda_long;
        opts_robust_fixed.min_grad_threshold = Inf;

        % Execute algorithms
        [~, ~, info_std] = olstec(Y_observed, Omega, [], dims, true_rank, X_init, opts_std);
        [~, ~, info_robust_fixed] = rsi_olstec(Y_observed, Omega, [], dims, true_rank, X_init, opts_robust_fixed, Aux_Signal);
        [~, ~, info_rsi] = rsi_olstec(Y_observed, Omega, [], dims, true_rank, X_init, opts_rsi, Aux_Signal);

        % Execute Fast OLSTEC (lambda = 0.10)
        opts_fast = opts_std;
        opts_fast.lambda = 0.10;
        [~, ~, info_fast] = olstec(Y_observed, Omega, [], dims, true_rank, X_init, opts_fast);

        % Calculate tracking errors with the shared external evaluator.
        temp_err_std = validate_complete_nre( ...
            compute_true_nre_tensor(X_true, info_std.L), T, ...
            sprintf('Standard OLSTEC (kappa=%.1f)', perturbation_scale), mc)';
        temp_err_robust_fixed = validate_complete_nre( ...
            compute_true_nre_tensor(X_true, info_robust_fixed.L), T, ...
            sprintf('Huber fixed-lambda (kappa=%.1f)', perturbation_scale), mc)';
        temp_err_rsi = validate_complete_nre( ...
            compute_true_nre_tensor(X_true, info_rsi.L), T, ...
            sprintf('RSI-OLSTEC (kappa=%.1f)', perturbation_scale), mc)';
        temp_err_fast = validate_complete_nre( ...
            compute_true_nre_tensor(X_true, info_fast.L), T, ...
            sprintf('Fast OLSTEC (kappa=%.1f)', perturbation_scale), mc)';

        Aligned_Error_Std(:, mc, s_idx) = temp_err_std(start_idx:end_idx);
        Aligned_Error_RobustFixed(:, mc, s_idx) = temp_err_robust_fixed(start_idx:end_idx);
        Aligned_Error_RSI(:, mc, s_idx) = temp_err_rsi(start_idx:end_idx);
        Aligned_Error_Fast(:, mc, s_idx) = temp_err_fast(start_idx:end_idx);
    end

    if mod(mc, 5) == 0 || mc == 1
        fprintf('Completed Trial %d/%d\n', mc, n_monte_carlo);
    end
end

%% 3. Statistical Processing
% -------------------------------------------------------------------------
mean_std = squeeze(mean(Aligned_Error_Std, 2));
std_std  = squeeze(std(Aligned_Error_Std, 0, 2));
mean_robust_fixed = squeeze(mean(Aligned_Error_RobustFixed, 2));
std_robust_fixed  = squeeze(std(Aligned_Error_RobustFixed, 0, 2));
mean_rsi = squeeze(mean(Aligned_Error_RSI, 2));
std_rsi  = squeeze(std(Aligned_Error_RSI, 0, 2));
mean_fast = squeeze(mean(Aligned_Error_Fast, 2));
std_fast  = squeeze(std(Aligned_Error_Fast, 0, 2));
mean_aux = squeeze(mean(Aligned_Aux, 2));
x_axis_relative = (-window_pre:window_post)';

%% 4. Visualization
% -------------------------------------------------------------------------
figure('Color', 'w', 'Position', [100, 100, 900, 650]);

color_rsi   = [0.8500 0.3250 0.0980];
color_std   = [0.0000 0.4470 0.7410];
color_fast  = [0.4660 0.6740 0.1880];
color_robust_fixed = [0.4940 0.1840 0.5560];
line_styles = {'-.', '--', '-'};    % Weak (dash-dot), Medium (dashed), Strong (solid)
line_widths = [1.2, 1.5, 2.0];      % Increasing line width indicates higher strength

% --- Subplot 1: Tracking Performance (Multi-Scale) ---
subplot(3, 1, [1 2]); hold on;

% Pre-allocate handle arrays to control legend order
h_std = gobjects(num_scales, 1);
h_robust_fixed = gobjects(num_scales, 1);
h_rsi = gobjects(num_scales, 1);
h_fast = gobjects(num_scales, 1);
lbl_std = cell(num_scales, 1);
lbl_robust_fixed = cell(num_scales, 1);
lbl_rsi = cell(num_scales, 1);
lbl_fast = cell(num_scales, 1);

for s_idx = 1:num_scales
    % Plot error standard deviation bands
    fill([x_axis_relative; flipud(x_axis_relative)], [mean_std(:, s_idx) + std_std(:, s_idx); flipud(max(0, mean_std(:, s_idx) - std_std(:, s_idx)))], ...
         color_std, 'EdgeColor', 'none', 'FaceAlpha', 0.08, 'HandleVisibility', 'off');
    fill([x_axis_relative; flipud(x_axis_relative)], [mean_robust_fixed(:, s_idx) + std_robust_fixed(:, s_idx); flipud(max(0, mean_robust_fixed(:, s_idx) - std_robust_fixed(:, s_idx)))], ...
         color_robust_fixed, 'EdgeColor', 'none', 'FaceAlpha', 0.08, 'HandleVisibility', 'off');
    fill([x_axis_relative; flipud(x_axis_relative)], [mean_rsi(:, s_idx) + std_rsi(:, s_idx); flipud(max(0, mean_rsi(:, s_idx) - std_rsi(:, s_idx)))], ...
         color_rsi, 'EdgeColor', 'none', 'FaceAlpha', 0.08, 'HandleVisibility', 'off');
    fill([x_axis_relative; flipud(x_axis_relative)], [mean_fast(:, s_idx) + std_fast(:, s_idx); flipud(max(0, mean_fast(:, s_idx) - std_fast(:, s_idx)))], ...
         color_fast, 'EdgeColor', 'none', 'FaceAlpha', 0.08, 'HandleVisibility', 'off');

    % Plot mean error curves
    h_std(s_idx) = plot(x_axis_relative, mean_std(:, s_idx), 'LineStyle', line_styles{s_idx}, 'Color', color_std, 'LineWidth', line_widths(s_idx));
    lbl_std{s_idx} = sprintf('OLSTEC ($\\kappa=%.1f$)', scales_to_test(s_idx));

    h_robust_fixed(s_idx) = plot(x_axis_relative, mean_robust_fixed(:, s_idx), 'LineStyle', line_styles{s_idx}, 'Color', color_robust_fixed, 'LineWidth', line_widths(s_idx));
    lbl_robust_fixed{s_idx} = sprintf('Huber no-aux fixed-$\\lambda$ ($\\kappa=%.1f$)', scales_to_test(s_idx));

    h_rsi(s_idx) = plot(x_axis_relative, mean_rsi(:, s_idx), 'LineStyle', line_styles{s_idx}, 'Color', color_rsi, 'LineWidth', line_widths(s_idx));
    lbl_rsi{s_idx} = sprintf('RSI-OLSTEC ($\\kappa=%.1f$)', scales_to_test(s_idx));

    h_fast(s_idx) = plot(x_axis_relative, mean_fast(:, s_idx), 'LineStyle', line_styles{s_idx}, 'Color', color_fast, 'LineWidth', line_widths(s_idx));
    lbl_fast{s_idx} = sprintf('Fast OLSTEC ($\\kappa=%.1f$)', scales_to_test(s_idx));
end

grid on;
ylabel('Mean NRE', 'FontSize', 13, 'Interpreter', 'latex');
title('(a) Tracking Response to Controlled Subspace Changes', 'FontSize', 14, 'Interpreter', 'latex');
set(gca, 'XTickLabel', [], 'FontName', 'Times New Roman', 'FontSize', 12, 'TickLabelInterpreter', 'latex');
xlim([-window_pre, window_post]);
ylim([0, max([mean_std(:); mean_robust_fixed(:); mean_rsi(:); mean_fast(:)]) * 1.1]);
yy = ylim;
line([0 0], yy, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1.2, 'HandleVisibility', 'off');

% Reorder handles to match MATLAB's column-wise legend population
h_leg = [h_std(1), h_std(2), h_std(3), h_fast(1), h_fast(2), h_fast(3), h_robust_fixed(1), h_robust_fixed(2), h_robust_fixed(3), h_rsi(1), h_rsi(2), h_rsi(3)];
lbl_leg = {lbl_std{1}, lbl_std{2}, lbl_std{3}, lbl_fast{1}, lbl_fast{2}, lbl_fast{3}, lbl_robust_fixed{1}, lbl_robust_fixed{2}, lbl_robust_fixed{3}, lbl_rsi{1}, lbl_rsi{2}, lbl_rsi{3}};
lgd = legend(h_leg, lbl_leg, 'Location', 'northeast', 'NumColumns', 4, 'Interpreter', 'latex');
lgd.FontSize = 10; lgd.Box = 'off';

% --- Subplot 2: Auxiliary Signal (Multi-Scale) ---
subplot(3, 1, 3); hold on;
h_aux = gobjects(num_scales, 1);
lbl_aux = cell(num_scales, 1);

for s_idx = 1:num_scales
    h_aux(s_idx) = plot(x_axis_relative, mean_aux(:, s_idx), 'Color', 'k', ...
         'LineStyle', line_styles{s_idx}, 'LineWidth', line_widths(s_idx));
    lbl_aux{s_idx} = sprintf('Sensor $s_t$ ($\\kappa=%.1f$)', scales_to_test(s_idx));
end

grid on;
ylabel('Aux Signal ($s_t$)', 'Interpreter', 'latex', 'FontSize', 13);
xlabel('Frames Relative to the First Mutated Frame', 'Interpreter', 'latex', 'FontSize', 13);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'TickLabelInterpreter', 'latex');
xlim([-window_pre, window_post]);
max_aux_val = 10.0 + 10.0 * max(scales_to_test);
ylim([8, max_aux_val * 1.15]);

% Position legend to utilize blank space before mutation
lgd_aux = legend(h_aux, lbl_aux, 'Location', 'northwest', 'NumColumns', 3, 'Interpreter', 'latex');
lgd_aux.FontSize = 10; lgd_aux.Box = 'off';

%% 5. Quantitative Summary
fprintf('\n=================================================================================\n');
fprintf('   MULTI-SCALE QUANTITATIVE RESULTS (Statistical Mean over %d Trials)\n', n_monte_carlo);
fprintf('=================================================================================\n');
zero_idx = window_pre + 1;
pre_mut_range  = (zero_idx - 50):(zero_idx - 1);
post_mut_range = zero_idx:(zero_idx + 49);

for s_idx = 1:num_scales
    fprintf(' MUTATION SCALE: kappa = %.1f \n', scales_to_test(s_idx));
    fprintf('%-30s | %-18s | %-18s | %-18s | %-18s\n', 'Metric', 'Standard OLSTEC', 'Fast OLSTEC (0.1)', 'Huber no-aux fixed lambda', 'RSI-OLSTEC (step aux)');
    fprintf('----------------------------------------------------------------------------------------------------------------------\n');
    fprintf('%-30s | %.4f             | %.4f             | %.4f             | %.4f\n', 'Pre-Mutation NRE (Steady)', mean(mean_std(pre_mut_range, s_idx)), mean(mean_fast(pre_mut_range, s_idx)), mean(mean_robust_fixed(pre_mut_range, s_idx)), mean(mean_rsi(pre_mut_range, s_idx)));
    fprintf('%-30s | %.4f             | %.4f             | %.4f             | %.4f\n', 'Post-Mutation Peak NRE', max(mean_std(post_mut_range, s_idx)), max(mean_fast(post_mut_range, s_idx)), max(mean_robust_fixed(post_mut_range, s_idx)), max(mean_rsi(post_mut_range, s_idx)));
    fprintf('%-30s | %.4f             | %.4f             | %.4f             | %.4f\n', 'Recovery Integral (50 frames)', sum(mean_std(post_mut_range, s_idx)), sum(mean_fast(post_mut_range, s_idx)), sum(mean_robust_fixed(post_mut_range, s_idx)), sum(mean_rsi(post_mut_range, s_idx)));
    fprintf('----------------------------------------------------------------------------------------------------------------------\n\n');
end

fprintf('Auxiliary signal step = %.2f * kappa, noise sigma = %.2f; controlled side-information setting.\n', aux_step_gain, aux_noise_sigma);

if export_results
    save(fullfile(result_dir, 'S1_stats.mat'), ...
        'n_monte_carlo', 'dims', 'true_rank', 'SNR_dB', 'sparse_ratio', ...
        'spatter_base_mag', 'observation_ratio', 'scales_to_test', ...
        'window_pre', 'window_post', 'lambda_long', 'mu', ...
        'aux_noise_sigma', 'aux_step_gain', 'mutation_frames', 'sensor_delays', ...
        'Aligned_Error_Std', 'Aligned_Error_RobustFixed', ...
        'Aligned_Error_RSI', 'Aligned_Error_Fast', 'Aligned_Aux', ...
        'mean_std', 'std_std', 'mean_robust_fixed', 'std_robust_fixed', ...
        'mean_rsi', 'std_rsi', 'mean_fast', 'std_fast', 'mean_aux');
    savefig(gcf, fullfile(result_dir, 'S1.fig'));
    set(gcf, 'Units', 'Inches');
    pos = get(gcf, 'Position');
    set(gcf, 'PaperPositionMode', 'auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
    print(gcf, fullfile(result_dir, 'S1.pdf'), '-dpdf', '-vector');
    print(gcf, fullfile(result_dir, 'S1.eps'), '-depsc', '-vector');
end
