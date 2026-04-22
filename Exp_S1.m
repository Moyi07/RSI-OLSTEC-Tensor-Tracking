%% Experiment S1
% =========================================================================
% Description: 
%   This script performs a Monte Carlo simulation to statistically validate 
%   the "Rapid Recovery" capability of RSI-OLSTEC across varying 
%   subspace mutation strengths (delta).
% =========================================================================
clear; clc; close all;

%% 1. Configuration
% -------------------------------------------------------------------------
n_monte_carlo = 100; 
I = 100; J = 100; T = 1500;
dims = [I, J, T];
true_rank = 5;          
SNR_dB = 50;            
missing_rate = 0.50;    

scales_to_test = [0.1, 0.5, 1.0]; 
num_scales = length(scales_to_test);

% Alignment Windows (For Statistical Step Response)
window_pre = 100;
window_post = 300;
eval_window = window_pre + window_post + 1;

% Storage Matrices (3D)
Aligned_Error_Std = zeros(eval_window, n_monte_carlo, num_scales);
Aligned_Error_RSI = zeros(eval_window, n_monte_carlo, num_scales);
Aligned_Aux       = zeros(eval_window, n_monte_carlo, num_scales);

% Hyperparameters
lambda_long = 0.90;     
mu = 0.01;              

fprintf('Starting Rigorous Multi-Scale Monte Carlo Simulation (%d trials)...\n', n_monte_carlo);

%% 2. Monte Carlo Loop
for mc = 1:n_monte_carlo
    rng(mc, 'twister'); 
    
    % --- 1. Randomization of Events ---
    change_point = randi([400, 800]); 
    sensor_delay = randi([2, 5]); 
    aux_start_point = change_point + sensor_delay;

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
        
        % Couple auxiliary sensor signal with mutation strength
        aux_noise_sigma = 0.2; 
        clean_aux = 10.0 * ones(T, 1);
        for t = aux_start_point:T
            if t == aux_start_point
                clean_aux(t) = 10.0 + 10.0 * perturbation_scale; 
            else
                clean_aux(t) = 10.0 + 0.8 * (clean_aux(t-1) - 10.0); 
            end
        end
        Aux_Signal = clean_aux + aux_noise_sigma * randn(T, 1);
        
        start_idx = change_point - window_pre;
        end_idx   = change_point + window_post;
        Aligned_Aux(:, mc, s_idx) = Aux_Signal(start_idx:end_idx);
        
        % Apply scaled tensor mutations
        [Q_A2, R_A2] = qr(A1 + perturbation_scale * perturbation_dir_A, 0);
        A2 = Q_A2 * diag(sign(diag(R_A2) + 1e-10)); 
        [Q_B2, R_B2] = qr(B1 + perturbation_scale * perturbation_dir_B, 0); 
        B2 = Q_B2 * diag(sign(diag(R_B2) + 1e-10));
        
        % Generate dynamic ground-truth tensor
        X_true = zeros(I, J, T); 
        c_t_current = 10.0 * ones(true_rank, 1); 
        
        for t = 1:T
            c_t_current = 10.0 + 0.98 * (c_t_current - 10.0) + 0.1 * randn(true_rank, 1);
            if t <= change_point
                slice = A1 * diag(c_t_current) * B1';
            else
                slice = A2 * diag(c_t_current) * B2'; 
            end
            X_true(:, :, t) = slice;
        end
        
        sig_pow = norm(X_true(:))^2 / numel(X_true);
        noise_sigma = sqrt(sig_pow / 10^(SNR_dB/10));
        Y_full = X_true + noise_sigma * randn(I, J, T);
        
        Omega = rand(I, J, T) > missing_rate;           
        Y_observed = Y_full .* Omega;                   
        
        % Initialization (shared starting point)
        rng(mc * 100); 
        [X_init.A, ~] = qr(randn(I, true_rank), 0);
        [X_init.B, ~] = qr(randn(J, true_rank), 0);
        X_init.C = zeros(T, true_rank);
        
        % Algorithm configurations
        opts_std = struct('lambda', lambda_long, 'mu', mu, 'verbose', 0, ...
                          'maxepochs', 1, 'tolcost', 1e-8, ...
                          'store_matrix', true, 'store_subinfo', true);
        
        burn_in_frames = 100;
        diff_aux = diff(Aux_Signal(1:burn_in_frames));
        mad_aux = median(abs(diff_aux - median(diff_aux)));
        est_aux_sigma = (1.4826 * mad_aux) / sqrt(2);
        opts_rsi = struct('lambda_max', lambda_long, 'lambda_min', 0.10, ...
                          'huber_delta', Inf, 'mu', mu, 'verbose', 0, ...
                          'maxepochs', 1, 'tolcost', 1e-8, ...
                          'store_matrix', true, 'store_subinfo', true, ...
                          'min_grad_threshold', 3 * sqrt(2) * est_aux_sigma);
        
        % Execute algorithms
        [~, ~, info_std] = olstec(Y_observed, Omega, [], dims, true_rank, X_init, opts_std);
        [~, ~, info_rsi] = rsi_olstec(Y_observed, Omega, [], dims, true_rank, X_init, opts_rsi, Aux_Signal);
        
        % Calculate tracking errors
        temp_err_std = zeros(T, 1);
        temp_err_rsi = zeros(T, 1);
        for t = 1:T
            X_gt = X_true(:, :, t); 
            norm_gt = norm(X_gt, 'fro');
            temp_err_std(t) = norm(X_gt - reshape(info_std.L(:, t), [I, J]), 'fro') / (norm_gt + 1e-10);
            temp_err_rsi(t) = norm(X_gt - reshape(info_rsi.L(:, t), [I, J]), 'fro') / (norm_gt + 1e-10);
        end
        
        Aligned_Error_Std(:, mc, s_idx) = temp_err_std(start_idx:end_idx);
        Aligned_Error_RSI(:, mc, s_idx) = temp_err_rsi(start_idx:end_idx);
    end
    
    if mod(mc, 5) == 0 || mc == 1
        fprintf('Completed Trial %d/%d\n', mc, n_monte_carlo); 
    end
end

%% 3. Statistical Processing
% -------------------------------------------------------------------------
mean_std = squeeze(mean(Aligned_Error_Std, 2)); 
std_std  = squeeze(std(Aligned_Error_Std, 0, 2));
mean_rsi = squeeze(mean(Aligned_Error_RSI, 2)); 
std_rsi  = squeeze(std(Aligned_Error_RSI, 0, 2));
mean_aux = squeeze(mean(Aligned_Aux, 2));
x_axis_relative = (-window_pre:window_post)'; 

%% 4. Visualization
% -------------------------------------------------------------------------
figure('Color', 'w', 'Position', [100, 100, 900, 650]); 

color_rsi = [0.8500 0.3250 0.0980]; 
color_std = [0.0000 0.4470 0.7410]; 
line_styles = {'-.', '--', '-'};    % Weak (dash-dot), Medium (dashed), Strong (solid)
line_widths = [1.2, 1.5, 2.0];      % Increasing line width indicates higher strength

% --- Subplot 1: Tracking Performance (Multi-Scale) ---
subplot(3, 1, [1 2]); hold on;

% Pre-allocate handle arrays to control legend order
h_std = gobjects(num_scales, 1);
h_rsi = gobjects(num_scales, 1);
lbl_std = cell(num_scales, 1);
lbl_rsi = cell(num_scales, 1);

for s_idx = 1:num_scales
    % Plot error standard deviation bands
    fill([x_axis_relative; flipud(x_axis_relative)], [mean_std(:, s_idx) + std_std(:, s_idx); flipud(max(0, mean_std(:, s_idx) - std_std(:, s_idx)))], ...
         color_std, 'EdgeColor', 'none', 'FaceAlpha', 0.08, 'HandleVisibility', 'off');
    fill([x_axis_relative; flipud(x_axis_relative)], [mean_rsi(:, s_idx) + std_rsi(:, s_idx); flipud(max(0, mean_rsi(:, s_idx) - std_rsi(:, s_idx)))], ...
         color_rsi, 'EdgeColor', 'none', 'FaceAlpha', 0.08, 'HandleVisibility', 'off');
         
    % Plot mean error curves
    h_std(s_idx) = plot(x_axis_relative, mean_std(:, s_idx), 'LineStyle', line_styles{s_idx}, 'Color', color_std, 'LineWidth', line_widths(s_idx));
    lbl_std{s_idx} = sprintf('OLSTEC ($\\delta=%.1f$)', scales_to_test(s_idx));
    
    h_rsi(s_idx) = plot(x_axis_relative, mean_rsi(:, s_idx), 'LineStyle', line_styles{s_idx}, 'Color', color_rsi, 'LineWidth', line_widths(s_idx));
    lbl_rsi{s_idx} = sprintf('RSI-OLSTEC ($\\delta=%.1f$)', scales_to_test(s_idx));
end

xline(0, 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off'); 
grid on;
ylabel('Relative Error (NRE)', 'FontSize', 13, 'Interpreter', 'latex');
title('(a) Tracking Agility under Varying Mutation Strengths $\delta$', 'FontSize', 14, 'Interpreter', 'latex');
set(gca, 'XTickLabel', [], 'FontName', 'Times New Roman', 'FontSize', 12, 'TickLabelInterpreter', 'latex');
xlim([-window_pre, window_post]); 
ylim([0, max(mean_std(:)) * 1.1]);

% Reorder handles to match MATLAB's column-wise legend population
h_leg = [h_std(1), h_std(2), h_std(3), h_rsi(1), h_rsi(2), h_rsi(3)];
lbl_leg = {lbl_std{1}, lbl_std{2}, lbl_std{3}, lbl_rsi{1}, lbl_rsi{2}, lbl_rsi{3}};
lgd = legend(h_leg, lbl_leg, 'Location', 'northeast', 'NumColumns', 2, 'Interpreter', 'latex');
lgd.FontSize = 10; lgd.Box = 'off';

% --- Subplot 2: Auxiliary Signal (Multi-Scale) ---
subplot(3, 1, 3); hold on;
h_aux = gobjects(num_scales, 1);
lbl_aux = cell(num_scales, 1);

for s_idx = 1:num_scales
    h_aux(s_idx) = plot(x_axis_relative, mean_aux(:, s_idx), 'Color', 'k', ...
         'LineStyle', line_styles{s_idx}, 'LineWidth', line_widths(s_idx));
    lbl_aux{s_idx} = sprintf('Sensor $w_t$ ($\\delta=%.1f$)', scales_to_test(s_idx));
end

grid on;
ylabel('Aux Signal ($w_t$)', 'Interpreter', 'latex', 'FontSize', 13);
xlabel('Relative Time Step ($t=0$ is Mutation)', 'Interpreter', 'latex', 'FontSize', 13);
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
post_mut_range = (zero_idx + 1):(zero_idx + 50); 

for s_idx = 1:num_scales
    fprintf(' MUTATION SCALE: delta = %.1f \n', scales_to_test(s_idx));
    fprintf('%-30s | %-18s | %-18s\n', 'Metric', 'Standard OLSTEC', 'RSI-OLSTEC');
    fprintf('---------------------------------------------------------------------------------\n');
    fprintf('%-30s | %.4f             | %.4f\n', 'Pre-Mutation NRE (Steady)', mean(mean_std(pre_mut_range, s_idx)), mean(mean_rsi(pre_mut_range, s_idx)));
    fprintf('%-30s | %.4f             | %.4f\n', 'Post-Mutation Peak NRE', max(mean_std(post_mut_range, s_idx)), max(mean_rsi(post_mut_range, s_idx)));
    fprintf('%-30s | %.4f             | %.4f\n', 'Recovery Integral (50 frames)', sum(mean_std(post_mut_range, s_idx)), sum(mean_rsi(post_mut_range, s_idx)));
    fprintf('---------------------------------------------------------------------------------\n\n');
end