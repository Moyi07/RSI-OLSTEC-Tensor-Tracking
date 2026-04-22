%% Experiment S2
% =========================================================================
% Objective: 
%   To validate the robustness of the RSI-OLSTEC algorithm in an environment 
%   contaminated with impulsive noise, which simulates the 'spatter' phenomenon 
%   often observed in Wire Arc Additive Manufacturing (WAAM) processes.
%
% Experimental Design (Ablation Study):
%   This script compares two variants to isolate the effect of the loss function:
%   1. RSI-OLSTEC (Robust): Enabled with Huber Loss for outlier suppression.
%   2. RSI-OLSTEC (Non-Robust/L2): Degraded to standard L2 norm (Huber disabled).
% =========================================================================
clear; close all; clc;

%% 1. Monte Carlo Configuration
% -------------------------------------------------------------------------
n_monte_carlo = 100;  
disp(['Starting Rigorous Monte Carlo Simulation with ', num2str(n_monte_carlo), ' runs...']);

% Tensor Dimensions & Common Parameters
I = 100; J = 100; T = 500; 
dims = [I, J, T];
true_rank = 5; 

% Data Generation Parameters
SNR_dB = 25;           
outlier_ratio = 0.05;  
spatter_base_mag = 1.0;
missing_rate = 0.50;   
drift_rate = 1e-4;

% Storage for Statistics
history_err_robust = zeros(T, n_monte_carlo); 
history_err_l2     = zeros(T, n_monte_carlo); 
history_f1         = zeros(n_monte_carlo, 1); 
history_time_rob   = zeros(n_monte_carlo, 1); 
history_time_l2    = zeros(n_monte_carlo, 1); 

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
    
    X_true = zeros(I, J, T);
    c_t_current = 10.0 * ones(true_rank, 1);
    
    for t = 1:T
        % 1. Spatial drift: QR decomposition with sign protection for manifold continuity
        [Q_A, R_A] = qr(A_curr + drift_rate * randn(I, true_rank), 0);
        A_curr = Q_A * diag(sign(diag(R_A) + 1e-10));
        
        [Q_B, R_B] = qr(B_curr + drift_rate * randn(J, true_rank), 0);
        B_curr = Q_B * diag(sign(diag(R_B) + 1e-10));
        
        % 2. Temporal evolution: AR(1) process to ensure energy fluctuation stability
        c_t_current = 10.0 + 0.98 * (c_t_current - 10.0) + 0.1 * randn(true_rank, 1);
        
        X_true(:, :, t) = A_curr * diag(c_t_current) * B_curr';
    end
    
    % Calculate dynamic background noise based on Signal Power
    sig_pow = norm(X_true(:))^2 / numel(X_true);
    noise_sigma = sqrt(sig_pow / 10^(SNR_dB/10)); 
    
    % Background Sensor Noise
    Gaussian_Noise = noise_sigma * randn(I, J, T);
    
    % Sparse Impulsive Noise (Simulating WAAM Spatter)
    S_mask = rand(I, J, T) < outlier_ratio;
    burn_in_frames = 30;
    
    Sparse_Noise = zeros(I, J, T);
    num_outliers = sum(S_mask(:));
    
    Sparse_Noise(S_mask) = spatter_base_mag * randn(num_outliers, 1);
    
    % Final Observation Tensor
    Y_full = X_true + Gaussian_Noise + Sparse_Noise;
    Omega = rand(I, J, T) > missing_rate;
    Y_observed = Y_full .* Omega;
    
    Aux_Signal = 10.0 * ones(T, 1); 
    
    % --- B. Algorithm Initialization ---
    [X_init.A, ~] = qr(randn(I, true_rank), 0);
    [X_init.B, ~] = qr(randn(J, true_rank), 0);
    X_init.C = zeros(T, true_rank);
    
    % Unsupervised Temporal-Difference MAD Estimation
    % Cancels out the spatial contrast of the background to isolate pure sensor noise.
    diff_pixels = [];
    for t_idx = 2:burn_in_frames
        common_mask = Omega(:,:,t_idx) & Omega(:,:,t_idx-1);
        diff_val = Y_observed(:,:,t_idx) - Y_observed(:,:,t_idx-1);
        diff_pixels = [diff_pixels; diff_val(common_mask)];
    end
    
    mad_val = median(abs(diff_pixels - median(diff_pixels)));
    est_sigma = (1.4826 * mad_val) / sqrt(2);
    
    % Algorithm is strictly denied the ground truth noise_sigma
    huber_threshold_est = 3 * est_sigma;
    
    % Lock temporal adaptation to isolate spatial robustness variable
    opts_common = struct('mu', 0.01, 'verbose', 0, ...
                         'lambda_max', 0.90, 'lambda_min', 0.90, ... 
                         'min_grad_threshold', Inf, ...              
                         'maxepochs', 1, 'tolcost', 1e-8, ...
                         'store_matrix', true, 'store_subinfo', true); 
    
    % Config 1: RSI-OLSTEC (Robust / Huber)
    opts_robust = opts_common;
    opts_robust.huber_delta = huber_threshold_est;
    
    % Config 2: RSI-OLSTEC (Non-Robust / L2)
    opts_l2 = opts_common;
    opts_l2.huber_delta = Inf; 
    
    % --- C. Execution ---
    tic;
    [~, ~, sub_infos_robust] = rsi_olstec(Y_observed, Omega, [], dims, true_rank, X_init, opts_robust, Aux_Signal);
    history_time_rob(mc) = toc;
    
    tic;
    [~, ~, sub_infos_l2] = rsi_olstec(Y_observed, Omega, [], dims, true_rank, X_init, opts_l2, Aux_Signal);
    history_time_l2(mc) = toc;
    
    % --- D. Performance Evaluation ---
    temp_err_rob = zeros(T, 1);
    temp_err_l2  = zeros(T, 1);
    
    for t = 1:T
        X_gt = X_true(:, :, t);
        norm_gt = norm(X_gt, 'fro');
        temp_err_rob(t) = norm(X_gt - reshape(sub_infos_robust.L(:, t), [I, J]), 'fro') / (norm_gt + 1e-10);
        temp_err_l2(t)  = norm(X_gt - reshape(sub_infos_l2.L(:, t), [I, J]), 'fro') / (norm_gt + 1e-10);
    end
    
    history_err_robust(:, mc) = temp_err_rob;
    history_err_l2(:, mc)     = temp_err_l2;
    
    % --- E. Unsupervised Outlier Detection (F1-Score) ---
    Omega_vec = reshape(Omega, [I*J, T]);
    
    Significant_Spatter = abs(Sparse_Noise) > (3 * noise_sigma);
    True_Outliers = reshape(Significant_Spatter, [I*J, T]) & Omega_vec;
    
    % Calculate residuals strictly using causal online outputs (sub_infos.L)
    Mat_Y = reshape(Y_observed, [I*J, T]);
    L_rob_mat = reshape(sub_infos_robust.L, [I*J, T]); 
    Residuals_Robust = abs(Mat_Y - L_rob_mat) .* Omega_vec; 
    Detected_Outliers = (Residuals_Robust > huber_threshold_est) & Omega_vec;
    
    eval_mask = false(1, T); eval_mask((burn_in_frames + 1):T) = true;
    
    TP = sum(sum(Detected_Outliers(:, eval_mask) & True_Outliers(:, eval_mask)));
    FP = sum(sum(Detected_Outliers(:, eval_mask) & ~True_Outliers(:, eval_mask)));
    FN = sum(sum(~Detected_Outliers(:, eval_mask) & True_Outliers(:, eval_mask)));
    
    precision = TP / (TP + FP + 1e-10);
    recall    = TP / (TP + FN + 1e-10);
    history_f1(mc)  = 2 * (precision * recall) / (precision + recall + 1e-10);
    
    if mod(mc, 2) == 0 || mc == 1, fprintf('Completed Trial %d/%d\n', mc, n_monte_carlo); end
end

%% 3. Statistical Analysis & Visualization
% -------------------------------------------------------------------------
mean_err_rob = mean(history_err_robust, 2); std_err_rob  = std(history_err_robust, 0, 2);
mean_err_l2  = mean(history_err_l2, 2);     std_err_l2   = std(history_err_l2, 0, 2);

figure('Name', 'Exp S2: Spatter Robustness', 'Color', 'w', 'Position', [100, 100, 1200, 600]);

% Subplot 1: Convergence Curves
subplot(1, 2, 1); x_axis = (1:T)'; hold on;

% Color palette for consistency
color_rsi = [0.8500 0.3250 0.0980]; 
color_l2  = [0.0000 0.4470 0.7410]; 

% Plot Baseline L2
fill([x_axis; flipud(x_axis)], max(1e-10, [mean_err_l2 + std_err_l2; flipud(mean_err_l2 - std_err_l2)]), ...
    color_l2, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
semilogy(x_axis, mean_err_l2, '-', 'Color', color_l2, 'LineWidth', 1.5, 'DisplayName', 'Non-Robust (L2 Norm)');

% Plot RSI-OLSTEC
fill([x_axis; flipud(x_axis)], max(1e-10, [mean_err_rob + std_err_rob; flipud(mean_err_rob - std_err_rob)]), ...
    color_rsi, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
semilogy(x_axis, mean_err_rob, '-', 'Color', color_rsi, 'LineWidth', 1.5, 'DisplayName', 'RSI-OLSTEC (Huber Penalty)');

grid on; set(gca, 'YScale', 'log');
xlabel('Time Index (Frame)'); ylabel('Average Relative Error (NRE)'); 
title('(a) Convergence under Spatter Noise'); legend('Location', 'northeast'); axis tight;

% Subplot 2: Steady-state Error Distribution
subplot(1, 2, 2);
steady_rob = mean(history_err_robust((T-100):T, :), 1)';
steady_l2  = mean(history_err_l2((T-100):T, :), 1)';
boxplot([steady_l2, steady_rob], 'Labels', {'Non-Robust (L2)', 'RSI-OLSTEC (Huber)'});
ylabel('Mean Steady-state Error (Last 100 frames)');
title('(b) Error Distribution Stability'); grid on;

% --- Subplot 3: Spatial Residual Snapshot ---
t_snap = T - 50; 
figure('Name', 'Spatial Residual', 'Color', 'w', 'Position', [150, 150, 1000, 350]);

subplot(1, 3, 1);
imagesc(X_true(:, :, t_snap)); axis off image; colorbar;
title('Ground Truth Background');

% L2 Residual
subplot(1, 3, 2);
Res_l2 = abs(X_true(:, :, t_snap) - reshape(sub_infos_l2.L(:, t_snap), I, J));
imagesc(Res_l2); axis off image; colorbar;
c_max = max(Res_l2(:)) * 0.8; 
clim([0, c_max]); title(sprintf('L2 Error (NRE: %.3f)', temp_err_l2(t_snap)));

% Huber Residual (Strictly using the same color scale as L2 for fair comparison)
subplot(1, 3, 3);
Res_rob = abs(X_true(:, :, t_snap) - reshape(sub_infos_robust.L(:, t_snap), I, J));
imagesc(Res_rob); axis off image; colorbar;
clim([0, c_max]); 
title(sprintf('Huber Error (NRE: %.3f)', temp_err_rob(t_snap)));

%% 4. Quantitative Results
fprintf('\n=================================================================================\n');
fprintf('   EXP S2 SUMMARY: L2 Norm vs. Huber Penalty (Spatter Noise)\n');
fprintf('=================================================================================\n');
fprintf('%-28s | %-20s | %-20s\n', 'Metric', 'Non-Robust (L2)', 'RSI-OLSTEC (Huber)');
fprintf('---------------------------------------------------------------------------------\n');
fprintf('%-28s | %.4f +/- %.4f  | %.4f +/- %.4f\n', 'Steady-State NRE', mean(steady_l2), std(steady_l2), mean(steady_rob), std(steady_rob));
fprintf('%-28s | %-20s | %.4f +/- %.4f\n', 'Outlier Det. F1-Score', 'N/A', mean(history_f1), std(history_f1));
fprintf('=================================================================================\n');