%% Experiment S4
% =========================================================================
% Description:
%   This script conducts a dual-dimension ablation study evaluating both 
%   Observation Ratio and Tensor Rank.
%   It generates an N-by-2 matrix of subplots:
%     - Each row corresponds to a specific Rank (e.g., Rank=5, 10, 15).
%     - Left column (a, c, e...): Macro trend bar charts of steady-state 
%       error across different observation ratios.
%     - Right column (b, d, f...): Micro convergence curves at the extreme 
%       target observation ratio.
% =========================================================================
clear; clc; close all;

%% 1. Core Experimental Variables (Dual Independent Variables)
% -------------------------------------------------------------------------
test_fractions = [0.70, 0.50, 0.30, 0.10]; % Independent Variable 1: Observation Ratio (70%, 50%, 30%, 10%)
test_ranks     = [5, 10, 15];              % Independent Variable 2: Background complexity / Tensor Rank
target_ts_fraction = 0.10;                 % Target observation ratio for the time-series curves (10% extreme test)
num_trials   = 100;                        % Number of Monte Carlo trials
tensor_dims  = [50, 50, 500]; 
sparse_ratio = 0.05;          
tolcost      = 1e-8;

alg_names = {'PETRELS', 'GRASTA', 'TeCPSGD', 'OLSTEC', 'RSI-OLSTEC'};
num_algs = length(alg_names);

num_fracs = length(test_fractions);
num_ranks = length(test_ranks);

% 3D Storage Matrices: (Algorithms) x (Observation Ratios) x (Ranks)
mean_errors_3D = zeros(num_algs, num_fracs, num_ranks);
std_errors_3D  = zeros(num_algs, num_fracs, num_ranks);
time_series_3D = zeros(num_algs, tensor_dims(3), num_ranks);

fprintf('Starting Comprehensive Cross-Ablation (Monte Carlo: %d Trials)...\n', num_trials);
total_start = tic;

%% 2. Dual-Loop Monte Carlo Simulation: Rank (Outer) & Fraction (Inner)
% -------------------------------------------------------------------------
for r_idx = 1:num_ranks
    rank_r = test_ranks(r_idx);
    fprintf('\n======================================================\n');
    fprintf('Main Group %d/%d: Target Rank = %d\n', r_idx, num_ranks, rank_r);
    fprintf('======================================================\n');
    
    for exp_idx = 1:num_fracs
        fraction = test_fractions(exp_idx);
        fprintf('\n▶ Subgroup %d/%d: Observation Fraction = %.2f\n', exp_idx, num_fracs, fraction);
        
        capture_ts = (abs(fraction - target_ts_fraction) < 1e-4);
        current_fraction_errors = zeros(num_algs, num_trials);
        current_ts_curves       = zeros(num_algs, tensor_dims(3));
        
        for trial = 1:num_trials
            fprintf('  - Trial %d/%d... ', trial, num_trials);
            
            % --- Dynamic Data Generation ---
            rows = tensor_dims(1); cols = tensor_dims(2); T = tensor_dims(3);
            rng(42 + r_idx * 1000 + exp_idx * 100 + trial); 
            
            A_true = randn(rows, rank_r); B_true = randn(cols, rank_r);
            C_true = zeros(T, rank_r); t_idx = (1:T)';
            for r = 1:rank_r
                C_true(:, r) = sin(2 * pi * t_idx / (100 + r*10)) + 0.1 * randn(T, 1);
            end
            
            Tensor_Y_Clean = zeros(rows, cols, T);
            for f = 1:T, Tensor_Y_Clean(:,:,f) = A_true * diag(C_true(f,:)) * B_true'; end
            
            aux_info = C_true(:, 1) + 0.2 * randn(T, 1); 
            aux_info = (aux_info - min(aux_info)) / (max(aux_info) - min(aux_info));
            
            Sparse_Noise = zeros(rows, cols, T);
            Sparse_Noise(rand(rows, cols, T) < sparse_ratio) = max(Tensor_Y_Clean(:)) * 2; 
            
            Tensor_Y_Noisy = Tensor_Y_Clean + Sparse_Noise;
            OmegaTensor = rand(rows, cols, T) < fraction;
            
            Matrix_Y_Noisy = reshape(Tensor_Y_Noisy, [rows*cols, T]);
            OmegaMatrix = reshape(OmegaTensor, [rows*cols, T]);
            numr = rows * cols; numc = T;  
            
            Xinit.A = randn(rows, rank_r); Xinit.B = randn(cols, rank_r); Xinit.C = randn(T, rank_r); 
            
            calc_true_nre = @(L_out) arrayfun(@(idx) norm(reshape(Tensor_Y_Clean(:,:,idx),[],1) - reshape(L_out(:,idx),[],1)) / (norm(reshape(Tensor_Y_Clean(:,:,idx),[],1))+1e-10), 1:T);
            
            % --- Algorithm Execution ---
            for a = 1:num_algs
                algo_name = alg_names{a};
                err_curve = ones(1, T); 
                
                try
                    switch algo_name
                        case 'PETRELS'
                            opts = struct('maxepochs', 1, 'tolcost', tolcost, 'permute_on', 0, 'rank', rank_r, 'lambda', 0.98, 'verbose', 0, 'store_matrix', 1, 'store_subinfo', 1);
                            [~, ~, info] = petrels_mod([], Matrix_Y_Noisy, OmegaMatrix, [], numr, numc, opts);
                            err_curve = calc_true_nre(info.L);
                        case 'GRASTA'
                            opts = struct('maxepochs', 1, 'tolcost', tolcost, 'permute_on', 0, 'RANK', rank_r, 'rho', 1.8, 'ITER_MAX', 20, 'MAX_MU', 10000, 'MIN_MU', 1, 'DIM_M', numr, 'USE_MEX', 0, 'verbose', 0, 'store_matrix', 1, 'store_subinfo', 1);
                            [~, ~, info] = grasta_mod([], Matrix_Y_Noisy, OmegaMatrix, [], numr, numc, opts);
                            err_curve = calc_true_nre(info.L);
                        case 'TeCPSGD'
                            opts = struct('maxepochs', 1, 'tolcost', tolcost, 'permute_on', 0, 'lambda', 0.99, 'stepsize', 0.1, 'mu', 0.01, 'verbose', 0, 'store_matrix', 1, 'store_subinfo', 1);
                            [~, ~, info] = TeCPSGD(Tensor_Y_Noisy, OmegaTensor, [], tensor_dims, rank_r, Xinit, opts);
                            err_curve = calc_true_nre(reshape(info.L, [numr, numc]));
                        case 'OLSTEC'
                            opts = struct('maxepochs', 1, 'tolcost', tolcost, 'permute_on', 0, 'lambda', 0.99, 'mu', 0.01, 'tw_flag', 0, 'tw_len', 10, 'verbose', 0, 'store_matrix', 1, 'store_subinfo', 1);
                            [~, ~, info] = olstec(Tensor_Y_Noisy, OmegaTensor, [], tensor_dims, rank_r, Xinit, opts);
                            err_curve = calc_true_nre(info.L);
                        case 'RSI-OLSTEC'
                            opts = struct('maxepochs', 1, 'tolcost', tolcost, 'permute_on', 0, 'lambda_max', 0.90, 'lambda_min', 0.70, 'huber_delta', 0.30, 'min_grad_threshold', 0.70, 'mu', 0.01, 'verbose', 0, 'store_matrix', 1, 'store_subinfo', 1);
                            [~, ~, info] = rsi_olstec(Tensor_Y_Noisy, OmegaTensor, [], tensor_dims, rank_r, Xinit, opts, aux_info);
                            err_curve = calc_true_nre(info.L);
                    end
                    err_curve(isnan(err_curve) | isinf(err_curve)) = 1.0;
                catch ME
                    if capture_ts && trial == 1, fprintf(' [Failed] '); end
                end
                
                current_fraction_errors(a, trial) = mean(err_curve(end-99:end));
                if capture_ts, current_ts_curves(a, :) = current_ts_curves(a, :) + err_curve; end
            end
            fprintf('OK.\n');
        end
        
        mean_errors_3D(:, exp_idx, r_idx) = mean(current_fraction_errors, 2);
        std_errors_3D(:, exp_idx, r_idx)  = std(current_fraction_errors, 0, 2);
        if capture_ts
            time_series_3D(:, :, r_idx) = current_ts_curves / num_trials;
        end
    end
end
fprintf('\nAll Cross-Ablation Experiments completed in %.1f seconds.\n', toc(total_start));

%% 3. Visualization: Matrix Subplots
% -------------------------------------------------------------------------
fig_height = 350 * num_ranks;
figure('Name', 'Robustness against Sparsity & Rank', 'Position', [100, 50, 1200, fig_height], 'Color', 'w');

hex_colors = containers.Map(...
    {'RSI-OLSTEC', 'OLSTEC', 'GRASTA', 'PETRELS', 'TeCPSGD'}, ...
    {'#D95319',    '#0072BD', '#77AC30', '#7E2F8E',  '#4DBEEE'} ...
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
        errorbar(x_pos, cur_mean(k,:), zeros(1, num_fracs), cur_std(k,:), 'k', 'LineStyle', 'none', 'LineWidth', 1.2, 'CapSize', 6);
    end
    set(gca, 'YScale', 'log', 'FontName', 'Times New Roman', 'FontSize', 12); 
    set(gca, 'XTick', 1:num_fracs, 'XTickLabel', arrayfun(@(x) sprintf('%.0f%%', x*100), test_fractions, 'UniformOutput', false));
    ylabel('Steady-State NRE (log)', 'Interpreter', 'latex', 'FontSize', 13);
    
    title_str = sprintf('(%s) Macro Trend (Rank = %d)', alphabet(2 * r_idx - 1), rank_r);
    title(title_str, 'Interpreter', 'latex', 'FontSize', 13, 'FontWeight', 'bold');
    
    % Display legend only on the first row to avoid clutter
    if r_idx == 1, legend(hb, alg_names, 'Location', 'northwest', 'FontSize', 10); end
    if r_idx == num_ranks, xlabel('Observation Ratio ($\rho$)', 'Interpreter', 'latex', 'FontSize', 13); end
    ylim([min(cur_mean(:)) * 0.5, max(cur_mean(:)) * 1.5]);
    
    % --- Right Column: Time-Series Curves (Micro Convergence) ---
    subplot(num_ranks, 2, 2 * r_idx); hold on; box on; grid on;
    
    cur_ts = time_series_3D(:,:,r_idx);
    for a = 1:num_algs
        plot(x_axis, cur_ts(a, :), 'Color', colors(a,:), 'LineWidth', 1.5, 'DisplayName', alg_names{a});
    end
    set(gca, 'YScale', 'log', 'FontName', 'Times New Roman', 'FontSize', 12);
    ylabel('Instant NRE (log)', 'Interpreter', 'latex', 'FontSize', 13);
    
    title_str = sprintf('(%s) Micro Convergence at $\\rho=%.0f\\%%$ (Rank = %d)', alphabet(2 * r_idx), target_ts_fraction*100, rank_r);
    title(title_str, 'Interpreter', 'latex', 'FontSize', 13, 'FontWeight', 'bold');
    
    if r_idx == 1, legend('Location', 'northeast', 'FontSize', 10); end
    if r_idx == num_ranks, xlabel('Time Index (Frames)', 'Interpreter', 'latex', 'FontSize', 13); end
    xlim([1, tensor_dims(3)]); ylim([min(cur_ts(:)) * 0.5, max(cur_ts(:)) * 1.5]);
end
fprintf('Comprehensive cross-ablation matrix plot generated successfully.\n');