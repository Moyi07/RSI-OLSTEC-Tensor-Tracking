%% Experiment S3
% =========================================================================
% Description:
%   This script generates controlled synthetic tensor data to theoretically 
%   validate the RSI-OLSTEC algorithm.
%   Core Concept: A 1D auxiliary signal (Side Information) highly correlated 
%   with the temporal factor matrix C is artificially generated to verify 
%   the RSI mechanism's accelerated convergence and noise robustness under 
%   a purely mathematical model.
% =========================================================================
clear; clc; close all;

%% 1. Global Configuration
% -------------------------------------------------------------------------
fprintf('Starting Synthetic Experiment S1...\n');

% Experiment Parameters
tensor_dims     = [50, 50, 500];  % [rows, cols, frames]
rank_r          = 15;             % True tensor CP-rank
fraction        = 0.10;           % Observation ratio (10%)
sparse_ratio    = 0.05;           % Sparse impulsive noise ratio (5%)
tolcost         = 1e-8;
maxepochs       = 1;
verbose         = 0;

% Algorithm Execution Switches
run_cpwopt      = true;  
run_petrels     = true;
run_grasta      = true;
run_grouse      = true;
run_tecpsgd     = true;
run_olstec      = true;
run_rsi_olstec  = true;

%% 2. Synthetic Data Generation
% -------------------------------------------------------------------------
fprintf('=== [Phase 1] Generating Synthetic Tensor and Side Information ===\n');
rows = tensor_dims(1); cols = tensor_dims(2); total_slices = tensor_dims(3);

rng(42); % Fix random seed for reproducibility

% 1. Generate true CP factor matrices
A_true = randn(rows, rank_r);
B_true = randn(cols, rank_r);
C_true = zeros(total_slices, rank_r);

% Generate smooth dynamic changes for temporal factor C_true to simulate physical processes
t = (1:total_slices)';
for r = 1:rank_r
    C_true(:, r) = sin(2 * pi * t / (100 + r*10)) + 0.1 * randn(total_slices, 1);
end

% 2. Construct the clean Ground Truth tensor (Y_Clean)
Tensor_Y_Clean = zeros(rows, cols, total_slices);
for f = 1:total_slices
    Tensor_Y_Clean(:,:,f) = A_true * diag(C_true(f,:)) * B_true';
end
norm_Y_clean = norm(Tensor_Y_Clean(:));

% 3. Generate correlated Side Information (Auxiliary Signal)
aux_info = C_true(:, 1) + 0.2 * randn(total_slices, 1); 

% Normalize auxiliary information to [0, 1] to simulate typical physical sensor readings
aux_info = (aux_info - min(aux_info)) / (max(aux_info) - min(aux_info));

% 4. Inject sparse impulsive noise (Simulating spatter)
Sparse_Noise = zeros(rows, cols, total_slices);
spike_magnitude = max(Tensor_Y_Clean(:)) * 2; 
Sparse_Noise(rand(rows, cols, total_slices) < sparse_ratio) = spike_magnitude;
Tensor_Y_Noisy = Tensor_Y_Clean + Sparse_Noise;

% 5. Apply sampling mask (Omega)
OmegaTensor = rand(rows, cols, total_slices) < fraction;
Matrix_Y_Noisy = reshape(Tensor_Y_Noisy, [rows*cols, total_slices]);
OmegaMatrix = reshape(OmegaTensor, [rows*cols, total_slices]);

% Matrix dimensions and rank
numr = rows * cols; numc = total_slices;  
matrix_rank = rank_r; 

% Random Initialization
Xinit.A = randn(rows, rank_r);
Xinit.B = randn(cols, rank_r);    
Xinit.C = randn(total_slices, rank_r); 

fprintf('Synthetic Data Generated: Rank=%d, Obs=%.1f%%, Noise=%.1f%%\n', rank_r, fraction*100, sparse_ratio*100);

%% 3. Algorithm Execution
% -------------------------------------------------------------------------
fprintf('=== [Phase 2] Running Algorithms ===\n');

if run_cpwopt
    fprintf('Running CP-WOPT (Batch)...\n');
    opts = struct('maxepochs', 30, 'display_iters', 1, 'verbose', verbose, 'store_matrix', false, 'store_subinfo', true);
    tic; [Xsol_cp, ~, sub_infos_cp_wopt] = cp_wopt_mod(Tensor_Y_Noisy, OmegaTensor, [], tensor_dims, rank_r, Xinit, opts); toc;
    
    % Reconstruct offline CP background
    L_cp_full = zeros(rows, cols, total_slices);
    for f = 1:total_slices
        L_cp_full(:,:,f) = Xsol_cp.A * diag(Xsol_cp.C(f,:)) * Xsol_cp.B';
    end
    final_cp_nre = norm(Tensor_Y_Clean(:) - L_cp_full(:)) / norm_Y_clean;
end

if run_petrels
    fprintf('Running Petrels...\n');
    opts = struct('maxepochs', 1, 'tolcost', tolcost, 'permute_on', false, 'rank', matrix_rank, 'lambda', 0.98, 'verbose', 0, 'store_matrix', true, 'store_subinfo', true);
    tic; [~, ~, sub_infos_petrels] = petrels_mod([], Matrix_Y_Noisy, OmegaMatrix, [], numr, numc, opts); toc;
end

if run_grasta
    fprintf('Running GRASTA...\n');
    opts = struct('maxepochs', 1, 'tolcost', tolcost, 'permute_on', false, 'RANK', matrix_rank, 'rho', 1.8, 'ITER_MAX', 20, 'MAX_MU', 10000, 'MIN_MU', 1, 'DIM_M', numr, 'USE_MEX', 0, 'verbose', 0, 'store_matrix', true, 'store_subinfo', true);
    tic; [~, ~, sub_infos_grasta] = grasta_mod([], Matrix_Y_Noisy, OmegaMatrix, [], numr, numc, opts); toc;
end

if run_grouse
    fprintf('Running Grouse...\n');
    opts = struct('maxepochs', 1, 'tolcost', tolcost, 'permute_on', false, 'maxrank', matrix_rank, 'step_size', 0.01, 'verbose', 0, 'store_matrix', true, 'store_subinfo', true);
    tic; [~, ~, sub_infos_grouse] = grouse_mod([], Matrix_Y_Noisy, OmegaMatrix, [], numr, numc, opts); toc;
end

if run_tecpsgd
    fprintf('Running TeCPSGD...\n');
    opts = struct('maxepochs', 1, 'tolcost', tolcost, 'permute_on', false, 'lambda', 0.99, 'stepsize', 0.1, 'mu', 0.01, 'verbose', 0, 'store_matrix', true, 'store_subinfo', true);
    tic; [~, ~, sub_infos_TeCPSGD] = TeCPSGD(Tensor_Y_Noisy, OmegaTensor, [], tensor_dims, rank_r, Xinit, opts); toc;
end

if run_olstec
    fprintf('Running OLSTEC with multiple lambdas...\n');
    lambda_list = [0.70, 0.80, 0.90, 0.99];
    sub_infos_olstec_multi = cell(length(lambda_list), 1);
    for i = 1:length(lambda_list)
        lam = lambda_list(i);
        fprintf('  -> Lambda = %.2f\n', lam);
        opts = struct('maxepochs', 1, 'tolcost', tolcost, 'permute_on', false, 'lambda', lam, 'mu', 0.01, 'tw_flag', 0, 'tw_len', 10, 'verbose', 0, 'store_matrix', true, 'store_subinfo', true);
        [~, ~, sub_infos_olstec_multi{i}] = olstec(Tensor_Y_Noisy, OmegaTensor, [], tensor_dims, rank_r, Xinit, opts);
    end
end

if run_rsi_olstec
    fprintf('Running RSI-OLSTEC...\n');
    opts = struct('maxepochs', 1, 'tolcost', tolcost, 'permute_on', false, 'lambda_max', 0.90, 'lambda_min', 0.70, 'huber_delta', 0.30, 'min_grad_threshold', 0.70, 'mu', 0.01, 'verbose', 0, 'store_matrix', true, 'store_subinfo', true);
    tic; [~, ~, sub_infos_rsi] = rsi_olstec(Tensor_Y_Noisy, OmegaTensor, [], tensor_dims, rank_r, Xinit, opts, aux_info); toc;
end

%% 4. Compute True Normalized Residual Error (NRE)
% -------------------------------------------------------------------------
fprintf('=== [Phase 3] Computing True Errors against Clean Ground Truth ===\n');
true_err = struct();
calc_true_err = @(info) arrayfun(@(t) norm(reshape(Tensor_Y_Clean(:,:,t),[],1) - reshape(info.L(:,t),[],1)) / (norm(reshape(Tensor_Y_Clean(:,:,t),[],1))+1e-10), 1:total_slices);

if run_petrels,  true_err.Petrels = calc_true_err(sub_infos_petrels); end
if run_grasta,   true_err.Grasta  = calc_true_err(sub_infos_grasta); end
if run_grouse,   true_err.Grouse  = calc_true_err(sub_infos_grouse); end

if run_tecpsgd
    if ndims(sub_infos_TeCPSGD.L) == 3
        true_err.TeCPSGD = arrayfun(@(t) norm(Tensor_Y_Clean(:,:,t) - sub_infos_TeCPSGD.L(:,:,t), 'fro') / (norm(Tensor_Y_Clean(:,:,t), 'fro')+1e-10), 1:total_slices);
    else
        true_err.TeCPSGD = calc_true_err(sub_infos_TeCPSGD);
    end
end

if run_olstec
    true_err.OLSTEC = zeros(length(lambda_list), total_slices);
    for i = 1:length(lambda_list)
        true_err.OLSTEC(i,:) = calc_true_err(sub_infos_olstec_multi{i});
    end
end

if run_rsi_olstec, true_err.RSI = calc_true_err(sub_infos_rsi); end

%% 5. Plotting & Visualization
% -------------------------------------------------------------------------
fprintf('=== [Phase 4] Plotting Results ===\n');
x_axis = 1:total_slices;
half_idx = max(1, floor(total_slices / 2)); 
h1 = figure('Name', 'Synthetic: True Residual Error', 'Position', [100, 100, 800, 500], 'Color', 'w'); 
hold on; grid on; box on; 

p_handles = gobjects(0); 
leg_str = {};

% Define HEX color dictionary consistent with other experiments
hex_colors = containers.Map(...
    {'RSI-OLSTEC', 'OLSTEC', 'GRASTA', 'PETRELS', 'TeCPSGD', 'GROUSE', 'CP-WOPT'}, ...
    {'#D95319',    '#0072BD', '#77AC30', '#7E2F8E',  '#4DBEEE', '#EDB120', '#000000'} ...
);

hex_to_rgb = @(hex_str) sscanf(hex_str(2:end), '%2x%2x%2x', [1 3]) / 255;
get_rgb = @(name) hex_to_rgb(hex_colors(name));

valid_errs = []; valid_min = inf;

% 1. CP-WOPT (Batch)
if run_cpwopt && exist('final_cp_nre','var')
    yline(final_cp_nre, '--', 'Color', get_rgb('CP-WOPT'), 'LineWidth', 2.0, ...
        'Label', 'CP-WOPT (Batch)', 'LabelHorizontalAlignment', 'left', ...
        'FontSize', 12, 'HandleVisibility', 'off'); 
end

% 2. GROUSE
if run_grouse && exist('true_err','var') && isfield(true_err, 'Grouse')
    h = semilogy(x_axis, true_err.Grouse, '-', 'Color', get_rgb('GROUSE'), 'LineWidth', 1.5); 
    p_handles(end+1) = h; leg_str{end+1} = 'GROUSE'; 
end

% 3. GRASTA
if run_grasta && exist('true_err','var') && isfield(true_err, 'Grasta')
    h = semilogy(x_axis, true_err.Grasta, '-', 'Color', get_rgb('GRASTA'), 'LineWidth', 1.5); 
    p_handles(end+1) = h; leg_str{end+1} = 'GRASTA'; 
    valid_errs = [valid_errs; true_err.Grasta(half_idx:end)']; valid_min = min(valid_min, min(true_err.Grasta(true_err.Grasta>0)));
end

% 4. PETRELS
if run_petrels && exist('true_err','var') && isfield(true_err, 'Petrels')
    h = semilogy(x_axis, true_err.Petrels, '-', 'Color', get_rgb('PETRELS'), 'LineWidth', 1.5); 
    p_handles(end+1) = h; leg_str{end+1} = 'PETRELS'; 
    valid_errs = [valid_errs; true_err.Petrels(half_idx:end)']; valid_min = min(valid_min, min(true_err.Petrels(true_err.Petrels>0)));
end

% 5. TeCPSGD
if run_tecpsgd && exist('true_err','var') && isfield(true_err, 'TeCPSGD')
    h = semilogy(x_axis, true_err.TeCPSGD, '-', 'Color', get_rgb('TeCPSGD'), 'LineWidth', 1.5); 
    p_handles(end+1) = h; leg_str{end+1} = 'TeCPSGD'; 
    valid_errs = [valid_errs; true_err.TeCPSGD(half_idx:end)']; valid_min = min(valid_min, min(true_err.TeCPSGD(true_err.TeCPSGD>0)));
end

% 6. OLSTEC
if run_olstec && exist('true_err','var') && isfield(true_err, 'OLSTEC')
    olstec_linespecs = {':', '--', '-', '-.'}; 
    for i = 1:length(lambda_list)
        h = semilogy(x_axis, true_err.OLSTEC(i, :), olstec_linespecs{i}, 'Color', get_rgb('OLSTEC'), 'LineWidth', 1.5);
        p_handles(end+1) = h; leg_str{end+1} = sprintf('OLSTEC ($\\lambda=%.2f$)', lambda_list(i));
    end
    valid_errs = [valid_errs; reshape(true_err.OLSTEC(:, half_idx:end), [], 1)];
    valid_min = min(valid_min, min(true_err.OLSTEC(true_err.OLSTEC>0)));
end

% 7. RSI-OLSTEC (Ours)
if run_rsi_olstec && exist('true_err','var') && isfield(true_err, 'RSI')
    h = semilogy(x_axis, true_err.RSI, '-', 'Color', get_rgb('RSI-OLSTEC'), 'LineWidth', 2.0); 
    p_handles(end+1) = h; leg_str{end+1} = '\textbf{RSI-OLSTEC}'; 
    valid_errs = [valid_errs; true_err.RSI(half_idx:end)']; valid_min = min(valid_min, min(true_err.RSI(true_err.RSI>0)));
end

hold off; 

% Global font and axis formatting
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12, 'TickLabelInterpreter', 'latex');
xlabel('Time Index (Frames)', 'Interpreter', 'latex', 'FontSize', 14); 
ylabel('True NRE (log)', 'Interpreter', 'latex', 'FontSize', 14);

lgd = legend(p_handles, leg_str, 'Location', 'southwest', 'FontSize', 11); 
lgd.Interpreter = 'latex'; lgd.Box = 'off';

% Dynamic Y-axis limits and X-axis truncation
if ~isempty(valid_errs) && valid_min > 0
    ylim([valid_min * 0.8, max(valid_errs(:)) * 2.5]); 
end
xlim([1, total_slices]);

title('Convergence Performance on Synthetic Data', 'Interpreter', 'latex', 'FontSize', 14, 'FontWeight', 'bold');

%% 6. Quantitative Results Summary
% -------------------------------------------------------------------------
fprintf('\n===========================================================\n');
fprintf('SYNTHETIC RESULTS: FINAL TRUE ERROR\n');
fprintf('===========================================================\n');
if run_cpwopt,  fprintf('%-20s | %-20.6e\n', 'CP-WOPT', final_cp_nre); end
if run_tecpsgd, fprintf('%-20s | %-20.6e\n', 'TeCPSGD', true_err.TeCPSGD(end)); end
if run_petrels, fprintf('%-20s | %-20.6e\n', 'Petrels', true_err.Petrels(end)); end
if run_grouse,  fprintf('%-20s | %-20.6e\n', 'Grouse', true_err.Grouse(end)); end
if run_grasta,  fprintf('%-20s | %-20.6e\n', 'Grasta', true_err.Grasta(end)); end
if run_olstec
    for i = 1:length(lambda_list)
        fprintf('%-20s | %-20.6e\n', sprintf('OLSTEC (lam=%.2f)', lambda_list(i)), true_err.OLSTEC(i, end)); 
    end
end
if run_rsi_olstec, fprintf('%-20s | %-20.6e\n', 'RSI-OLSTEC (Ours)', true_err.RSI(end)); end
fprintf('===========================================================\n');