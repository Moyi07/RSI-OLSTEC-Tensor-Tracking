%% Experiment R3
% =========================================================================
% Objective:
%   Evaluate the performance of RSI-OLSTEC against baseline algorithms 
%   on real-world video data with synthetic noise (spatter).
% =========================================================================
clear;
clc;
close all;

%% 1. Configuration
% -------------------------------------------------------------------------
% Algorithm Switches
run_cpwopt      = true;  
run_petrels     = true;
run_grasta      = true;
run_grouse      = true;
run_tecpsgd     = true;
run_olstec      = true;
run_rsi_olstec  = true;

% Experiment Parameters
image_display_flag  = true;   
store_matrix_flag   = true;
permute_on_flag     = false;
maxepochs           = 1;
verbose             = 0;   
tolcost             = 1e-8;
rank_r              = 20;     
fraction            = 0.1;    

% Data Paths
video_filename = 'RSI_OLSTEC\dataset\video\250312-110206-video_1.mp4';
meta_filename  = 'RSI_OLSTEC\dataset\WAMVID_metadata.csv';

% Preprocessing Parameters
scale_ratio = 0.2;     
max_frames  = 623;     

%% 2. Data Loading & Preprocessing
% -------------------------------------------------------------------------
fprintf('Reading video: %s ...\n', video_filename);
if ~exist(video_filename, 'file')
    error('Video file not found: %s', video_filename);
end

v = VideoReader(video_filename);
Tensor_Y_Noiseless = [];
frame_idx = 0;

while hasFrame(v) && frame_idx < max_frames
    frame_idx = frame_idx + 1;
    raw_frame = readFrame(v);
    
    if size(raw_frame, 3) == 3
        gray_frame = rgb2gray(raw_frame);
    else
        gray_frame = raw_frame;
    end
    
    img_resized = imresize(gray_frame, scale_ratio);
    Tensor_Y_Noiseless(:, :, frame_idx) = im2double(img_resized);
end

[rows, cols, total_slices] = size(Tensor_Y_Noiseless);
tensor_dims = [rows, cols, total_slices];
fprintf('Preprocessing complete.\n');
fprintf('Resolution: %d x %d \n', rows, cols);
fprintf('Total frames: %d\n', total_slices);

rng(42); 
OmegaTensor = rand(rows, cols, total_slices) < fraction;

% -------------------------------------------------------------------------
% Separation of Ground Truth and Observed Data
% -------------------------------------------------------------------------
% 1. Clean Ground Truth Data
Tensor_Y_Clean = Tensor_Y_Noiseless; 
Matrix_Y_Clean = reshape(Tensor_Y_Clean, [rows*cols, total_slices]);

% 2. Noisy Observation Data (Synthetic Spatter Injection)
fprintf('Injecting synthetic heavy spatter noise (1%% pixels)...\n');
rng(100); 
spatter_mask = rand(rows, cols, total_slices) < 0.01; 
Tensor_Y_Noisy = Tensor_Y_Clean; 
Tensor_Y_Noisy(spatter_mask) = 1.0; 

Matrix_Y_Noisy = reshape(Tensor_Y_Noisy, [rows*cols, total_slices]);
OmegaMatrix = reshape(OmegaTensor, [rows*cols, total_slices]);

% Calculate Matrix Rank
numr = rows * cols;
numc = total_slices;  
num_params_of_tensor = rank_r * sum(tensor_dims);
matrix_rank = floor( num_params_of_tensor/ (numr+numc) );
if matrix_rank < 1, matrix_rank = 1; end

% Initialize Factor Matrices
Xinit.A = randn(rows, rank_r);
Xinit.B = randn(cols, rank_r);    
Xinit.C = randn(total_slices, rank_r); 

%% 3. Load Auxiliary Data (Side Information)
% -------------------------------------------------------------------------
fprintf('Loading auxiliary data...\n');
if exist(meta_filename, 'file')
    meta_table = readtable(meta_filename);
    
    [~, vid_name, vid_ext] = fileparts(video_filename);
    target_vid_name = [vid_name, vid_ext]; 
    
    row_idx = find(contains(meta_table.Video_filepath, target_vid_name), 1);
    
    if isempty(row_idx)
        error('Metadata not found for video: %s', target_vid_name);
    else
        csv_matched_path = meta_table.Video_filepath{row_idx};
        fprintf('Matched video [%s] to CSV row %d\n', target_vid_name, row_idx);
    end
    
    width_str = meta_table.Width_mm{row_idx}; 
    width_str = strrep(width_str, '[', '');
    width_str = strrep(width_str, ']', '');
    width_str = strrep(width_str, 'nan', 'NaN');
    width_vals = str2num(width_str); %#ok<ST2NM>
    
    if length(width_vals) > max_frames
        aux_width = width_vals(1:max_frames)';
    else
        aux_width = [width_vals'; repmat(width_vals(end), max_frames-length(width_vals), 1)];
    end
    
    aux_width = fillmissing(aux_width, 'previous');
    if isnan(aux_width(1)), aux_width(1) = 0; end
else
    warning('Metadata file not found. Using random data.');
    aux_width = rand(max_frames, 1);
end

%% 4. Algorithm Execution
% -------------------------------------------------------------------------
%% (1) CP-WOPT (Batch)
if run_cpwopt
    fprintf('Running CP-WOPT...\n');
    clear options;   
    options.maxepochs       = 30;
    options.display_iters   = 1;
    options.store_subinfo   = true;     
    options.store_matrix    = store_matrix_flag; 
    options.verbose         = verbose; 
    
    tic;
    [Xsol_cp_wopt, info_cp_wopt, sub_infos_cp_wopt] = cp_wopt_mod(Tensor_Y_Noisy, OmegaTensor, [], tensor_dims, rank_r, Xinit, options);
    elapsed_time_cpwopt = toc;
end

%% (2) Petrels (Matrix)
if run_petrels
    fprintf('Running Petrels...\n');
    clear options;
    options.maxepochs           = maxepochs;
    options.tolcost             = tolcost;
    options.rank                = matrix_rank;
    options.permute_on          = permute_on_flag;    
    options.store_subinfo       = true;     
    options.store_matrix        = store_matrix_flag; 
    options.verbose             = verbose;
    options.lambda              = 0.99;
    
    tic; 
    [Xsol_petrels, infos_petrels, sub_infos_petrels, ~] = petrels_mod([], Matrix_Y_Noisy, OmegaMatrix, [], numr, numc, options);    
    elapsed_time_petrels = toc;    
end

%% (3) GRASTA (Matrix)
if run_grasta
    fprintf('Running GRASTA...\n');
    clear options;
    options.maxepochs           = maxepochs;
    options.tolcost             = tolcost;
    options.permute_on          = permute_on_flag;    
    options.verbose             = verbose;
    options.store_subinfo       = true;     
    options.store_matrix        = store_matrix_flag; 
    options.RANK                = matrix_rank;
    options.rho                 = 1.8;    
    options.MAX_MU              = 10000;
    options.MIN_MU              = 1;
    options.ITER_MAX            = 20; 
    options.DIM_M               = rows * cols;
    options.USE_MEX             = 0;                                     
    tic; 
    [Xsol_grasta, infos_grasta, sub_infos_grasta, ~] = grasta_mod([], Matrix_Y_Noisy, OmegaMatrix, [], numr, numc, options);
    elapsed_time_grasta = toc;
end

%% (4) Grouse (Matrix)
if run_grouse
    fprintf('Running Grouse...\n');
    clear options;    
    options.maxrank         = matrix_rank;
    options.step_size       = 0.0001;
    options.maxepochs       = maxepochs;       
    options.tolcost         = tolcost;
    options.permute_on      = permute_on_flag;    
    options.store_subinfo   = true;     
    options.store_matrix    = store_matrix_flag; 
    options.verbose         = verbose;   
    tic;        
    [Xsol_grouse, infos_grouse, sub_infos_grouse, ~] = grouse_mod([], Matrix_Y_Noisy, OmegaMatrix, [], numr, numc, options);
    elapsed_time_grouse = toc;
end

%% (5) TeCPSGD
if run_tecpsgd
    fprintf('Running TeCPSGD...\n');
    clear options;   
    options.maxepochs       = maxepochs;
    options.tolcost         = tolcost;
    options.lambda          = 0.99;
    options.stepsize        = 0.10;
    options.mu              = 0.01;
    options.permute_on      = permute_on_flag;    
    options.store_subinfo   = true;     
    options.store_matrix    = store_matrix_flag; 
    options.verbose         = verbose; 
    tic; 
    [Xsol_TeCPSGD, info_TeCPSGD, sub_infos_TeCPSGD] = TeCPSGD(Tensor_Y_Noisy, OmegaTensor, [], tensor_dims, rank_r, Xinit, options);
    elapsed_time_tecpsgd = toc;    
end

%% (6) OLSTEC
if run_olstec
    fprintf('Running OLSTEC with multiple lambdas...\n');
    lambda_list = [0.70, 0.80, 0.90, 0.99];
    
    sub_infos_olstec_multi = cell(length(lambda_list), 1);
    elapsed_time_olstec_multi = zeros(length(lambda_list), 1);
    
    for i = 1:length(lambda_list)
        current_lambda = lambda_list(i);
        fprintf('  -> Testing OLSTEC with lambda = %.2f\n', current_lambda);
        
        clear options;
        options.maxepochs       = maxepochs;
        options.tolcost         = tolcost;
        options.permute_on      = permute_on_flag;    
        options.lambda          = current_lambda; 
        options.mu              = 0.01; 
        options.tw_flag         = 0;
        options.tw_len          = 10;
        options.store_subinfo   = true;     
        options.store_matrix    = store_matrix_flag; 
        options.verbose         = 0; 
        tic;
        [~, ~, sub_infos_olstec_multi{i}] = olstec(Tensor_Y_Noisy, OmegaTensor, [], tensor_dims, rank_r, Xinit, options);
        elapsed_time_olstec_multi(i) = toc;
    end
    
    target_lambda = 0.99;
    idx_display = find(abs(lambda_list - target_lambda) < 1e-5, 1);
    if isempty(idx_display), idx_display = length(lambda_list); end
    
    sub_infos_olstec = sub_infos_olstec_multi{idx_display}; 
    elapsed_time_olstec = elapsed_time_olstec_multi(idx_display);
end

%% (7) RSI-OLSTEC
if run_rsi_olstec
    fprintf('Running RSI-OLSTEC...\n');
    clear options;
    options                = struct();
    options.maxepochs      = maxepochs;
    options.lambda_max     = 0.80;
    options.lambda_min     = 0.70;             
    options.huber_delta    = 0.05;
    options.min_grad_threshold = 0.70; 
    options.mu             = 0.01;
    options.tolcost        = tolcost; 
    options.permute_on     = permute_on_flag;
    options.verbose        = verbose;
    options.store_matrix   = store_matrix_flag;
    options.store_subinfo  = true;
    tic
    [Xsol_rsi, infos_rsi, sub_infos_rsi] = rsi_olstec(Tensor_Y_Noisy, OmegaTensor, [], size(Tensor_Y_Noisy), rank_r, Xinit, options, aux_width);
    elapsed_time_rsi = toc;
end

%% 4.5 True NRE Calculation based on Clean Ground Truth
% -------------------------------------------------------------------------
fprintf('Calculating true error based on clean Ground Truth...\n');
true_err_petrels = zeros(1, total_slices);
true_err_grasta  = zeros(1, total_slices);
true_err_grouse  = zeros(1, total_slices);
true_err_tecpsgd = zeros(1, total_slices);
true_err_rsi     = zeros(1, total_slices);
if run_olstec
    true_err_olstec_multi = zeros(length(lambda_list), total_slices);
end

for t = 1:total_slices
    gt_frame = Tensor_Y_Clean(:, :, t); 
    norm_gt = norm(gt_frame(:)) + 1e-10; 

    if run_petrels && exist('sub_infos_petrels','var') && isfield(sub_infos_petrels, 'L')
        L_frame = reshape(sub_infos_petrels.L(:, t), [rows, cols]);
        true_err_petrels(t) = norm(gt_frame(:) - L_frame(:)) / norm_gt;
    end
    
    if run_grasta && exist('sub_infos_grasta','var') && isfield(sub_infos_grasta, 'L')
        L_frame = reshape(sub_infos_grasta.L(:, t), [rows, cols]);
        true_err_grasta(t) = norm(gt_frame(:) - L_frame(:)) / norm_gt;
    end
    
    if run_grouse && exist('sub_infos_grouse','var') && isfield(sub_infos_grouse, 'L')
        L_frame = reshape(sub_infos_grouse.L(:, t), [rows, cols]);
        true_err_grouse(t) = norm(gt_frame(:) - L_frame(:)) / norm_gt;
    end
    
    if run_tecpsgd && exist('sub_infos_TeCPSGD','var') && isfield(sub_infos_TeCPSGD, 'L')
        if ndims(sub_infos_TeCPSGD.L) == 3
            L_frame = sub_infos_TeCPSGD.L(:, :, t);
        else
            L_frame = reshape(sub_infos_TeCPSGD.L(:, t), [rows, cols]);
        end
        true_err_tecpsgd(t) = norm(gt_frame(:) - L_frame(:)) / norm_gt;
    end
    
    if run_olstec && exist('sub_infos_olstec_multi','var')
        for k = 1:length(lambda_list)
            if isfield(sub_infos_olstec_multi{k}, 'L')
                L_frame = reshape(sub_infos_olstec_multi{k}.L(:, t), [rows, cols]);
                true_err_olstec_multi(k, t) = norm(gt_frame(:) - L_frame(:)) / norm_gt;
            end
        end
    end
    
    if run_rsi_olstec && exist('sub_infos_rsi','var') && isfield(sub_infos_rsi, 'L')
        L_frame = reshape(sub_infos_rsi.L(:, t), [rows, cols]);
        true_err_rsi(t) = norm(gt_frame(:) - L_frame(:)) / norm_gt;
    end
end

run_avg_petrels = cumsum(true_err_petrels) ./ (1:total_slices);
run_avg_grasta  = cumsum(true_err_grasta)  ./ (1:total_slices);
run_avg_grouse  = cumsum(true_err_grouse)  ./ (1:total_slices);
run_avg_tecpsgd = cumsum(true_err_tecpsgd) ./ (1:total_slices);
run_avg_rsi     = cumsum(true_err_rsi)     ./ (1:total_slices);

if run_olstec
    run_avg_olstec_multi = cumsum(true_err_olstec_multi, 2) ./ (1:total_slices);
end

if run_cpwopt && exist('Xsol_cp_wopt','var')
    fprintf('Parsing CP-WOPT offline reconstructed tensor...\n');
    if isa(Xsol_cp_wopt, 'ktensor')
        L_cp_full = double(Xsol_cp_wopt);
    elseif isstruct(Xsol_cp_wopt) && isfield(Xsol_cp_wopt, 'A') && isfield(Xsol_cp_wopt, 'B') && isfield(Xsol_cp_wopt, 'C')
        L_cp_full = zeros(rows, cols, total_slices);
        for f = 1:total_slices
            L_cp_full(:,:,f) = Xsol_cp_wopt.A * diag(Xsol_cp_wopt.C(f,:)) * Xsol_cp_wopt.B';
        end
    elseif iscell(Xsol_cp_wopt) && length(Xsol_cp_wopt) >= 3
        L_cp_full = zeros(rows, cols, total_slices);
        for f = 1:total_slices
            L_cp_full(:,:,f) = Xsol_cp_wopt{1} * diag(Xsol_cp_wopt{3}(f,:)) * Xsol_cp_wopt{2}';
        end
    elseif isnumeric(Xsol_cp_wopt) && ndims(Xsol_cp_wopt) == 3
        L_cp_full = Xsol_cp_wopt;
    else
        warning('Unknown CP-WOPT output structure. Skipping computation.');
        L_cp_full = zeros(rows, cols, total_slices);
    end
    final_cp_nre = norm(Tensor_Y_Clean(:) - L_cp_full(:)) / (norm(Tensor_Y_Clean(:)) + 1e-10);
end

%% 5. Plotting & Comparison
% -------------------------------------------------------------------------
fs = 14;
x_axis = 1:total_slices;
half_idx = max(1, floor(total_slices / 2)); 

% --- Figure 1: Normalized Residual Error ---
h1 = figure('Name', 'Residual Error Comparison'); 
hold on; grid on; box on;
p1_handles = gobjects(0); 
leg_str = {};

if run_cpwopt && exist('final_cp_nre','var')
    yline(final_cp_nre, '--k', 'CP-WOPT (Batch)', 'linewidth', 2.0, 'LabelHorizontalAlignment', 'left', 'HandleVisibility', 'off'); 
end

if run_grouse && exist('sub_infos_grouse','var')
    h = semilogy(x_axis, true_err_grouse, '-g', 'linewidth', 2.0); 
    p1_handles(end+1) = h; leg_str{end+1} = 'Grouse'; 
end

valid_errs_fig1 = [];
valid_min_fig1 = inf;

if run_grasta && exist('sub_infos_grasta','var')
    h = semilogy(x_axis, true_err_grasta, '-y', 'linewidth', 2.0); 
    p1_handles(end+1) = h; leg_str{end+1} = 'Grasta'; 
    valid_errs_fig1 = [valid_errs_fig1; true_err_grasta(half_idx:end)'];
    valid_min_fig1 = min(valid_min_fig1, min(true_err_grasta(true_err_grasta>0)));
end

if run_petrels && exist('sub_infos_petrels','var')
    h = semilogy(x_axis, true_err_petrels, '-m', 'linewidth', 2.0); 
    p1_handles(end+1) = h; leg_str{end+1} = 'Petrels'; 
    valid_errs_fig1 = [valid_errs_fig1; true_err_petrels(half_idx:end)'];
    valid_min_fig1 = min(valid_min_fig1, min(true_err_petrels(true_err_petrels>0)));
end

if run_tecpsgd && exist('sub_infos_TeCPSGD','var')
    h = semilogy(x_axis, true_err_tecpsgd, '-b', 'linewidth', 2.0); 
    p1_handles(end+1) = h; leg_str{end+1} = 'TeCPSGD'; 
    valid_errs_fig1 = [valid_errs_fig1; true_err_tecpsgd(half_idx:end)'];
    valid_min_fig1 = min(valid_min_fig1, min(true_err_tecpsgd(true_err_tecpsgd>0)));
end

if run_olstec && exist('true_err_olstec_multi','var')
    olstec_linespecs = {':c', '--c', '-c', '-.c'}; 
    for i = 1:length(lambda_list)
        h = semilogy(x_axis, true_err_olstec_multi(i, :), olstec_linespecs{i}, 'linewidth', 2.0);
        p1_handles(end+1) = h; leg_str{end+1} = sprintf('OLSTEC (\\lambda=%.2f)', lambda_list(i));
    end
    valid_errs_fig1 = [valid_errs_fig1; reshape(true_err_olstec_multi(:, half_idx:end), [], 1)];
    valid_min_fig1 = min(valid_min_fig1, min(true_err_olstec_multi(true_err_olstec_multi>0)));
end

if run_rsi_olstec && exist('sub_infos_rsi','var')
    h = semilogy(x_axis, true_err_rsi, '-r', 'linewidth', 2.0); 
    p1_handles(end+1) = h; leg_str{end+1} = 'RSI-OLSTEC'; 
    valid_errs_fig1 = [valid_errs_fig1; true_err_rsi(half_idx:end)'];
    valid_min_fig1 = min(valid_min_fig1, min(true_err_rsi(true_err_rsi>0)));
end
hold off; 

legend(p1_handles, leg_str, 'location', 'best', 'FontSize', 12); 
xlabel('Data Stream Index (Frames)'); ylabel('True Normalized Residual Error');
set(gca, 'FontSize', fs);

if ~isempty(valid_errs_fig1) && valid_min_fig1 > 0
    y_ceiling_fig1 = max(valid_errs_fig1(:));
    ylim([valid_min_fig1 * 0.8, y_ceiling_fig1 * 2.5]); 
end
xlim([1, total_slices]);
savefig(h1, 'Fig_Residual_Error_R3.fig');

% --- Figure 2: Running Average Error ---
h2 = figure('Name', 'Running Average Error');
hold on; grid on; box on;
p2_handles = gobjects(0); 
leg_str_avg = {};

if run_grouse && exist('sub_infos_grouse','var')
    h = semilogy(x_axis, run_avg_grouse, '-g', 'linewidth', 2.0); 
    p2_handles(end+1) = h; leg_str_avg{end+1} = 'Grouse'; 
end

valid_errs_fig2 = [];
valid_min_fig2 = inf;

if run_grasta && exist('sub_infos_grasta','var')
    h = semilogy(x_axis, run_avg_grasta, '-y', 'linewidth', 2.0); 
    p2_handles(end+1) = h; leg_str_avg{end+1} = 'Grasta'; 
    valid_errs_fig2 = [valid_errs_fig2; run_avg_grasta(half_idx:end)'];
    valid_min_fig2 = min(valid_min_fig2, min(run_avg_grasta(run_avg_grasta>0)));
end

if run_petrels && exist('sub_infos_petrels','var')
    h = semilogy(x_axis, run_avg_petrels, '-m', 'linewidth', 2.0); 
    p2_handles(end+1) = h; leg_str_avg{end+1} = 'Petrels'; 
    valid_errs_fig2 = [valid_errs_fig2; run_avg_petrels(half_idx:end)'];
    valid_min_fig2 = min(valid_min_fig2, min(run_avg_petrels(run_avg_petrels>0)));
end

if run_tecpsgd && exist('sub_infos_TeCPSGD','var')
    h = semilogy(x_axis, run_avg_tecpsgd, '-b', 'linewidth', 2.0); 
    p2_handles(end+1) = h; leg_str_avg{end+1} = 'TeCPSGD'; 
    valid_errs_fig2 = [valid_errs_fig2; run_avg_tecpsgd(half_idx:end)'];
    valid_min_fig2 = min(valid_min_fig2, min(run_avg_tecpsgd(run_avg_tecpsgd>0)));
end

if run_olstec && exist('run_avg_olstec_multi','var')
    for i = 1:length(lambda_list)
        h = semilogy(x_axis, run_avg_olstec_multi(i, :), olstec_linespecs{i}, 'linewidth', 2.0);
        p2_handles(end+1) = h; leg_str_avg{end+1} = sprintf('OLSTEC (\\lambda=%.2f)', lambda_list(i));
    end
    valid_errs_fig2 = [valid_errs_fig2; reshape(run_avg_olstec_multi(:, half_idx:end), [], 1)];
    valid_min_fig2 = min(valid_min_fig2, min(run_avg_olstec_multi(run_avg_olstec_multi>0)));
end

if run_rsi_olstec && exist('sub_infos_rsi','var')
    h = semilogy(x_axis, run_avg_rsi, '-r', 'linewidth', 2.0); 
    p2_handles(end+1) = h; leg_str_avg{end+1} = 'RSI-OLSTEC'; 
    valid_errs_fig2 = [valid_errs_fig2; run_avg_rsi(half_idx:end)'];
    valid_min_fig2 = min(valid_min_fig2, min(run_avg_rsi(run_avg_rsi>0)));
end
hold off; 

legend(p2_handles, leg_str_avg, 'location', 'best', 'FontSize', 12);
xlabel('Data Stream Index (Frames)'); ylabel('True Running Average Error');
set(gca, 'FontSize', fs);

if ~isempty(valid_errs_fig2) && valid_min_fig2 > 0
    y_ceiling_fig2 = max(valid_errs_fig2(:));
    ylim([valid_min_fig2 * 0.8, y_ceiling_fig2 * 2.5]); 
end
xlim([1, total_slices]);
savefig(h2, 'Fig_Running_Average_Error_R3.fig');
fprintf('Saved figures: Fig_Residual_Error_R3.fig and Fig_Running_Average_Error_R3.fig\n');

%% 6. Quantitative Results
% -------------------------------------------------------------------------
fprintf('\n=======================================================================\n');
fprintf('QUANTITATIVE RESULTS COMPARISON (True Error vs. Clean Ground Truth)\n');
fprintf('=======================================================================\n');
fprintf('%-18s | %-20s | %-20s\n', 'Algorithm', 'Final True Resid Err', 'Final True Run. Avg');
fprintf('-----------------------------------------------------------------------\n');

if run_cpwopt && exist('final_cp_nre', 'var'), fprintf('%-18s | %-20.6e | %-20s\n', 'CP-WOPT', final_cp_nre, 'N/A (Batch)'); end
if run_tecpsgd && exist('sub_infos_TeCPSGD', 'var'), fprintf('%-18s | %-20.6e | %-20.6e\n', 'TeCPSGD', true_err_tecpsgd(end), run_avg_tecpsgd(end)); end
if run_petrels && exist('sub_infos_petrels', 'var'), fprintf('%-18s | %-20.6e | %-20.6e\n', 'Petrels', true_err_petrels(end), run_avg_petrels(end)); end
if run_grouse && exist('sub_infos_grouse', 'var'), fprintf('%-18s | %-20.6e | %-20.6e\n', 'Grouse', true_err_grouse(end), run_avg_grouse(end)); end
if run_grasta && exist('sub_infos_grasta', 'var'), fprintf('%-18s | %-20.6e | %-20.6e\n', 'Grasta', true_err_grasta(end), run_avg_grasta(end)); end

if run_olstec && exist('true_err_olstec_multi', 'var')
    for i = 1:length(lambda_list)
        algo_name_str = sprintf('OLSTEC (lam=%.2f)', lambda_list(i));
        fprintf('%-18s | %-20.6e | %-20.6e\n', algo_name_str, true_err_olstec_multi(i, end), run_avg_olstec_multi(i, end)); 
    end
end
if run_rsi_olstec && exist('sub_infos_rsi', 'var'), fprintf('%-18s | %-20.6e | %-20.6e\n', 'RSI-OLSTEC (Ours)', true_err_rsi(end), run_avg_rsi(end)); end
fprintf('=======================================================================\n');

%% 7. Image Display
% -------------------------------------------------------------------------
observe_percent = 100 * fraction; 
if image_display_flag
    figure('Name', 'Visual Comparison', 'Position', [100, 100, 1600, 800]);
    
    display_list = {};
    if run_petrels && exist('sub_infos_petrels', 'var'), display_list(end+1,:) = {true, sub_infos_petrels, 'Petrels'}; end
    if run_grasta && exist('sub_infos_grasta', 'var'),   display_list(end+1,:) = {true, sub_infos_grasta, 'Grasta'}; end
    if run_grouse && exist('sub_infos_grouse', 'var'),   display_list(end+1,:) = {true, sub_infos_grouse, 'Grouse'}; end
    if run_tecpsgd && exist('sub_infos_TeCPSGD', 'var'), display_list(end+1,:) = {true, sub_infos_TeCPSGD, 'TeCPSGD'}; end
    if run_olstec && exist('sub_infos_olstec', 'var'),   display_list(end+1,:) = {true, sub_infos_olstec, 'OLSTEC'}; end
    if run_rsi_olstec && exist('sub_infos_rsi', 'var'),  display_list(end+1,:) = {true, sub_infos_rsi, 'RSI-OLSTEC'}; end
    
    num_algos_to_plot = size(display_list, 1);
    plot_height = 3;
    plot_width = num_algos_to_plot; 
    
    if num_algos_to_plot > 0
        for i = 1:total_slices
            for k = 1:num_algos_to_plot
                current_info = display_list{k, 2};
                current_name = display_list{k, 3};
                
                display_images_local(rows, cols, observe_percent, plot_height, plot_width, k, i, current_info, current_name);
            end
            pause(0.01);
        end
    else
        fprintf('No algorithms selected for visualization.\n');
    end
end    

%% 8. Spatio-Temporal Comparison (Paper Figure)
% -------------------------------------------------------------------------
target_frames = [50, 200, 400, 600];
target_frames = target_frames(target_frames <= total_slices);
num_cols = length(target_frames);

if num_cols == 0
    warning('No frames available for display. Please check target_frames settings.');
else
    fprintf('Generating multi-frame comparison matrix (Frames: %s)...\n', num2str(target_frames));
    
    algo_plot_list = {};
    
    if run_grouse && exist('sub_infos_grouse','var'),     algo_plot_list(end+1,:) = {sub_infos_grouse, 'Grouse'}; end
    if run_tecpsgd && exist('sub_infos_TeCPSGD','var'),   algo_plot_list(end+1,:) = {sub_infos_TeCPSGD, 'TeCPSGD'}; end
    if run_grasta && exist('sub_infos_grasta','var'),     algo_plot_list(end+1,:) = {sub_infos_grasta, 'Grasta'}; end
    if run_petrels && exist('sub_infos_petrels','var'),   algo_plot_list(end+1,:) = {sub_infos_petrels, 'Petrels'}; end
    
    if run_olstec && exist('sub_infos_olstec','var')
        olstec_label = sprintf('OLSTEC (\\lambda=%.2f)', lambda_list(idx_display)); 
        algo_plot_list(end+1,:) = {sub_infos_olstec, olstec_label}; 
    end
    
    if run_rsi_olstec && exist('sub_infos_rsi','var'),    algo_plot_list(end+1,:) = {sub_infos_rsi, 'RSI-OLSTEC'}; end
    
    num_algos = size(algo_plot_list, 1);
    num_rows = num_algos + 1; 
    
    fig_height = 200 * num_rows;
    fig_width  = 200 * num_cols;
    h_matrix = figure('Name', 'Multi-Frame Comparison', 'Position', [50, 50, fig_width, fig_height]);
    
    use_tiled = exist('tiledlayout', 'builtin'); 
    if use_tiled
        t = tiledlayout(num_rows, num_cols, 'TileSpacing', 'none', 'Padding', 'compact');
    end

    for j = 1:num_cols
        frame_idx = target_frames(j);
        
        if use_tiled
            nexttile;
        else
            subplot(num_rows, num_cols, j);
        end
        
        imagesc(Tensor_Y_Noisy(:, :, frame_idx));
        colormap(gray); axis image; axis off;
        
        title(['Frame ' num2str(frame_idx)], 'FontSize', 12, 'FontWeight', 'bold');
        
        if j == 1
            hY = ylabel('Original', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
            set(hY, 'Visible', 'on'); 
        end
    end
    
    for i = 1:num_algos
        algo_data = algo_plot_list{i, 1};
        algo_name = algo_plot_list{i, 2};
        
        is_tensor_struct = false;
        if isfield(algo_data, 'L') && ndims(algo_data.L) == 3
            is_tensor_struct = true;
        elseif isfield(algo_data, 'Xsol') && ndims(algo_data.Xsol) == 3
             is_tensor_struct = true;
        end
        
        for j = 1:num_cols
            frame_idx = target_frames(j);
            
            if use_tiled
                nexttile;
            else
                curr_row_idx = i + 1; 
                subplot_idx = (curr_row_idx - 1) * num_cols + j;
                subplot(num_rows, num_cols, subplot_idx);
            end
            
            L_frame = zeros(rows, cols); 
            
            if isfield(algo_data, 'L') && ~isempty(algo_data.L)
                if ndims(algo_data.L) == 3
                    L_frame = algo_data.L(:, :, frame_idx);
                else
                    L_frame = reshape(algo_data.L(:, frame_idx), [rows, cols]);
                end
            elseif isfield(algo_data, 'U') && isfield(algo_data, 'V')
                u_vec = algo_data.U(:, frame_idx);
                v_vec = algo_data.V(:, frame_idx);
                L_frame = reshape(u_vec * v_vec', [rows, cols]); 
            else
                warning('Algorithm %s has no background matrix L saved in sub_infos. Skipping frame rendering.', algo_name);
            end
            
            imagesc(L_frame);
            colormap(gray); axis image; axis off;
            
            if j == 1
                hY = ylabel(algo_name, 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');
                set(hY, 'Visible', 'on');
            end
        end
    end
    
    savefig(h_matrix, 'Fig_MultiFrame_Matrix.fig');
    fprintf('Saved multi-frame matrix comparison plot: Fig_MultiFrame_Matrix.fig\n');
end

%% Local Functions
% -------------------------------------------------------------------------
function display_images_local(rows, cols, observe, height, width, test_idx, frame, sub_infos, algorithm)
    subplot(height, width, test_idx);
    if isfield(sub_infos, 'I') && size(sub_infos.I, 2) >= frame
        imagesc(reshape(sub_infos.I(:,frame),[rows cols]));
    else
        imagesc(reshape(sub_infos.L(:,frame),[rows cols])); 
    end
    colormap(gray); axis image; axis off;
    title([algorithm, ': ', num2str(observe), '% obs'], 'Interpreter', 'none');

    subplot(height, width, width + test_idx);
    imagesc(reshape(sub_infos.L(:,frame),[rows cols]));
    colormap(gray); axis image; axis off;
    title(['Low-rank: f = ', num2str(frame)]);

    subplot(height, width, 2*width + test_idx);
    if isfield(sub_infos, 'S')
        img_E = sub_infos.S(:,frame);
    elseif isfield(sub_infos, 'E')
        img_E = sub_infos.E(:,frame);
    else
        img_E = zeros(rows*cols, 1);
    end
    imagesc(reshape(img_E,[rows cols]));
    colormap(gray); axis image; axis off;
    
    curr_err = 0;
    if isfield(sub_infos, 'err_residual') && length(sub_infos.err_residual) >= frame
        curr_err = sub_infos.err_residual(frame);
    end
    title(['Resid: err = ', sprintf('%.4f', curr_err)]);    
end