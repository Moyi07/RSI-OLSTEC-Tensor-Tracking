%% Experiment R1
% =========================================================================
% Objective: 
%   Evaluate the convergence performance and steady-state error of 
%   RSI-OLSTEC against baseline methods on real-world WAAM video data.
%
% Data Source:
%   - Video: WAAM process monitoring video (video_1.mp4).
%   - Metadata: Synchronized welding parameters (Width_mm).
%
% Key Features:
%   - Real-world data processing (Resize, Gray-scale, Normalization).
%   - Monte Carlo simulation (N=100) with random missing masks.
%   - Causal alignment of side information.
%   - Publication-quality visualization (Error Bands + Boxplots).
% =========================================================================
clear; clc; close all;

%% 1. Configuration
% -------------------------------------------------------------------------
fprintf('Starting Experiment R1: Real-World Benchmark...\n');

% Data Paths
video_filename = 'RSI_OLSTEC\dataset\video\250312-110206-video_1.mp4'; 
meta_filename  = 'RSI_OLSTEC\dataset\WAMVID_metadata.csv';

% Experiment Parameters
num_trials      = 100;   % Number of Monte Carlo trials
max_frames      = 623;   % Truncate video length for efficiency
scale_ratio     = 0.2;   % Downsampling ratio (Speed/Accuracy trade-off)
rank_r          = 20;    % CP-Rank for tensor methods
fraction        = 0.1;   % Observation ratio (10% pixels observed)
tolcost         = 1e-8;  % Convergence tolerance

% Algorithm Switches
run_cpwopt      = true;
run_petrels     = true;
run_grasta      = true;
run_grouse      = true;
run_tecpsgd     = true;
run_olstec      = true;
run_rsi_olstec  = true;

%% 2. Data Loading & Preprocessing
% -------------------------------------------------------------------------
fprintf('=== [Phase 1] Data Preprocessing ===\n');

% Check File Existence
if ~exist(video_filename, 'file')
    error('Video file not found: %s', video_filename);
end

% A. Load Video Tensor
v = VideoReader(video_filename);
Tensor_Y_GT = []; 
frame_idx = 0;
fprintf('Reading video frames...\n');

while hasFrame(v) && frame_idx < max_frames
    frame_idx = frame_idx + 1;
    raw_frame = readFrame(v);
    
    % Convert to grayscale and resize
    if size(raw_frame, 3) == 3
        gray_frame = rgb2gray(raw_frame);
    else
        gray_frame = raw_frame;
    end
    img_resized = imresize(gray_frame, scale_ratio);
    
    % Initialize storage on first frame
    if isempty(Tensor_Y_GT)
        Tensor_Y_GT = zeros(size(img_resized,1), size(img_resized,2), max_frames);
    end
    
    % Normalize to [0,1]
    Tensor_Y_GT(:, :, frame_idx) = im2double(img_resized);
end

% Trim unused pre-allocated frames
if frame_idx < max_frames
    Tensor_Y_GT = Tensor_Y_GT(:,:,1:frame_idx);
end
[rows, cols, total_slices] = size(Tensor_Y_GT);
tensor_dims = [rows, cols, total_slices];
fprintf('Data Loaded: %d x %d x %d (Frames: %d)\n', rows, cols, total_slices, total_slices);

% B. Load Auxiliary Signal (Side Information)
fprintf('Loading Metadata...\n');
if exist(meta_filename, 'file')
    try
        meta_table = readtable(meta_filename);
        
        % Parse 'Width_mm' column by matching filename
        [~, vid_name, vid_ext] = fileparts(video_filename);
        target_vid_name = [vid_name, vid_ext]; 
        
        row_idx = find(contains(meta_table.Video_filepath, target_vid_name), 1);
        
        if isempty(row_idx)
            error('Error: Metadata not found for video: %s', target_vid_name);
        else
            fprintf('Monte Carlo Validation: Matched CSV row %d\n', row_idx);
        end
        
        width_str = meta_table.Width_mm{row_idx};
        width_str = strrep(strrep(strrep(width_str, '[', ''), ']', ''), 'nan', 'NaN');
        width_vals = str2num(width_str); %#ok<ST2NM>
        
        % Align length
        if length(width_vals) > total_slices
            aux_width = width_vals(1:total_slices)';
        else
            aux_width = [width_vals'; repmat(width_vals(end), total_slices-length(width_vals), 1)];
        end
        
        % Causal Filling: Avoid future information leakage
        aux_width = fillmissing(aux_width, 'previous'); 
        fprintf('Metadata loaded successfully.\n');
    catch
        warning('Metadata parsing failed. Fallback to random aux data.');
        aux_width = rand(total_slices, 1);
    end
else
    warning('Metadata file not found. Fallback to random aux data.');
    aux_width = rand(total_slices, 1);
end

% Calculate matrix rank
numr = rows * cols;
numc = total_slices;  
num_params_of_tensor = rank_r * sum(tensor_dims);
matrix_rank = floor( num_params_of_tensor/ (numr+numc) );
if matrix_rank < 1, matrix_rank = 1; end

%% 3. Monte Carlo Loop
% -------------------------------------------------------------------------
fprintf('\n=== [Phase 2] Monte Carlo Simulation (%d Trials) ===\n', num_trials);

% OLSTEC Lambda grid
lambda_list = [0.70, 0.80, 0.90, 0.99];

% Storage Initialization
stats = struct();
alg_list = {}; 
if run_cpwopt,  alg_list{end+1} = 'CP_WOPT'; end
if run_petrels, alg_list{end+1} = 'Petrels'; end
if run_grasta,  alg_list{end+1} = 'Grasta'; end
if run_grouse,  alg_list{end+1} = 'Grouse'; end
if run_tecpsgd, alg_list{end+1} = 'TeCPSGD'; end

% Generate OLSTEC variants
if run_olstec
    for lam = lambda_list
        alg_list{end+1} = sprintf('OLSTEC_%02d', round(lam*100));
    end
end
if run_rsi_olstec, alg_list{end+1} = 'RSI_OLSTEC'; end

for i = 1:length(alg_list)
    stats.(alg_list{i}) = zeros(num_trials, total_slices); 
end

% Padding functions to handle early convergence length mismatch
pad_err = @(err, L) [reshape(err(1:min(end, L)), 1, []), repmat(err(end), 1, max(0, L - length(err)))];
clean_err = @(err, L) err( (length(err) > L && err(1) == 0) + 1 : end );

total_start_time = tic;

for trial = 1:num_trials
    iter_timer = tic;
    fprintf('Processing Trial %d / %d... ', trial, num_trials);
    
    % 3.1. Generate Random Missing Mask
    rng(trial + 1000); % Offset seed for variance
    OmegaTensor = rand(rows, cols, total_slices) < fraction;
    
    % Reshape for Matrix Algorithms
    OmegaMatrix = reshape(OmegaTensor, [numr, numc]);
    Matrix_Y_GT = reshape(Tensor_Y_GT, [numr, numc]);
    
    % 3.2. Random Initialization
    scale = 0.1; 
    Xinit.A = randn(rows, rank_r) * scale;
    Xinit.B = randn(cols, rank_r) * scale;    
    Xinit.C = randn(total_slices, rank_r) * scale;
    
    % --- Execute Algorithms ---
    % 0. CP-WOPT (Batch Tensor Baseline)
    if run_cpwopt
        opts = struct('maxepochs', 30, 'display_iters', 1, 'verbose', 0, ...
                      'tolcost', tolcost, 'store_matrix', false, 'store_subinfo', true);
        [~, ~, sub_info] = cp_wopt_mod(Tensor_Y_GT, OmegaTensor, [], tensor_dims, rank_r, Xinit, opts);
        
        raw_err = clean_err(sub_info.err_residual, total_slices);
        stats.CP_WOPT(trial, :) = pad_err(raw_err, total_slices);
    end
    
    % 1. Petrels (Matrix Baseline)
    if run_petrels
        opts = struct('maxepochs', 1, 'rank', matrix_rank, 'lambda', 0.98, 'verbose', 0, ...
                      'tolcost', tolcost, 'permute_on', false, 'store_matrix', false, 'store_subinfo', true);
        [~, ~, sub_info, ~] = petrels_mod([], Matrix_Y_GT, OmegaMatrix, [], numr, numc, opts);
        raw_err = clean_err(sub_info.err_residual, total_slices);
        stats.Petrels(trial, :) = pad_err(raw_err, total_slices);
    end
    
    % 2. GRASTA (Robust Matrix Baseline)
    if run_grasta
        opts = struct('maxepochs', 1, 'RANK', matrix_rank, 'rho', 1.8, 'ITER_MAX', 20, ...
                      'MAX_MU', 10000, 'MIN_MU', 1, 'DIM_M', numr, 'USE_MEX', 0, ...
                      'verbose', 0, 'tolcost', tolcost, 'permute_on', false, 'store_matrix', false, 'store_subinfo', true);
        [~, ~, sub_info, ~] = grasta_mod([], Matrix_Y_GT, OmegaMatrix, [], numr, numc, opts);
        raw_err = clean_err(sub_info.err_residual, total_slices);
        stats.Grasta(trial, :) = pad_err(raw_err, total_slices);
    end
    
    % 3. Grouse (SGD Matrix Baseline)
    if run_grouse
        opts = struct('maxepochs', 1, 'maxrank', matrix_rank, 'step_size', 0.0001, ...
                      'verbose', 0, 'tolcost', tolcost, 'permute_on', false, 'store_matrix', false, 'store_subinfo', true);
        [~, ~, sub_info, ~] = grouse_mod([], Matrix_Y_GT, OmegaMatrix, [], numr, numc, opts);
        raw_err = clean_err(sub_info.err_residual, total_slices);
        stats.Grouse(trial, :) = pad_err(raw_err, total_slices);
    end
    
    % 4. TeCPSGD (Tensor SGD Baseline)
    if run_tecpsgd
        opts = struct('maxepochs', 1, 'lambda', 0.99, 'stepsize', 0.1, 'mu', 0.01, ...
                      'verbose', 0, 'tolcost', tolcost, 'permute_on', false, 'store_matrix', false, 'store_subinfo', true);
        [~, ~, sub_info] = TeCPSGD(Tensor_Y_GT, OmegaTensor, [], tensor_dims, rank_r, Xinit, opts);
        raw_err = clean_err(sub_info.err_residual, total_slices);
        stats.TeCPSGD(trial, :) = pad_err(raw_err, total_slices);
    end
    
    % 5. OLSTEC (Standard Tensor Baseline)
    if run_olstec
        for k = 1:length(lambda_list)
            lam = lambda_list(k);
            alg_name = sprintf('OLSTEC_%02d', round(lam*100));
            
            opts = struct('maxepochs', 1, 'lambda', lam, 'mu', 0.01, 'verbose', 0, ...
                          'tw_flag', 0, 'tw_len', 10, 'tolcost', tolcost, 'permute_on', false, 'store_matrix', false, 'store_subinfo', true);
            [~, ~, sub_info] = olstec(Tensor_Y_GT, OmegaTensor, [], tensor_dims, rank_r, Xinit, opts);
            raw_err = clean_err(sub_info.err_residual, total_slices);
            stats.(alg_name)(trial, :) = pad_err(raw_err, total_slices);
        end
    end
    
    % 6. RSI-OLSTEC (Proposed)
    if run_rsi_olstec
        opts = struct('maxepochs', 1, 'lambda_max', 0.80, 'lambda_min', 0.70, ...
                      'huber_delta', 0.30, 'min_grad_threshold', 0.70, 'mu', 0.01, ...
                      'verbose', 0, 'tolcost', tolcost, 'permute_on', false, 'store_matrix', false, 'store_subinfo', true);
        [~, ~, sub_info] = rsi_olstec(Tensor_Y_GT, OmegaTensor, [], tensor_dims, rank_r, Xinit, opts, aux_width);
        raw_err = clean_err(sub_info.err_residual, total_slices);
        stats.RSI_OLSTEC(trial, :) = pad_err(raw_err, total_slices);
    end
    
    fprintf('Done (%.1fs)\n', toc(iter_timer));
end
fprintf('\nSimulation Completed in %.1f minutes.\n', toc(total_start_time)/60);

%% 4. Statistical Analysis & Visualization
% -------------------------------------------------------------------------
fprintf('=== [Phase 3] Generating Plots ===\n');

% Color Palette & Line Styles
cols_map = containers.Map();
line_styles = containers.Map();

if run_cpwopt,  cols_map('CP_WOPT') = [0 0 0]; line_styles('CP_WOPT') = '-'; end
if run_petrels, cols_map('Petrels') = [0.6 0.6 0.6]; line_styles('Petrels') = '-'; end
if run_grasta,  cols_map('Grasta')  = [0.929 0.694 0.125]; line_styles('Grasta') = '-'; end
if run_grouse,  cols_map('Grouse')  = [0.466 0.674 0.188]; line_styles('Grouse') = '-'; end
if run_tecpsgd, cols_map('TeCPSGD') = [0.000 0.447 0.741]; line_styles('TeCPSGD') = '-'; end
if run_rsi_olstec, cols_map('RSI_OLSTEC') = [0.850 0.325 0.098]; line_styles('RSI_OLSTEC') = '-'; end

% Assign Cyan color and different line styles for OLSTEC variants
if run_olstec
    olstec_styles = {':', '--', '-', '-.'};
    for k = 1:length(lambda_list)
        alg_name = sprintf('OLSTEC_%02d', round(lambda_list(k)*100));
        cols_map(alg_name) = [0.301 0.745 0.933]; 
        line_styles(alg_name) = olstec_styles{k};
    end
end

% Font Settings
linewidth_normal = 1.5;
font_name = 'Times New Roman'; 
font_size = 14;                

% --- Figure 1: Convergence Time-Series ---
fig1 = figure('Name', 'Exp R1: Real-World Benchmark', 'Position', [100, 100, 800, 550], 'Color', 'w');
hold on; grid on; box on;
ax = gca;
ax.FontName = font_name; ax.FontSize = font_size;
ax.LineWidth = 1.2; ax.TickLabelInterpreter = 'latex';
set(gca, 'YScale', 'log'); 

x_axis = 1:total_slices;
legend_handles = [];
legend_names = {};

% Sort algorithms to put RSI at the end
alg_list_sorted = alg_list(~strcmp(alg_list, 'RSI_OLSTEC'));
if isfield(stats, 'RSI_OLSTEC'), alg_list_sorted{end+1} = 'RSI_OLSTEC'; end

global_min = inf; 
global_max = -inf;

for i = 1:length(alg_list_sorted)
    name = alg_list_sorted{i};
    data = stats.(name);
    
    mu = mean(data, 1);
    sigma = std(data, 0, 1);
    conf_interval = 1.96 * sigma / sqrt(num_trials); 
    
    col = cols_map(name);
    l_style = line_styles(name);
    
    % Calculate global minimum based on valid mean values
    valid_mu = mu(mu > 0);
    if ~isempty(valid_mu)
        global_min = min(global_min, min(valid_mu));
    end
    
    % Calculate global maximum (excluding initial transient errors)
    half_idx = max(1, floor(total_slices / 2));
    if ~strcmp(name, 'Grouse')
        global_max = max(global_max, max(mu(half_idx:end) + conf_interval(half_idx:end)));
    end
    
    % Ensure lower bound is strictly positive for log-scale plotting
    lower_bound = mu - conf_interval;
    lower_bound(lower_bound <= 0) = mu(lower_bound <= 0) * 0.1;
    
    % Plot Error Band
    fill([x_axis, fliplr(x_axis)], [mu+conf_interval, fliplr(lower_bound)],...
         col, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
     
    % Plot Mean Line
    h = plot(x_axis, mu, 'Color', col, 'LineWidth', linewidth_normal, 'LineStyle', l_style);
    legend_handles(end+1) = h; 
    
    % Legend formatting
    if startsWith(name, 'OLSTEC_')
        lam_val = str2double(name(8:end))/100;
        legend_names{end+1} = sprintf('OLSTEC ($\\lambda=%.2f$)', lam_val);
    elseif strcmp(name, 'RSI_OLSTEC')
        legend_names{end+1} = '\textbf{RSI-OLSTEC}'; 
    else
        legend_names{end+1} = strrep(name, '_', '-'); 
    end
end

xlabel('Time Index (Frames)', 'Interpreter', 'latex', 'FontSize', font_size+2);
ylabel('Normalized Residual Error (log)', 'Interpreter', 'latex', 'FontSize', font_size+2);
title('Convergence Performance', 'Interpreter', 'latex', 'FontSize', font_size+2);

lgd = legend(legend_handles, legend_names, 'Location', 'southwest');
lgd.Interpreter = 'latex'; lgd.FontSize = font_size - 2; lgd.Box = 'off';

% Dynamic Y-axis and strict X-axis limits
if global_min > 0 && global_max > global_min
    ylim([global_min * 0.5, global_max * 1.5]); 
end
xlim([1, total_slices]);

% --- Figure 1: Inset Plot (Time Series) ---
ax1_pos = get(ax, 'Position'); 
inset_ax = axes('Position', [ax1_pos(1) + ax1_pos(3)*0.55, ax1_pos(2) + ax1_pos(4)*0.50, ax1_pos(3)*0.40, ax1_pos(4)*0.40]); 
box on; grid on; hold on;
set(inset_ax, 'YScale', 'log', 'FontName', font_name, 'FontSize', 10, 'TickLabelInterpreter', 'latex');

zoom_start = max(1, total_slices - 100);
zoom_end = total_slices;
zoom_x = zoom_start:zoom_end;

% Select algorithms for inset plot
zoom_algs = {};
if run_petrels, zoom_algs{end+1} = 'Petrels'; end
if run_olstec
    for k = 1:length(lambda_list)
        zoom_algs{end+1} = sprintf('OLSTEC_%02d', round(lambda_list(k)*100));
    end
end
if run_rsi_olstec, zoom_algs{end+1} = 'RSI_OLSTEC'; end

zoom_min = inf;
zoom_max = -inf;

for i = 1:length(zoom_algs)
    name = zoom_algs{i};
    if isfield(stats, name)
        mu = mean(stats.(name)(:, zoom_x), 1);
        plot(inset_ax, zoom_x, mu, 'Color', cols_map(name), 'LineWidth', 1.5, 'LineStyle', line_styles(name));
        zoom_min = min(zoom_min, min(mu));
        zoom_max = max(zoom_max, max(mu));
    end
end
xlim(inset_ax, [zoom_start, zoom_end]);
if zoom_min > 0 && zoom_max > zoom_min
    ylim(inset_ax, [zoom_min * 0.9, zoom_max * 1.1]);
end
title(inset_ax, 'Zoom-in (Steady State)', 'Interpreter', 'latex', 'FontSize', 11);

% =========================================================================
% --- Figure 2: Final Accuracy Distribution (Boxplot) ---
% =========================================================================
fig2 = figure('Name', 'Exp R1: Final Accuracy Boxplot', 'Position', [150, 150, 800, 550], 'Color', 'w');
hold on; grid on; box on;

final_errors = [];
final_errors_no_grouse = []; 
group_labels = {};
colors_for_boxplot = [];

for i = 1:length(alg_list_sorted)
    name = alg_list_sorted{i};
    % Average final 50 frames for convergence error
    final_vals = mean(stats.(name)(:, end-49:end), 2); 
    
    final_errors = [final_errors, final_vals];
    if ~strcmp(name, 'Grouse')
        final_errors_no_grouse = [final_errors_no_grouse; final_vals];
    end
    
    colors_for_boxplot = [colors_for_boxplot; cols_map(name)];
    
    if startsWith(name, 'OLSTEC_')
        lam_val = str2double(name(8:end))/100;
        group_labels{end+1} = sprintf('OLSTEC ($\\lambda=%.2f$)', lam_val);
    elseif strcmp(name, 'RSI_OLSTEC')
        group_labels{end+1} = '\textbf{RSI-OLSTEC}';
    else
        group_labels{end+1} = strrep(name, '_', '-');
    end
end

% Plot Boxplot
hBox = boxplot(final_errors, 'Labels', group_labels, 'Symbol', 'o', 'Widths', 0.6);
set(gca, 'TickLabelInterpreter', 'tex');

% Style Outliers and Boxes
hOutliers = findobj(gca,'Tag','Outliers');
set(hOutliers, 'MarkerEdgeColor', [0.5 0.5 0.5], 'MarkerSize', 4);
set(hBox, {'LineWidth'}, {1.2});

boxes = findobj(gca, 'Tag', 'Box');
if length(boxes) == length(alg_list_sorted)
    for j = 1:length(boxes)
        idx = length(boxes) - j + 1;
        patch(get(boxes(j),'XData'), get(boxes(j),'YData'), colors_for_boxplot(idx,:), ...
            'FaceAlpha', 0.5, 'EdgeColor', colors_for_boxplot(idx,:), 'LineWidth', 1.5);
    end
end

% Axis formatting
ax2 = gca;
ax2.FontName = font_name; ax2.FontSize = font_size;
ax2.LineWidth = 1.2; ax2.TickLabelInterpreter = 'latex';
xlabel('Algorithm', 'Interpreter', 'latex', 'FontSize', font_size+2);
ylabel('Final Residual Error (log)', 'Interpreter', 'latex', 'FontSize', font_size+2);
title('Statistical Accuracy Distribution', 'Interpreter', 'latex', 'FontSize', font_size+2);
xtickangle(30); 
ax2.XMinorTick = 'off'; ax2.YMinorTick = 'off';
set(gca, 'YScale', 'log'); 

% Apply truncated Y-axis range
box_min = min(final_errors_no_grouse(:));
box_max_no_grouse = max(final_errors_no_grouse(:));
if box_min > 0 && box_max_no_grouse > box_min
    ylim([box_min * 0.5, box_max_no_grouse * 1.5]); 
end

% --- Figure 2: Inset Plot (Boxplot) ---
ax2_pos = get(ax2, 'Position'); 
inset_ax2 = axes('Position', [ax2_pos(1) + ax2_pos(3)*0.55, ax2_pos(2) + ax2_pos(4)*0.55, ax2_pos(3)*0.40, ax2_pos(4)*0.30]); 
box on; grid on; hold on;
set(inset_ax2, 'YScale', 'log', 'FontName', font_name, 'FontSize', 10, 'TickLabelInterpreter', 'latex');

zoom_algs_box = {};
if run_olstec
    for k = 1:length(lambda_list)
        zoom_algs_box{end+1} = sprintf('OLSTEC_%02d', round(lambda_list(k)*100));
    end
end
if run_rsi_olstec, zoom_algs_box{end+1} = 'RSI_OLSTEC'; end

zoom_data_box = [];
zoom_labels_box = {};
zoom_colors_box = [];

for i = 1:length(zoom_algs_box)
    name = zoom_algs_box{i};
    if isfield(stats, name)
        final_vals = mean(stats.(name)(:, end-49:end), 2); 
        zoom_data_box = [zoom_data_box, final_vals];
        zoom_colors_box = [zoom_colors_box; cols_map(name)];
        
        if startsWith(name, 'OLSTEC_')
            lam_val = str2double(name(8:end))/100;
            zoom_labels_box{end+1} = sprintf('OLSTEC(%.2f)', lam_val); 
        elseif strcmp(name, 'RSI_OLSTEC')
            zoom_labels_box{end+1} = '\bf RSI-OLSTEC';
        end
    end
end

if ~isempty(zoom_data_box)
    axes(inset_ax2); 
    hBox_inset = boxplot(zoom_data_box, 'Labels', zoom_labels_box, 'Symbol', 'o', 'Widths', 0.5);
    set(inset_ax2, 'TickLabelInterpreter', 'tex');
    
    hOutliers_inset = findobj(inset_ax2, 'Tag', 'Outliers');
    set(hOutliers_inset, 'MarkerEdgeColor', [0.5 0.5 0.5], 'MarkerSize', 3);
    set(hBox_inset, {'LineWidth'}, {1.2});
    
    boxes_inset = findobj(inset_ax2, 'Tag', 'Box');
    if length(boxes_inset) == length(zoom_algs_box)
        for j = 1:length(boxes_inset)
            idx = length(boxes_inset) - j + 1;
            patch(get(boxes_inset(j),'XData'), get(boxes_inset(j),'YData'), zoom_colors_box(idx,:), ...
                'FaceAlpha', 0.5, 'EdgeColor', zoom_colors_box(idx,:), 'LineWidth', 1.5, 'Parent', inset_ax2);
        end
    end
    
    y_min_inset = min(zoom_data_box(:));
    y_max_inset = max(zoom_data_box(:));
    if y_min_inset > 0 && y_max_inset > y_min_inset
        ylim(inset_ax2, [y_min_inset * 0.95, y_max_inset * 1.05]); 
    end
    title(inset_ax2, 'OLSTEC vs Ours', 'Interpreter', 'latex', 'FontSize', 11);
end

%% 5. Quantitative Results Output
% -------------------------------------------------------------------------
fprintf('\n==========================================================================================\n');
fprintf('QUANTITATIVE RESULTS COMPARISON (Monte Carlo: %d Trials)\n', num_trials);
fprintf('==========================================================================================\n');
fprintf('%-25s | %-30s \n', 'Algorithm', 'Final Norm. Resid Err (Mean ± Std)');
fprintf('------------------------------------------------------------------------------------------\n');

for i = 1:length(alg_list_sorted)
    name = alg_list_sorted{i};
    if isfield(stats, name)
        final_vals = mean(stats.(name)(:, end-49:end), 2); 
        
        mean_err = mean(final_vals);
        std_err = std(final_vals);
        
        if startsWith(name, 'OLSTEC_')
            lam_val = str2double(name(8:end))/100;
            display_name = sprintf('OLSTEC (lam=%.2f)', lam_val);
        elseif strcmp(name, 'RSI_OLSTEC')
            display_name = 'RSI-OLSTEC (Ours)';
        else
            display_name = strrep(name, '_', '-');
        end
        
        fprintf('%-25s | %.6e ± %.6e\n', display_name, mean_err, std_err);
    end
end
fprintf('------------------------------------------------------------------------------------------\n');
fprintf('Experiment R1 Completed.\n');