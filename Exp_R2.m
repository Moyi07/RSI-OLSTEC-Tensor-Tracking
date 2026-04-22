%% Experiment R2
% =========================================================================
% Objective:
%   Evaluate and compare the performance of RSI-OLSTEC against baseline 
%   algorithms on real-world video data. 
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

% Generate Sampling Mask (Omega)
rng(42); 
OmegaTensor = rand(rows, cols, total_slices) < fraction;
Tensor_Y_Normalized = Tensor_Y_Noiseless .* OmegaTensor;

% Format for Matrix-based Algorithms
Matrix_Y_Noiseless = reshape(Tensor_Y_Noiseless, [rows*cols, total_slices]);
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
    width_vals = width_vals(:);
    
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
    [Xsol_cp_wopt, info_cp_wopt, sub_infos_cp_wopt] = cp_wopt_mod(Tensor_Y_Noiseless, OmegaTensor, [], tensor_dims, rank_r, Xinit, options);
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
    options.lambda              = 0.98;
    
    tic; 
    [Xsol_petrels, infos_petrels, sub_infos_petrels, ~] = petrels_mod([], Matrix_Y_Noiseless, OmegaMatrix, [], numr, numc, options);    
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
    [Xsol_grasta, infos_grasta, sub_infos_grasta, ~] = grasta_mod([], Matrix_Y_Noiseless, OmegaMatrix, [], numr, numc, options);
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
    [Xsol_grouse, infos_grouse, sub_infos_grouse, ~] = grouse_mod([], Matrix_Y_Noiseless, OmegaMatrix, [], numr, numc, options);
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
    [Xsol_TeCPSGD, info_TeCPSGD, sub_infos_TeCPSGD] = TeCPSGD(Tensor_Y_Noiseless, OmegaTensor, [], tensor_dims, rank_r, Xinit, options);
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
        [~, ~, sub_infos_olstec_multi{i}] = olstec(Tensor_Y_Noiseless, OmegaTensor, [], tensor_dims, rank_r, Xinit, options);
        elapsed_time_olstec_multi(i) = toc;
    end
    
    sub_infos_olstec = sub_infos_olstec_multi{3}; 
    elapsed_time_olstec = elapsed_time_olstec_multi(3);
end

%% (7) RSI-OLSTEC
if run_rsi_olstec
    fprintf('Running RSI-OLSTEC...\n');
    clear options;
    options                = struct();
    options.maxepochs      = maxepochs;
    options.lambda_max     = 0.80;
    options.lambda_min     = 0.70;             
    options.huber_delta    = 0.30;
    options.min_grad_threshold = 0.70; 
    options.mu             = 0.01;
    options.tolcost        = tolcost; 
    options.permute_on     = permute_on_flag;
    options.verbose        = verbose;
    options.store_matrix   = store_matrix_flag;
    options.store_subinfo  = true;
    tic
    [Xsol_rsi, infos_rsi, sub_infos_rsi] = rsi_olstec(Tensor_Y_Noiseless, OmegaTensor, [], size(Tensor_Y_Noiseless), rank_r, Xinit, options, aux_width);
    elapsed_time_rsi = toc;
end

%% 5. Plotting & Visualization
% -------------------------------------------------------------------------
fs = 14;

% --- Figure 1: Normalized Residual Error ---
h1 = figure('Name', 'Residual Error Comparison'); 
hold on;
leg_str = {};
safe_plot = @(info, color, name) plot_safe_log(info.inner_iter, info.err_residual, color, name);

if run_cpwopt && exist('sub_infos_cp_wopt','var')
    safe_plot(sub_infos_cp_wopt, '-k', 'CP-WOPT'); 
    leg_str{end+1} = 'CP-WOPT'; 
end
if run_grouse && exist('sub_infos_grouse','var'), safe_plot(sub_infos_grouse, '-g', 'Grouse'); leg_str{end+1} = 'Grouse'; end
if run_grasta && exist('sub_infos_grasta','var'), safe_plot(sub_infos_grasta, '-y', 'Grasta'); leg_str{end+1} = 'Grasta'; end
if run_petrels && exist('sub_infos_petrels','var'), safe_plot(sub_infos_petrels, '-m', 'Petrels'); leg_str{end+1} = 'Petrels'; end
if run_tecpsgd && exist('sub_infos_TeCPSGD','var'), safe_plot(sub_infos_TeCPSGD, '-b', 'TeCPSGD'); leg_str{end+1} = 'TeCPSGD'; end

if run_olstec && exist('sub_infos_olstec_multi','var')
    olstec_linespecs = {':c', '--c', '-c', '-.c'}; 
    for i = 1:length(lambda_list)
        safe_plot(sub_infos_olstec_multi{i}, olstec_linespecs{i}, ''); 
        leg_str{end+1} = sprintf('OLSTEC (\\lambda=%.2f)', lambda_list(i));
    end
end

if run_rsi_olstec && exist('sub_infos_rsi','var'), safe_plot(sub_infos_rsi, '-r', 'RSI-OLSTEC'); leg_str{end+1} = 'RSI-OLSTEC'; end

hold off; grid on;
legend(leg_str, 'location', 'best', 'FontSize', 12); 
xlabel('Data Stream Index'); ylabel('Normalized Residual Error');
set(gca, 'FontSize', fs);
xlim([1, total_slices]);
savefig(h1, 'Fig_Residual_Error.fig'); 

% --- Figure 2: Running Average Error ---
h2 = figure('Name', 'Running Average Error');
hold on;
leg_str_avg = {};
safe_plot_avg = @(info, color, name) plot_safe_log(info.inner_iter, info.err_run_ave, color, name);

if run_grouse && exist('sub_infos_grouse','var'), safe_plot_avg(sub_infos_grouse, '-g', 'Grouse'); leg_str_avg{end+1} = 'Grouse'; end
if run_grasta && exist('sub_infos_grasta','var'), safe_plot_avg(sub_infos_grasta, '-y', 'Grasta'); leg_str_avg{end+1} = 'Grasta'; end
if run_petrels && exist('sub_infos_petrels','var'), safe_plot_avg(sub_infos_petrels, '-m', 'Petrels'); leg_str_avg{end+1} = 'Petrels'; end
if run_tecpsgd && exist('sub_infos_TeCPSGD','var'), safe_plot_avg(sub_infos_TeCPSGD, '-b', 'TeCPSGD'); leg_str_avg{end+1} = 'TeCPSGD'; end

if run_olstec && exist('sub_infos_olstec_multi','var')
    for i = 1:length(lambda_list)
        safe_plot_avg(sub_infos_olstec_multi{i}, olstec_linespecs{i}, ''); 
        leg_str_avg{end+1} = sprintf('OLSTEC (\\lambda=%.2f)', lambda_list(i));
    end
end

if run_rsi_olstec && exist('sub_infos_rsi','var'), safe_plot_avg(sub_infos_rsi, '-r', 'RSI-OLSTEC'); leg_str_avg{end+1} = 'RSI-OLSTEC'; end

hold off; grid on;
legend(leg_str_avg, 'location', 'best', 'FontSize', 12);
xlabel('Data Stream Index'); ylabel('Running Average Error');
set(gca, 'FontSize', fs);
xlim([1, total_slices]);
savefig(h2, 'Fig_Running_Average_Error.fig');
fprintf('Figures saved successfully.\n');

%% 6. Quantitative Results
% -------------------------------------------------------------------------
fprintf('\n==========================================================================================\n');
fprintf('QUANTITATIVE RESULTS COMPARISON\n');
fprintf('==========================================================================================\n');
fprintf('%-22s | %-12s | %-20s | %-20s\n', 'Algorithm', 'Time [sec]', 'Final Norm. Resid Err', 'Final Run. Avg Err');
fprintf('------------------------------------------------------------------------------------------\n');

getLast = @(v) v(end);

if run_cpwopt && exist('sub_infos_cp_wopt', 'var'), fprintf('%-22s | %-12.4f | %-20.6e | %-20s\n', 'CP-WOPT', elapsed_time_cpwopt, getLast(sub_infos_cp_wopt.err_residual), 'N/A (Batch)'); end
if run_tecpsgd && exist('sub_infos_TeCPSGD', 'var'), fprintf('%-22s | %-12.4f | %-20.6e | %-20.6e\n', 'TeCPSGD', elapsed_time_tecpsgd, getLast(sub_infos_TeCPSGD.err_residual), getLast(sub_infos_TeCPSGD.err_run_ave)); end
if run_petrels && exist('sub_infos_petrels', 'var'), fprintf('%-22s | %-12.4f | %-20.6e | %-20.6e\n', 'Petrels', elapsed_time_petrels, getLast(sub_infos_petrels.err_residual), getLast(sub_infos_petrels.err_run_ave)); end
if run_grouse && exist('sub_infos_grouse', 'var'), fprintf('%-22s | %-12.4f | %-20.6e | %-20.6e\n', 'Grouse', elapsed_time_grouse, getLast(sub_infos_grouse.err_residual), getLast(sub_infos_grouse.err_run_ave)); end
if run_grasta && exist('sub_infos_grasta', 'var'), fprintf('%-22s | %-12.4f | %-20.6e | %-20.6e\n', 'Grasta', elapsed_time_grasta, getLast(sub_infos_grasta.err_residual), getLast(sub_infos_grasta.err_run_ave)); end

if run_olstec && exist('sub_infos_olstec_multi', 'var')
    for i = 1:length(lambda_list)
        algo_name_str = sprintf('OLSTEC (lam=%.2f)', lambda_list(i));
        fprintf('%-22s | %-12.4f | %-20.6e | %-20.6e\n', algo_name_str, elapsed_time_olstec_multi(i), getLast(sub_infos_olstec_multi{i}.err_residual), getLast(sub_infos_olstec_multi{i}.err_run_ave)); 
    end
end
if run_rsi_olstec && exist('sub_infos_rsi', 'var'), fprintf('%-22s | %-12.4f | %-20.6e | %-20.6e\n', 'RSI-OLSTEC (Ours)', elapsed_time_rsi, getLast(sub_infos_rsi.err_residual), getLast(sub_infos_rsi.err_run_ave)); end
fprintf('------------------------------------------------------------------------------------------\n');

%% 7. Real-time Frame Visualization
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
    warning('No frames available for display. Please check target_frames.');
else
    fprintf('Generating multi-frame comparison matrix (Frames: %s)...\n', num2str(target_frames));
    
    algo_plot_list = {};
    
    if run_grouse && exist('sub_infos_grouse','var'),     algo_plot_list(end+1,:) = {sub_infos_grouse, 'Grouse'}; end
    if run_tecpsgd && exist('sub_infos_TeCPSGD','var'),   algo_plot_list(end+1,:) = {sub_infos_TeCPSGD, 'TeCPSGD'}; end
    if run_grasta && exist('sub_infos_grasta','var'),     algo_plot_list(end+1,:) = {sub_infos_grasta, 'Grasta'}; end
    if run_petrels && exist('sub_infos_petrels','var'),   algo_plot_list(end+1,:) = {sub_infos_petrels, 'Petrels'}; end
    
    if run_olstec && exist('sub_infos_olstec','var')
        olstec_label = sprintf('OLSTEC (\\lambda=%.2f)', lambda_list(3)); 
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
        
        imagesc(Tensor_Y_Noiseless(:, :, frame_idx));
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
                warning('L matrix not found for algorithm %s. Frame skipped.', algo_name);
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
    fprintf('Saved multi-frame comparison: Fig_MultiFrame_Matrix.fig\n');
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

function plot_safe_log(x, y, color, ~)
    valid_idx = (x > 0) & (y > 0);
    semilogy(x(valid_idx), y(valid_idx), color, 'linewidth', 2.0);
end