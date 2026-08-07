%% Export supplementary tracking video
% =========================================================================
% Export a frame-by-frame comparison for the real WAAM-ViD case using the
% same tracking configuration as Exp_R2.m.
%
% Output:
%   result/R2/supplementary_video/R2_supplementary_tracking_video.mp4
%
% The video compares online trackers only. CP-WOPT is excluded because it is
% an offline batch method.
% =========================================================================

clear;
clc;
close all;

repo_dir = fileparts(mfilename('fullpath'));
run(fullfile(repo_dir, 'run_me_first.m'));

%% User-adjustable export settings
cfg = struct();
cfg.video_filename = fullfile(repo_dir, 'dataset', 'video', '250312-110206-video_1.mp4');
cfg.meta_filename  = fullfile(repo_dir, 'dataset', 'WAMVID_metadata.csv');
cfg.output_dir     = fullfile(repo_dir, 'result', 'R2', 'supplementary_video');
cfg.output_name    = 'R2_supplementary_tracking_video';

cfg.frame_rate     = 15;
cfg.frame_step     = 1;      % Use 1 for all frames; increase for a shorter preview.
cfg.video_quality  = 95;
cfg.overwrite      = false;  % If false, a timestamp suffix is added when needed.
cfg.show_figure    = true;   % Set false for unattended export if supported by MATLAB.

%% R2 experiment settings
cfg.fraction        = 0.10;
cfg.rank_r          = 20;
cfg.max_frames      = 623;
cfg.scale_ratio     = 0.20;
cfg.random_seed     = 42;
cfg.matrix_init_seed = 40001;
cfg.initial_calibration_frames = 30;
cfg.maxepochs       = 1;
cfg.tolcost         = 1e-8;
cfg.verbose         = 0;
cfg.permute_on_flag = false;
cfg.store_matrix    = true;

cfg.petrels_lambda  = 0.98;
cfg.tecpsgd_lambda  = 0.99;
cfg.olstec_lambda   = 0.80;
cfg.rsi_lambda_max  = 0.80;
cfg.rsi_lambda_min  = 0.70;
cfg.rsi_mu          = 0.01;
cfg.olstec_mu       = 0.01;
cfg.rsi_min_grad_floor = 0.05;
cfg.rsi_grad_ema_alpha = 0.999;
cfg.rsi_huber_delta    = NaN;
cfg.display_intensity_limits = [0, 1];
cfg.display_residual_limits  = [-1, 1];

if ~exist(cfg.output_dir, 'dir')
    mkdir(cfg.output_dir);
end

fprintf('Preparing R2 supplementary video export...\n');
fprintf('Repository: %s\n', repo_dir);

%% Load and preprocess the selected WAAM video
[Tensor_Y_Original, aux_width, aux_meta] = local_load_r2_video(cfg);
[rows, cols, total_slices] = size(Tensor_Y_Original);
tensor_dims = [rows, cols, total_slices];
if cfg.initial_calibration_frames < 2 || ...
        cfg.initial_calibration_frames ~= ...
        floor(cfg.initial_calibration_frames) || ...
        cfg.initial_calibration_frames >= total_slices
    error('ExportR2Video:InvalidCalibrationLength', ...
        ['initial_calibration_frames must be an integer between 2 and ' ...
         'the number of frames minus one.']);
end

fprintf('Effective R2 video size: %d x %d x %d.\n', rows, cols, total_slices);
fprintf('Auxiliary metadata row: %d.\n', aux_meta.row_idx);

%% Construct the R2 observation mask and adaptive thresholds
rng(cfg.random_seed, 'twister');
OmegaTensor = rand(rows, cols, total_slices) < cfg.fraction;
GammaTensor = [];

[huber_delta_video, huber_delta_strategy] = ...
    local_estimate_initial_calibration_huber_delta( ...
    Tensor_Y_Original, OmegaTensor, cfg.rsi_huber_delta, ...
    cfg.initial_calibration_frames);

diff_aux = diff(aux_width(1:cfg.initial_calibration_frames));
[adaptive_min_grad, ~] = local_estimate_aux_grad_threshold( ...
    diff_aux, cfg.rsi_min_grad_floor);

fprintf(['Initial %d-frame calibration: Huber delta %.4f (%s), ' ...
    'side-information threshold %.4f.\n'], ...
    cfg.initial_calibration_frames, huber_delta_video, ...
    huber_delta_strategy, adaptive_min_grad);

Matrix_Y_Original = reshape(Tensor_Y_Original, [rows * cols, total_slices]);
OmegaMatrix = reshape(OmegaTensor, [rows * cols, total_slices]);
GammaMatrix = [];
numr = rows * cols;
numc = total_slices;
matrix_rank = cfg.rank_r;

% Keep the same initialization position relative to the random mask as R2.
Xinit.A = randn(rows, cfg.rank_r);
Xinit.B = randn(cols, cfg.rank_r);
Xinit.C = randn(total_slices, cfg.rank_r);

% Matrix methods share one initial column space.
matrix_init_stream = RandStream('mt19937ar', ...
    'Seed', cfg.matrix_init_seed);
matrix_init = struct();
matrix_init.U = randn(matrix_init_stream, numr, matrix_rank);
matrix_init.Weight = randn(matrix_init_stream, matrix_rank, numc);

%% Run the online algorithms used in the R2 visual comparison
fprintf('Running PETRELS...\n');
clear options;
options.maxepochs     = cfg.maxepochs;
options.tolcost       = cfg.tolcost;
options.rank          = matrix_rank;
options.permute_on    = cfg.permute_on_flag;
options.store_subinfo = true;
options.store_matrix  = cfg.store_matrix;
options.verbose       = cfg.verbose;
options.lambda        = cfg.petrels_lambda;
[~, ~, sub_infos_petrels, ~] = petrels_mod(matrix_init, Matrix_Y_Original, OmegaMatrix, ...
    GammaMatrix, numr, numc, options);

fprintf('Running GRASTA...\n');
clear options;
options.maxepochs     = cfg.maxepochs;
options.tolcost       = cfg.tolcost;
options.permute_on    = cfg.permute_on_flag;
options.verbose       = cfg.verbose;
options.store_subinfo = true;
options.store_matrix  = cfg.store_matrix;
options.RANK          = matrix_rank;
options.rho           = 1.8;
options.MAX_MU        = 10000;
options.MIN_MU        = 1;
options.ITER_MAX      = 20;
options.DIM_M         = rows * cols;
options.USE_MEX       = 0;
% Reproduce GRASTA's orthonormal basis from the shared matrix seed.
rng(cfg.matrix_init_seed, 'twister');
[~, ~, sub_infos_grasta, ~] = grasta_mod(matrix_init, Matrix_Y_Original, OmegaMatrix, ...
    GammaMatrix, numr, numc, options);

fprintf('Running GROUSE...\n');
clear options;
options.maxrank       = matrix_rank;
options.step_size     = 0.0001;
options.maxepochs     = cfg.maxepochs;
options.tolcost       = cfg.tolcost;
options.permute_on    = cfg.permute_on_flag;
options.store_subinfo = true;
options.store_matrix  = cfg.store_matrix;
options.verbose       = cfg.verbose;
[~, ~, sub_infos_grouse, ~] = grouse_mod(matrix_init, Matrix_Y_Original, OmegaMatrix, ...
    GammaMatrix, numr, numc, options);

fprintf('Running TeCPSGD...\n');
clear options;
options.maxepochs     = cfg.maxepochs;
options.tolcost       = cfg.tolcost;
options.lambda        = cfg.tecpsgd_lambda;
options.stepsize      = 0.10;
options.mu            = 0.01;
options.permute_on    = cfg.permute_on_flag;
options.store_subinfo = true;
options.store_matrix  = cfg.store_matrix;
options.verbose       = cfg.verbose;
[~, ~, sub_infos_tecpsgd] = TeCPSGD(Tensor_Y_Original, OmegaTensor, GammaTensor, ...
    tensor_dims, cfg.rank_r, Xinit, options);

fprintf('Running OLSTEC (lambda=%.2f)...\n', cfg.olstec_lambda);
clear options;
options.maxepochs     = cfg.maxepochs;
options.tolcost       = cfg.tolcost;
options.permute_on    = cfg.permute_on_flag;
options.lambda        = cfg.olstec_lambda;
options.mu            = cfg.olstec_mu;
options.tw_flag       = 0;
options.tw_len        = 10;
options.store_subinfo = true;
options.store_matrix  = cfg.store_matrix;
options.verbose       = cfg.verbose;
[~, ~, sub_infos_olstec] = olstec(Tensor_Y_Original, OmegaTensor, GammaTensor, ...
    tensor_dims, cfg.rank_r, Xinit, options);

fprintf('Running RSI-OLSTEC...\n');
clear options;
options.maxepochs          = cfg.maxepochs;
options.lambda_max         = cfg.rsi_lambda_max;
options.lambda_min         = cfg.rsi_lambda_min;
options.huber_delta        = huber_delta_video;
options.min_grad_threshold = adaptive_min_grad;
options.grad_ema_alpha     = cfg.rsi_grad_ema_alpha;
options.mu                 = cfg.rsi_mu;
options.tolcost            = cfg.tolcost;
options.permute_on         = cfg.permute_on_flag;
options.verbose            = cfg.verbose;
options.store_matrix       = true;
options.store_subinfo      = true;
[~, ~, sub_infos_rsi] = rsi_olstec(Tensor_Y_Original, OmegaTensor, GammaTensor, ...
    tensor_dims, cfg.rank_r, Xinit, options, aux_width);

%% Assemble visual comparison list
plot_infos = { ...
    sub_infos_petrels, 'Petrels'; ...
    sub_infos_grasta,  'Grasta'; ...
    sub_infos_grouse,  'Grouse'; ...
    sub_infos_tecpsgd, 'TeCPSGD'; ...
    sub_infos_olstec,  'OLSTEC'; ...
    sub_infos_rsi,     'RSI-OLSTEC'};

for k = 1:size(plot_infos, 1)
    local_validate_video_output(plot_infos{k, 1}, plot_infos{k, 2}, ...
        rows, cols, total_slices);
end

%% Export video
[video_writer, video_path] = local_open_video_writer(cfg);
fig = local_initialize_video_figure(plot_infos(:, 2), rows, cols, ...
    cfg.fraction, cfg.show_figure, cfg.display_intensity_limits, ...
    cfg.display_residual_limits);

cleanup_obj = onCleanup(@() local_cleanup_export(video_writer, fig));

frame_ids = 1:cfg.frame_step:total_slices;
fprintf('Writing %d video frames to:\n%s\n', numel(frame_ids), video_path);

target_frame_size = [];
for n = 1:numel(frame_ids)
    frame_idx = frame_ids(n);
    local_update_video_frame(fig, plot_infos, rows, cols, frame_idx);
    drawnow;
    [video_frame, target_frame_size] = local_capture_fixed_frame(fig, target_frame_size);
    writeVideo(video_writer, video_frame);

    if mod(n, 25) == 0 || n == numel(frame_ids)
        fprintf('  Exported %d / %d frames.\n', n, numel(frame_ids));
    end
end

close(video_writer);
delete(cleanup_obj);
if isvalid(fig)
    close(fig);
end

fprintf('R2 supplementary video export complete:\n%s\n', video_path);

%% Local helper functions
function [Tensor_Y_Original, aux_width, aux_meta] = local_load_r2_video(cfg)
    fprintf('Reading video: %s ...\n', cfg.video_filename);
    if ~exist(cfg.video_filename, 'file')
        error('ExportR2Video:VideoNotFound', 'Video file not found: %s', cfg.video_filename);
    end

    v = VideoReader(cfg.video_filename);
    if ~hasFrame(v)
        error('ExportR2Video:NoFramesRead', 'No frames were read from video: %s', cfg.video_filename);
    end

    raw_frame = readFrame(v);
    if size(raw_frame, 3) == 3
        gray_frame = rgb2gray(raw_frame);
    else
        gray_frame = raw_frame;
    end

    img_resized = imresize(gray_frame, cfg.scale_ratio);
    Tensor_Y_Original = zeros(size(img_resized, 1), size(img_resized, 2), cfg.max_frames);
    frame_idx = 1;
    Tensor_Y_Original(:, :, frame_idx) = im2double(img_resized);

    while hasFrame(v) && frame_idx < cfg.max_frames
        frame_idx = frame_idx + 1;
        raw_frame = readFrame(v);

        if size(raw_frame, 3) == 3
            gray_frame = rgb2gray(raw_frame);
        else
            gray_frame = raw_frame;
        end

        img_resized = imresize(gray_frame, cfg.scale_ratio);
        Tensor_Y_Original(:, :, frame_idx) = im2double(img_resized);
    end

    if frame_idx < cfg.max_frames
        Tensor_Y_Original = Tensor_Y_Original(:, :, 1:frame_idx);
    end

    total_slices = size(Tensor_Y_Original, 3);
    [aux_width, aux_meta] = load_waam_width_signal(cfg.meta_filename, ...
        cfg.video_filename, total_slices, 'trim_leading_nan');

    if aux_meta.num_trimmed_front > 0
        Tensor_Y_Original = Tensor_Y_Original(:, :, aux_meta.trim_start_frame:end);
    end

    if numel(aux_width) ~= size(Tensor_Y_Original, 3)
        error('ExportR2Video:AuxLengthMismatch', ...
            'Auxiliary signal length (%d) does not match effective frame count (%d).', ...
            numel(aux_width), size(Tensor_Y_Original, 3));
    end
end

function [huber_delta_video, strategy] = ...
    local_estimate_initial_calibration_huber_delta( ...
    Tensor_Y, OmegaTensor, override_delta, calibration_frames)

    diff_pixels = cell(calibration_frames - 1, 1);

    for t_idx = 2:calibration_frames
        common_mask = OmegaTensor(:, :, t_idx) & OmegaTensor(:, :, t_idx - 1);
        diff_frame = Tensor_Y(:, :, t_idx) - Tensor_Y(:, :, t_idx - 1);
        diff_pixels{t_idx - 1} = diff_frame(common_mask);
    end

    if isempty(diff_pixels)
        diff_pixels = [];
    else
        diff_pixels = vertcat(diff_pixels{:});
    end

    strategy = 'initial_calibration_pixel_diff_mad';
    if ~isnan(override_delta)
        huber_delta_video = override_delta;
        strategy = 'config_override';
    elseif ~isempty(diff_pixels)
        mad_val = median(abs(diff_pixels - median(diff_pixels, 'omitnan')), 'omitnan');
        est_sigma = (1.4826 * mad_val) / sqrt(2);
        huber_delta_video = max(0.10, min(0.15, 6 * est_sigma));
    else
        huber_delta_video = 0.15;
        strategy = 'initial_calibration_fallback_default';
    end
end

function [threshold, stats] = local_estimate_aux_grad_threshold(diff_aux, floor_value)
    finite_diff = diff_aux(:);
    finite_diff = finite_diff(isfinite(finite_diff));
    abs_diff = abs(finite_diff);

    stats = struct();
    stats.count = numel(finite_diff);
    stats.median_abs = NaN;
    stats.mad_signed = NaN;

    if isempty(finite_diff)
        threshold = floor_value;
        return;
    end

    stats.median_abs = median(abs_diff, 'omitnan');
    signed_center = median(finite_diff, 'omitnan');
    stats.mad_signed = median(abs(finite_diff - signed_center), 'omitnan');
    est_aux_sigma = (1.4826 * stats.mad_signed) / sqrt(2);
    threshold = 3 * sqrt(2) * est_aux_sigma;

    if ~isfinite(threshold)
        threshold = floor_value;
    end
    threshold = max(floor_value, threshold);
end

function [writer, video_path] = local_open_video_writer(cfg)
    video_path = fullfile(cfg.output_dir, [cfg.output_name '.mp4']);
    if exist(video_path, 'file') && ~cfg.overwrite
        stamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
        video_path = fullfile(cfg.output_dir, [cfg.output_name '_' stamp '.mp4']);
    end

    try
        writer = VideoWriter(video_path, 'MPEG-4');
    catch
        [folder, base_name] = fileparts(video_path);
        video_path = fullfile(folder, [base_name '.avi']);
        writer = VideoWriter(video_path, 'Motion JPEG AVI');
    end

    writer.FrameRate = cfg.frame_rate;
    if isprop(writer, 'Quality')
        writer.Quality = cfg.video_quality;
    end
    open(writer);
end

function fig = local_initialize_video_figure(algo_names, rows, cols, ...
    fraction, show_figure, display_intensity_limits, ...
    display_residual_limits)
    visible_state = 'off';
    if show_figure
        visible_state = 'on';
    end

    fig_width = 1800;
    fig_height = 900;
    fig = figure('Name', 'R2 Supplementary Tracking Video', ...
        'Color', 'w', 'Visible', visible_state, ...
        'Position', [50, 50, fig_width, fig_height]);

    num_algos = numel(algo_names);
    layout = tiledlayout(fig, 3, num_algos, 'TileSpacing', 'compact', 'Padding', 'compact');
    try
        layout.Units = 'normalized';
        layout.Position = [0.035, 0.045, 0.94, 0.835];
    catch
        % Older MATLAB releases may not expose the Position property.
    end
    handles = struct();
    handles.obs = gobjects(num_algos, 1);
    handles.lowrank = gobjects(num_algos, 1);
    handles.resid = gobjects(num_algos, 1);
    handles.lowrank_title = gobjects(num_algos, 1);
    handles.resid_title = gobjects(num_algos, 1);
    handles.header = [];
    handles.subheader = [];

    observe_percent = 100 * fraction;
    display_note = sprintf(['Observation ratio = %.0f%% | Fixed display ', ...
        'scales: intensity [%g, %g], signed full-frame residual [%g, %g]'], ...
        observe_percent, display_intensity_limits(1), ...
        display_intensity_limits(2), display_residual_limits(1), ...
        display_residual_limits(2));
    handles.display_note = display_note;

    annotation(fig, 'textbox', [0.02, 0.955, 0.96, 0.032], ...
        'String', 'Supplementary Video: Online Low-Rank Background Tracking in Wire Arc Additive Manufacturing Video', ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
        'BackgroundColor', 'w', 'FontSize', 15, 'FontWeight', 'bold', 'Interpreter', 'none');
    handles.subheader = annotation(fig, 'textbox', [0.02, 0.915, 0.96, 0.032], ...
        'String', display_note, ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
        'BackgroundColor', 'w', 'FontSize', 11, 'Interpreter', 'none');

    for k = 1:num_algos
        ax = nexttile(layout, k);
        handles.obs(k) = imagesc(ax, zeros(rows, cols));
        colormap(ax, gray);
        set(ax, 'CLim', display_intensity_limits);
        axis(ax, 'image');
        axis(ax, 'off');
        title(ax, sprintf('%s: %.0f%% obs', algo_names{k}, observe_percent), ...
            'Interpreter', 'tex', 'FontWeight', 'bold');
        if k == 1
            ylabel(ax, 'Partially observed frame', 'Visible', 'on', 'FontWeight', 'bold');
        end

        ax = nexttile(layout, num_algos + k);
        handles.lowrank(k) = imagesc(ax, zeros(rows, cols));
        colormap(ax, gray);
        set(ax, 'CLim', display_intensity_limits);
        axis(ax, 'image');
        axis(ax, 'off');
        handles.lowrank_title(k) = title(ax, 'Low-rank: f = 1');
        if k == 1
            ylabel(ax, 'Low-rank background', 'Visible', 'on', 'FontWeight', 'bold');
        end

        ax = nexttile(layout, 2 * num_algos + k);
        handles.resid(k) = imagesc(ax, zeros(rows, cols));
        colormap(ax, gray);
        set(ax, 'CLim', display_residual_limits);
        axis(ax, 'image');
        axis(ax, 'off');
        handles.resid_title(k) = title(ax, 'NRE = NaN');
        if k == 1
            ylabel(ax, 'Full-frame residual', 'Visible', 'on', 'FontWeight', 'bold');
        end
    end

    setappdata(fig, 'r2_video_handles', handles);
end

function local_update_video_frame(fig, plot_infos, rows, cols, frame_idx)
    handles = getappdata(fig, 'r2_video_handles');

    for k = 1:size(plot_infos, 1)
        current_info = plot_infos{k, 1};
        obs_frame = local_get_observation_frame(current_info, frame_idx, rows, cols);
        L_frame = local_get_lowrank_frame(current_info, frame_idx, rows, cols);
        residual_frame = local_get_full_residual_frame(current_info, frame_idx, rows, cols);
        curr_err = local_metric_at_frame(current_info, frame_idx);

        set(handles.obs(k), 'CData', obs_frame);
        set(handles.lowrank(k), 'CData', L_frame);
        set(handles.resid(k), 'CData', residual_frame);
        set(handles.lowrank_title(k), 'String', sprintf('Low-rank: f = %d', frame_idx));
        set(handles.resid_title(k), 'String', sprintf('NRE = %.4f', curr_err));
    end

    if ~isempty(handles.subheader) && isvalid(handles.subheader)
        set(handles.subheader, 'String', sprintf( ...
            'Frame %d | %s', frame_idx, handles.display_note));
    end
end

function local_validate_video_output(sub_info, algorithm_name, ...
    rows, cols, expected_length)

    expected_size = [rows * cols, expected_length];
    field_names = {'I', 'L', 'E'};

    for field_idx = 1:numel(field_names)
        field_name = field_names{field_idx};
        if ~isfield(sub_info, field_name)
            error('ExportR2Video:MissingOutput', ...
                '%s did not return the required field %s.', ...
                algorithm_name, field_name);
        end

        value = sub_info.(field_name);
        if ~(isnumeric(value) && isreal(value) && ...
                isequal(size(value), expected_size))
            error('ExportR2Video:InvalidOutput', ...
                ['%s returned an invalid %s array. Expected a real ', ...
                 'numeric array of size %d-by-%d.'], ...
                algorithm_name, field_name, ...
                expected_size(1), expected_size(2));
        end

        if any(~isfinite(value(:)))
            error('ExportR2Video:NonfiniteOutput', ...
                '%s returned nonfinite values in %s.', ...
                algorithm_name, field_name);
        end
    end

    for frame_idx = 1:expected_length
        metric_value = local_metric_at_frame(sub_info, frame_idx);
        if ~(isnumeric(metric_value) && isreal(metric_value) && ...
                isscalar(metric_value) && isfinite(metric_value))
            error('ExportR2Video:InvalidMetric', ...
                '%s has no valid NRE at frame %d.', ...
                algorithm_name, frame_idx);
        end
    end
end

function obs_frame = local_get_observation_frame(sub_info, frame_idx, rows, cols)
    obs_frame = reshape(sub_info.I(:, frame_idx), [rows, cols]);
end

function L_frame = local_get_lowrank_frame(sub_info, frame_idx, rows, cols)
    L_frame = reshape(sub_info.L(:, frame_idx), [rows, cols]);
end

function residual_frame = local_get_full_residual_frame(sub_info, frame_idx, rows, cols)
    % E is the full-frame evaluation residual Y - L.
    residual_frame = reshape(sub_info.E(:, frame_idx), [rows, cols]);
end

function val = local_metric_at_frame(sub_info, frame_idx)
    metric = [];
    if isfield(sub_info, 'err_residual')
        metric = sub_info.err_residual(:);
    end

    val = NaN;
    if isempty(metric)
        return;
    end

    if isfield(sub_info, 'inner_iter') && ~isempty(sub_info.inner_iter)
        idx = find(sub_info.inner_iter == frame_idx, 1, 'first');
        if ~isempty(idx) && idx <= numel(metric)
            val = metric(idx);
            return;
        end
    end

    if numel(metric) >= frame_idx + 1 && (isnan(metric(1)) || metric(1) == 0)
        val = metric(frame_idx + 1);
    elseif numel(metric) >= frame_idx
        val = metric(frame_idx);
    end
end

function [frame_out, target_frame_size] = local_capture_fixed_frame(fig, target_frame_size)
    frame_out = getframe(fig);

    if isempty(target_frame_size)
        target_frame_size = size(frame_out.cdata);
        return;
    end

    current_size = size(frame_out.cdata);
    if numel(current_size) < 3
        current_size(3) = 1;
    end

    if any(current_size(1:2) ~= target_frame_size(1:2))
        frame_out.cdata = imresize(frame_out.cdata, target_frame_size(1:2));
    end
end

function local_cleanup_export(writer, fig)
    try
        close(writer);
    catch
    end
    try
        if isvalid(fig)
            close(fig);
        end
    catch
    end
end
