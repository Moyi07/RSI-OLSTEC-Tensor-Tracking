function result = run_r2_waamvid_single(config)
%RUN_R2_WAAMVID_SINGLE Run one R2 WAAM-ViD video with fixed settings.
%   RESULT contains the experiment settings and quantitative results.

    if nargin < 1 || isempty(config)
        config = struct();
    end

    repo_dir = fileparts(fileparts(mfilename('fullpath')));

    run_cpwopt      = local_get(config, 'run_cpwopt', true);
    run_petrels     = local_get(config, 'run_petrels', true);
    run_grasta      = local_get(config, 'run_grasta', true);
    run_grouse      = local_get(config, 'run_grouse', true);
    run_tecpsgd     = local_get(config, 'run_tecpsgd', true);
    run_olstec      = local_get(config, 'run_olstec', true);
    run_rsi_olstec  = local_get(config, 'run_rsi_olstec', true);

    image_display_flag = local_get(config, 'image_display_flag', strcmp(getenv('RSI_IMAGE_DISPLAY'), '1'));
    make_figures       = local_get(config, 'make_figures', true);
    make_matrix_figure = local_get(config, 'make_matrix_figure', true);
    close_figures      = local_get(config, 'close_figures', false);
    store_matrix_flag  = local_get(config, 'store_matrix_flag', true);
    visual_output_required = image_display_flag || make_matrix_figure;
    if visual_output_required && ~store_matrix_flag
        error('run_r2_waamvid_single:VisualOutputRequiresStoredMatrices', ...
            ['Visual output requires store_matrix_flag=true so that the ', ...
             'observation, reconstruction, and residual matrices are available.']);
    end
    permute_on_flag    = local_get(config, 'permute_on_flag', false);
    maxepochs          = local_get(config, 'maxepochs', 1);
    verbose            = local_get(config, 'verbose', 0);
    tolcost            = local_get(config, 'tolcost', 1e-8);
    rank_r             = local_get(config, 'rank_r', 20);
    fraction           = local_get(config, 'fraction', 0.1);
    random_seed        = local_get(config, 'random_seed', 42);
    matrix_init_seed   = local_get(config, 'matrix_init_seed', 40001);
    metric_label       = 'Full-frame Normalized Residual Error';
    export_results     = local_get(config, 'export_results', true);
    result_dir         = local_get(config, 'result_dir', fullfile(repo_dir, 'result', 'R2'));

    video_filename = local_get(config, 'video_filename', ...
        fullfile(repo_dir, 'dataset', 'video', '250312-110206-video_1.mp4'));
    meta_filename = local_get(config, 'meta_filename', ...
        fullfile(repo_dir, 'dataset', 'WAMVID_metadata.csv'));

    scale_ratio = local_get(config, 'scale_ratio', 0.2);
    max_frames  = local_get(config, 'max_frames', 623);
    aux_missing_policy = local_get(config, 'aux_missing_policy', 'trim_leading_nan');
    initial_calibration_frames = local_get( ...
        config, 'initial_calibration_frames', 30);
    lambda_list = local_get(config, 'lambda_list', [0.70, 0.80, 0.90, 0.99]);
    rsi_lambda_max = local_get(config, 'rsi_lambda_max', 0.80);
    rsi_lambda_min = local_get(config, 'rsi_lambda_min', 0.70);
    rsi_mu = local_get(config, 'rsi_mu', 0.01);
    rsi_min_grad_floor = local_get(config, 'rsi_min_grad_floor', 0.05);
    rsi_grad_ema_alpha = local_get(config, 'rsi_grad_ema_alpha', 0.999);
    rsi_huber_delta = local_get(config, 'rsi_huber_delta', NaN);
    olstec_mu = local_get(config, 'olstec_mu', 0.01);
    tecpsgd_lambda = local_get(config, 'tecpsgd_lambda', 0.99);
    petrels_lambda = local_get(config, 'petrels_lambda', 0.98);
    display_intensity_limits = local_get(config, 'display_intensity_limits', [0, 1]);
    display_residual_limits = local_get(config, 'display_residual_limits', [-1, 1]);

    if export_results && ~exist(result_dir, 'dir')
        mkdir(result_dir);
    end

    elapsed_time_cpwopt = NaN;
    elapsed_time_petrels = NaN;
    elapsed_time_grasta = NaN;
    elapsed_time_grouse = NaN;
    elapsed_time_tecpsgd = NaN;
    elapsed_time_olstec_multi = NaN(numel(lambda_list), 1);
    elapsed_time_rsi = NaN;
    idx_display = find(abs(lambda_list - rsi_lambda_max) < 1e-12, 1);
    if isempty(idx_display)
        [~, idx_display] = min(abs(lambda_list - rsi_lambda_max));
        if run_olstec
            warning('run_r2_waamvid_single:NearestRepresentativeLambda', ...
                ['lambda_list does not contain rsi_lambda_max=%.3f; ', ...
                'using nearest OLSTEC representative lambda=%.3f.'], ...
                rsi_lambda_max, lambda_list(idx_display));
        end
    end

    %% 1. Data loading and preprocessing
    fprintf('Reading video: %s ...\n', video_filename);
    if ~exist(video_filename, 'file')
        error('run_r2_waamvid_single:VideoNotFound', ...
            'Video file not found: %s', video_filename);
    end

    v = VideoReader(video_filename);
    if ~hasFrame(v)
        error('run_r2_waamvid_single:NoFramesRead', ...
            'No frames were read from video: %s', video_filename);
    end

    raw_frame = readFrame(v);
    if size(raw_frame, 3) == 3
        gray_frame = rgb2gray(raw_frame);
    else
        gray_frame = raw_frame;
    end
    img_resized = imresize(gray_frame, scale_ratio);
    Tensor_Y_Original = zeros(size(img_resized, 1), size(img_resized, 2), max_frames);
    frame_idx = 1;
    Tensor_Y_Original(:, :, frame_idx) = im2double(img_resized);

    while hasFrame(v) && frame_idx < max_frames
        frame_idx = frame_idx + 1;
        raw_frame = readFrame(v);

        if size(raw_frame, 3) == 3
            gray_frame = rgb2gray(raw_frame);
        else
            gray_frame = raw_frame;
        end

        img_resized = imresize(gray_frame, scale_ratio);
        Tensor_Y_Original(:, :, frame_idx) = im2double(img_resized);
    end

    if frame_idx < max_frames
        Tensor_Y_Original = Tensor_Y_Original(:, :, 1:frame_idx);
    end

    [rows, cols, total_slices] = size(Tensor_Y_Original);
    tensor_dims = [rows, cols, total_slices];

    fprintf('Preprocessing complete.\n');
    fprintf('Resolution: %d x %d \n', rows, cols);
    fprintf('Total frames: %d\n', total_slices);

    fprintf('Loading auxiliary data...\n');
    [aux_width, aux_meta] = load_waam_width_signal(meta_filename, video_filename, total_slices, aux_missing_policy);
    fprintf('Metadata loaded successfully from row %d.\n', aux_meta.row_idx);
    if aux_meta.num_trimmed_front > 0
        Tensor_Y_Original = Tensor_Y_Original(:, :, aux_meta.trim_start_frame:end);
        [rows, cols, total_slices] = size(Tensor_Y_Original);
        tensor_dims = [rows, cols, total_slices];
        fprintf('Trimmed first %d frames with unavailable Width_mm. Effective data: %d x %d x %d.\n', ...
            aux_meta.num_trimmed_front, rows, cols, total_slices);
    end
    if numel(aux_width) ~= total_slices
        error('run_r2_waamvid_single:AuxLengthMismatch', ...
            'Auxiliary signal length (%d) does not match effective frame count (%d).', ...
            numel(aux_width), total_slices);
    end
    if ~(isnumeric(initial_calibration_frames) && ...
            isscalar(initial_calibration_frames) && ...
            isfinite(initial_calibration_frames) && ...
            initial_calibration_frames == floor(initial_calibration_frames) && ...
            initial_calibration_frames >= 2 && ...
            initial_calibration_frames < total_slices)
        error('run_r2_waamvid_single:InvalidCalibrationLength', ...
            ['initial_calibration_frames must be an integer between 2 and ' ...
             'the number of frames minus one.']);
    end

    rng(random_seed, 'twister');
    OmegaTensor = rand(rows, cols, total_slices) < fraction;
    GammaTensor = [];
    fprintf('Reporting %s without a held-out mask.\n', metric_label);

    diff_pixels = cell(initial_calibration_frames - 1, 1);
    for t_idx = 2:initial_calibration_frames
        common_mask = OmegaTensor(:,:,t_idx) & OmegaTensor(:,:,t_idx-1);
        diff_frame = Tensor_Y_Original(:,:,t_idx) - Tensor_Y_Original(:,:,t_idx-1);
        diff_pixels{t_idx-1} = diff_frame(common_mask);
    end
    if isempty(diff_pixels)
        diff_pixels = [];
    else
        diff_pixels = vertcat(diff_pixels{:});
    end
    huber_delta_strategy = 'initial_calibration_pixel_diff_mad';
    if ~isnan(rsi_huber_delta)
        huber_delta_video = rsi_huber_delta;
        huber_delta_strategy = 'config_override';
    elseif ~isempty(diff_pixels)
        mad_val = median(abs(diff_pixels - median(diff_pixels, 'omitnan')), 'omitnan');
        est_sigma = (1.4826 * mad_val) / sqrt(2);
        huber_delta_video = max(0.10, min(0.15, 6 * est_sigma));
    else
        huber_delta_video = 0.15;
        huber_delta_strategy = 'initial_calibration_fallback_default';
    end
    fprintf(['Huber delta estimated from the initial %d-frame calibration ' ...
        'interval: %.4f (strategy=%s)\n'], initial_calibration_frames, ...
        huber_delta_video, huber_delta_strategy);

    Matrix_Y_Original = reshape(Tensor_Y_Original, [rows*cols, total_slices]);
    OmegaMatrix = reshape(OmegaTensor, [rows*cols, total_slices]);
    if isempty(GammaTensor)
        GammaMatrix = [];
    else
        GammaMatrix = reshape(GammaTensor, [rows*cols, total_slices]);
    end

    numr = rows * cols;
    numc = total_slices;
    matrix_rank = rank_r;
    fprintf('Matrix baseline rank set to %d.\n', matrix_rank);

    Xinit.A = randn(rows, rank_r);
    Xinit.B = randn(cols, rank_r);
    Xinit.C = randn(total_slices, rank_r);

    % Matrix methods share one initial column space.
    matrix_init_stream = RandStream('mt19937ar', ...
        'Seed', matrix_init_seed);
    matrix_init = struct();
    matrix_init.U = randn(matrix_init_stream, numr, matrix_rank);
    matrix_init.Weight = randn(matrix_init_stream, matrix_rank, numc);

    diff_aux = diff(aux_width(1:initial_calibration_frames));
    [adaptive_min_grad, aux_grad_stats] = local_estimate_aux_grad_threshold( ...
        diff_aux, rsi_min_grad_floor);
    fprintf(['Side-information threshold estimated from the initial %d-frame ' ...
        'calibration interval: %.4f (MAD-3sigma, median|diff|=%.4f, ' ...
        'MAD=%.4f)\n'], initial_calibration_frames, adaptive_min_grad, ...
        aux_grad_stats.median_abs, aux_grad_stats.mad_signed);

    %% 2. Algorithm execution
    if run_cpwopt
        fprintf('Running CP-WOPT...\n');
        clear options;
        options.maxepochs       = 30;
        options.display_iters   = 1;
        options.tolcost         = tolcost;
        options.store_subinfo   = true;
        options.store_matrix    = store_matrix_flag;
        options.verbose         = verbose;

        tic;
        [~, ~, sub_infos_cp_wopt] = cp_wopt_mod(Tensor_Y_Original, OmegaTensor, GammaTensor, tensor_dims, rank_r, Xinit, options);
        elapsed_time_cpwopt = toc;
    end

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
        options.lambda              = petrels_lambda;

        tic;
        [~, ~, sub_infos_petrels, ~] = petrels_mod(matrix_init, Matrix_Y_Original, OmegaMatrix, GammaMatrix, numr, numc, options);
        elapsed_time_petrels = toc;
    end

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
        % Reproduce GRASTA's orthonormal basis from the shared matrix seed.
        rng(matrix_init_seed, 'twister');
        tic;
        [~, ~, sub_infos_grasta, ~] = grasta_mod(matrix_init, Matrix_Y_Original, OmegaMatrix, GammaMatrix, numr, numc, options);
        elapsed_time_grasta = toc;
    end

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
        [~, ~, sub_infos_grouse, ~] = grouse_mod(matrix_init, Matrix_Y_Original, OmegaMatrix, GammaMatrix, numr, numc, options);
        elapsed_time_grouse = toc;
    end

    if run_tecpsgd
        fprintf('Running TeCPSGD...\n');
        clear options;
        options.maxepochs       = maxepochs;
        options.tolcost         = tolcost;
        options.lambda          = tecpsgd_lambda;
        options.stepsize        = 0.10;
        options.mu              = 0.01;
        options.permute_on      = permute_on_flag;
        options.store_subinfo   = true;
        options.store_matrix    = store_matrix_flag;
        options.verbose         = verbose;
        tic;
        [~, ~, sub_infos_TeCPSGD] = TeCPSGD(Tensor_Y_Original, OmegaTensor, GammaTensor, tensor_dims, rank_r, Xinit, options);
        elapsed_time_tecpsgd = toc;
    end

    if run_olstec
        fprintf('Running OLSTEC with multiple lambdas...\n');
        sub_infos_olstec_multi = cell(length(lambda_list), 1);

        for i = 1:length(lambda_list)
            current_lambda = lambda_list(i);
            fprintf('  -> Testing OLSTEC with lambda = %.2f\n', current_lambda);

            clear options;
            options.maxepochs       = maxepochs;
            options.tolcost         = tolcost;
            options.permute_on      = permute_on_flag;
            options.lambda          = current_lambda;
            options.mu              = olstec_mu;
            options.tw_flag         = 0;
            options.tw_len          = 10;
            options.store_subinfo   = true;
            options.store_matrix    = store_matrix_flag;
            options.verbose         = 0;
            tic;
            [~, ~, sub_infos_olstec_multi{i}] = olstec(Tensor_Y_Original, OmegaTensor, GammaTensor, tensor_dims, rank_r, Xinit, options);
            elapsed_time_olstec_multi(i) = toc;
        end

        sub_infos_olstec = sub_infos_olstec_multi{idx_display};
        fprintf('OLSTEC representative fixed at lambda = %.2f.\n', lambda_list(idx_display));
    end

    if run_rsi_olstec
        fprintf('Running RSI-OLSTEC...\n');
        clear options;
        options                = struct();
        options.maxepochs      = maxepochs;
        options.lambda_max     = rsi_lambda_max;
        options.lambda_min     = rsi_lambda_min;
        options.huber_delta    = huber_delta_video;
        options.min_grad_threshold = adaptive_min_grad;
        options.grad_ema_alpha = rsi_grad_ema_alpha;
        options.mu             = rsi_mu;
        options.tolcost        = tolcost;
        options.permute_on     = permute_on_flag;
        options.verbose        = verbose;
        options.store_matrix   = store_matrix_flag;
        options.store_subinfo  = true;
        tic;
        [~, ~, sub_infos_rsi] = rsi_olstec(Tensor_Y_Original, OmegaTensor, GammaTensor, size(Tensor_Y_Original), rank_r, Xinit, options, aux_width);
        elapsed_time_rsi = toc;
    end

    % Reject incomplete or nonfinite trajectories before exporting figures.
    if run_cpwopt
        local_validate_algorithm_metrics(sub_infos_cp_wopt, ...
            'CP-WOPT (Batch)', total_slices, false);
    end
    if run_petrels
        local_validate_algorithm_metrics(sub_infos_petrels, ...
            'Petrels', total_slices, true);
    end
    if run_grasta
        local_validate_algorithm_metrics(sub_infos_grasta, ...
            'Grasta', total_slices, true);
    end
    if run_grouse
        local_validate_algorithm_metrics(sub_infos_grouse, ...
            'Grouse', total_slices, true);
    end
    if run_tecpsgd
        local_validate_algorithm_metrics(sub_infos_TeCPSGD, ...
            'TeCPSGD', total_slices, true);
    end
    if run_olstec
        for i = 1:numel(lambda_list)
            local_validate_algorithm_metrics(sub_infos_olstec_multi{i}, ...
                sprintf('OLSTEC (lam=%.2f)', lambda_list(i)), ...
                total_slices, true);
        end
    end
    if run_rsi_olstec
        local_validate_algorithm_metrics(sub_infos_rsi, ...
            'RSI-OLSTEC (Ours)', total_slices, true);
    end

    if visual_output_required
        if run_petrels
            local_validate_visual_outputs(sub_infos_petrels, ...
                'Petrels', rows, cols, total_slices);
        end
        if run_grasta
            local_validate_visual_outputs(sub_infos_grasta, ...
                'Grasta', rows, cols, total_slices);
        end
        if run_grouse
            local_validate_visual_outputs(sub_infos_grouse, ...
                'Grouse', rows, cols, total_slices);
        end
        if run_tecpsgd
            local_validate_visual_outputs(sub_infos_TeCPSGD, ...
                'TeCPSGD', rows, cols, total_slices);
        end
        if run_olstec
            local_validate_visual_outputs(sub_infos_olstec, ...
                'OLSTEC', rows, cols, total_slices);
        end
        if run_rsi_olstec
            local_validate_visual_outputs(sub_infos_rsi, ...
                'RSI-OLSTEC', rows, cols, total_slices);
        end
    end

    %% 3. Figures for the single-video run
    if make_figures
        fs = 14;

        h1 = figure('Name', 'Residual Error Comparison');
        hold on;
        leg_str = cell(0, 1);
        safe_plot = @(info, color, name) local_plot_safe_log(info.inner_iter, local_get_err_metric(info), color, name);

        if run_cpwopt && exist('sub_infos_cp_wopt','var')
            safe_plot(sub_infos_cp_wopt, '-k', 'CP-WOPT (Batch, offline)');
            leg_str{end+1, 1} = 'CP-WOPT (Batch, offline)';
        end
        if run_grouse && exist('sub_infos_grouse','var'), safe_plot(sub_infos_grouse, '-g', 'Grouse'); leg_str{end+1, 1} = 'Grouse'; end
        if run_grasta && exist('sub_infos_grasta','var'), safe_plot(sub_infos_grasta, '-y', 'Grasta'); leg_str{end+1, 1} = 'Grasta'; end
        if run_petrels && exist('sub_infos_petrels','var'), safe_plot(sub_infos_petrels, '-m', 'Petrels'); leg_str{end+1, 1} = 'Petrels'; end
        if run_tecpsgd && exist('sub_infos_TeCPSGD','var'), safe_plot(sub_infos_TeCPSGD, '-b', 'TeCPSGD'); leg_str{end+1, 1} = 'TeCPSGD'; end

        if run_olstec && exist('sub_infos_olstec_multi','var')
            olstec_labels = cell(length(lambda_list), 1);
            for i = 1:length(lambda_list)
                safe_plot(sub_infos_olstec_multi{i}, local_olstec_linespec(i), '');
                olstec_labels{i} = sprintf('OLSTEC (\\lambda=%.2f)', lambda_list(i));
            end
            leg_str = [leg_str; olstec_labels];
        end

        if run_rsi_olstec && exist('sub_infos_rsi','var'), safe_plot(sub_infos_rsi, '-r', 'RSI-OLSTEC'); leg_str{end+1, 1} = 'RSI-OLSTEC'; end

        hold off; grid on;
        legend(leg_str, 'location', 'best', 'FontSize', 12);
        xlabel('Data Stream Index'); ylabel(metric_label);
        set(gca, 'FontSize', fs);
        xlim([1, total_slices]);
        if export_results
            savefig(h1, fullfile(result_dir, 'Fig_Residual_Error.fig'));
        end

        h2 = figure('Name', 'Running Average Error');
        hold on;
        leg_str_avg = cell(0, 1);
        safe_plot_avg = @(info, color, name) local_plot_safe_log(info.inner_iter, local_get_run_metric(info), color, name);

        if run_grouse && exist('sub_infos_grouse','var'), safe_plot_avg(sub_infos_grouse, '-g', 'Grouse'); leg_str_avg{end+1, 1} = 'Grouse'; end
        if run_grasta && exist('sub_infos_grasta','var'), safe_plot_avg(sub_infos_grasta, '-y', 'Grasta'); leg_str_avg{end+1, 1} = 'Grasta'; end
        if run_petrels && exist('sub_infos_petrels','var'), safe_plot_avg(sub_infos_petrels, '-m', 'Petrels'); leg_str_avg{end+1, 1} = 'Petrels'; end
        if run_tecpsgd && exist('sub_infos_TeCPSGD','var'), safe_plot_avg(sub_infos_TeCPSGD, '-b', 'TeCPSGD'); leg_str_avg{end+1, 1} = 'TeCPSGD'; end

        if run_olstec && exist('sub_infos_olstec_multi','var')
            olstec_avg_labels = cell(length(lambda_list), 1);
            for i = 1:length(lambda_list)
                safe_plot_avg(sub_infos_olstec_multi{i}, local_olstec_linespec(i), '');
                olstec_avg_labels{i} = sprintf('OLSTEC (\\lambda=%.2f)', lambda_list(i));
            end
            leg_str_avg = [leg_str_avg; olstec_avg_labels];
        end

        if run_rsi_olstec && exist('sub_infos_rsi','var'), safe_plot_avg(sub_infos_rsi, '-r', 'RSI-OLSTEC'); leg_str_avg{end+1, 1} = 'RSI-OLSTEC'; end

        hold off; grid on;
        legend(leg_str_avg, 'location', 'best', 'FontSize', 12);
        xlabel('Data Stream Index'); ylabel([metric_label ' Running Average']);
        set(gca, 'FontSize', fs);
        xlim([1, total_slices]);
        if export_results
            savefig(h2, fullfile(result_dir, 'Fig_Running_Average_Error.fig'));
        end
        fprintf('Figures saved successfully.\n');
    end

    %% 4. Quantitative table
    algorithm_results = local_empty_algorithm_table();
    if run_cpwopt && exist('sub_infos_cp_wopt', 'var')
        algorithm_results = [algorithm_results; local_algorithm_row('CP-WOPT (Batch)', elapsed_time_cpwopt, ...
            local_get_err_metric(sub_infos_cp_wopt), [], NaN, false, NaN, NaN, total_slices)];
    end
    if run_tecpsgd && exist('sub_infos_TeCPSGD', 'var')
        algorithm_results = [algorithm_results; local_algorithm_row('TeCPSGD', elapsed_time_tecpsgd, ...
            local_get_err_metric(sub_infos_TeCPSGD), local_get_run_metric(sub_infos_TeCPSGD), NaN, false, NaN, NaN, total_slices)];
    end
    if run_petrels && exist('sub_infos_petrels', 'var')
        algorithm_results = [algorithm_results; local_algorithm_row('Petrels', elapsed_time_petrels, ...
            local_get_err_metric(sub_infos_petrels), local_get_run_metric(sub_infos_petrels), NaN, false, NaN, NaN, total_slices)];
    end
    if run_grouse && exist('sub_infos_grouse', 'var')
        algorithm_results = [algorithm_results; local_algorithm_row('Grouse', elapsed_time_grouse, ...
            local_get_err_metric(sub_infos_grouse), local_get_run_metric(sub_infos_grouse), NaN, false, NaN, NaN, total_slices)];
    end
    if run_grasta && exist('sub_infos_grasta', 'var')
        algorithm_results = [algorithm_results; local_algorithm_row('Grasta', elapsed_time_grasta, ...
            local_get_err_metric(sub_infos_grasta), local_get_run_metric(sub_infos_grasta), NaN, false, NaN, NaN, total_slices)];
    end
    if run_olstec && exist('sub_infos_olstec_multi', 'var')
        for i = 1:length(lambda_list)
            algo_name_str = sprintf('OLSTEC (lam=%.2f)', lambda_list(i));
            algorithm_results = [algorithm_results; local_algorithm_row(algo_name_str, elapsed_time_olstec_multi(i), ...
                local_get_err_metric(sub_infos_olstec_multi{i}), local_get_run_metric(sub_infos_olstec_multi{i}), ...
                lambda_list(i), i == idx_display, NaN, NaN, total_slices)]; %#ok<AGROW>
        end
    end
    if run_rsi_olstec && exist('sub_infos_rsi', 'var')
        [lambda_mean, lambda_low_pct] = local_lambda_stats(sub_infos_rsi, rsi_lambda_max);
        algorithm_results = [algorithm_results; local_algorithm_row('RSI-OLSTEC (Ours)', elapsed_time_rsi, ...
            local_get_err_metric(sub_infos_rsi), local_get_run_metric(sub_infos_rsi), NaN, true, ...
            lambda_mean, lambda_low_pct, total_slices)];
    end

    fprintf('\n==========================================================================================\n');
    fprintf('QUANTITATIVE RESULTS COMPARISON\n');
    fprintf('==========================================================================================\n');
    fprintf('%-22s | %-12s | %-20s | %-20s\n', 'Algorithm', 'Time [sec]', 'Final Metric Err', 'Final Run Avg');
    fprintf('------------------------------------------------------------------------------------------\n');
    for i = 1:height(algorithm_results)
        run_avg = algorithm_results.final_metric_run_avg(i);
        if isnan(run_avg)
            run_avg_str = 'N/A';
        else
            run_avg_str = sprintf('%.6e', run_avg);
        end
        fprintf('%-22s | %-12.4f | %-20.6e | %-20s\n', ...
            algorithm_results.algorithm{i}, algorithm_results.time_sec(i), ...
            algorithm_results.final_metric_err(i), run_avg_str);
    end
    fprintf('------------------------------------------------------------------------------------------\n');

    %% 5. Optional visualization
    observe_percent = 100 * fraction;
    if image_display_flag
        h_visual = figure('Name', 'Visual Comparison', ...
            'Position', [100, 100, 1600, 800]);

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
                    local_display_images(rows, cols, observe_percent, ...
                        plot_height, plot_width, k, i, current_info, ...
                        current_name, display_intensity_limits, ...
                        display_residual_limits);
                end
                pause(0.01);
            end
            if export_results
                drawnow;
                savefig(h_visual, fullfile(result_dir, ...
                    'Fig_Running_Final.fig'));
                fprintf('Saved final dynamic comparison: Fig_Running_Final.fig\n');
            end
        else
            fprintf('No algorithms selected for visualization.\n');
        end
    end

    if make_matrix_figure
        matrix_plot_infos = struct();
        if run_grouse && exist('sub_infos_grouse', 'var'), matrix_plot_infos.grouse = sub_infos_grouse; end
        if run_tecpsgd && exist('sub_infos_TeCPSGD', 'var'), matrix_plot_infos.tecpsgd = sub_infos_TeCPSGD; end
        if run_grasta && exist('sub_infos_grasta', 'var'), matrix_plot_infos.grasta = sub_infos_grasta; end
        if run_petrels && exist('sub_infos_petrels', 'var'), matrix_plot_infos.petrels = sub_infos_petrels; end
        if run_olstec && exist('sub_infos_olstec', 'var'), matrix_plot_infos.olstec = sub_infos_olstec; end
        if run_rsi_olstec && exist('sub_infos_rsi', 'var'), matrix_plot_infos.rsi = sub_infos_rsi; end
        local_make_matrix_figure(Tensor_Y_Original, rows, cols, total_slices, ...
            matrix_plot_infos, result_dir, export_results, ...
            display_intensity_limits);
    end

    result = struct();
    result.video_filename = video_filename;
    result.meta_filename = meta_filename;
    result.rank_r = rank_r;
    result.matrix_rank = matrix_rank;
    result.fraction = fraction;
    result.random_seed = random_seed;
    result.matrix_init_seed = matrix_init_seed;
    result.tolcost = tolcost;
    result.scale_ratio = scale_ratio;
    result.max_frames = max_frames;
    result.rows = rows;
    result.cols = cols;
    result.total_slices = total_slices;
    result.aux_meta = aux_meta;
    result.aux_missing_policy = aux_missing_policy;
    result.initial_calibration_frames = initial_calibration_frames;
    result.adaptive_min_grad = adaptive_min_grad;
    result.rsi_min_grad_floor = rsi_min_grad_floor;
    result.rsi_grad_ema_alpha = rsi_grad_ema_alpha;
    result.rsi_huber_delta = rsi_huber_delta;
    result.aux_grad_stats = aux_grad_stats;
    result.huber_delta_video = huber_delta_video;
    result.huber_delta_strategy = huber_delta_strategy;
    result.display_intensity_limits = display_intensity_limits;
    result.display_residual_limits = display_residual_limits;
    result.lambda_list = lambda_list;
    result.idx_display = idx_display;
    result.algorithm_results = algorithm_results;

    if export_results
        save(fullfile(result_dir, 'R2_summary.mat'), 'result');
        writetable(algorithm_results, fullfile(result_dir, 'R2_algorithm_results.csv'));
    end

    if close_figures
        close all;
    end
end

function value = local_get(config, name, default)
    if isfield(config, name) && ~isempty(config.(name))
        value = config.(name);
    else
        value = default;
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

function out = local_empty_algorithm_table()
    algorithm = cell(0, 1);
    time_sec = zeros(0, 1);
    final_metric_err = zeros(0, 1);
    final_metric_run_avg = zeros(0, 1);
    lambda_value = zeros(0, 1);
    is_representative = false(0, 1);
    lambda_mean = zeros(0, 1);
    lambda_below_max_pct = zeros(0, 1);
    out = table(algorithm, time_sec, final_metric_err, final_metric_run_avg, ...
        lambda_value, is_representative, lambda_mean, lambda_below_max_pct);
end

function local_validate_algorithm_metrics(sub_info, algorithm_name, ...
    expected_length, validate_running_average)
    local_normalize_metric(local_get_err_metric(sub_info), ...
        algorithm_name, expected_length);
    if validate_running_average
        local_normalize_metric(local_get_run_metric(sub_info), ...
            algorithm_name, expected_length);
    end
end

function local_validate_visual_outputs(sub_info, algorithm_name, ...
    rows, cols, expected_length)

    expected_size = [rows * cols, expected_length];
    field_names = {'I', 'L', 'E'};

    for field_idx = 1:numel(field_names)
        field_name = field_names{field_idx};
        if ~isfield(sub_info, field_name)
            error('run_r2_waamvid_single:MissingVisualOutput', ...
                '%s did not return the required field %s.', ...
                algorithm_name, field_name);
        end

        value = sub_info.(field_name);
        if ~(isnumeric(value) && isreal(value) && ...
                isequal(size(value), expected_size))
            error('run_r2_waamvid_single:InvalidVisualOutput', ...
                ['%s returned an invalid %s array. Expected a real ', ...
                 'numeric array of size %d-by-%d.'], ...
                algorithm_name, field_name, ...
                expected_size(1), expected_size(2));
        end

        if any(~isfinite(value(:)))
            error('run_r2_waamvid_single:NonfiniteVisualOutput', ...
                '%s returned nonfinite values in %s.', ...
                algorithm_name, field_name);
        end
    end
end

function row = local_algorithm_row(algorithm_name, time_sec, err_metric, ...
    run_metric, lambda_value, is_representative, lambda_mean, ...
    lambda_below_max_pct, expected_length)
    algorithm = {algorithm_name};
    err_metric = local_normalize_metric(err_metric, algorithm_name, expected_length);
    final_metric_err = err_metric(end);
    if isempty(run_metric)
        final_metric_run_avg = NaN;
    else
        run_metric = local_normalize_metric(run_metric, algorithm_name, expected_length);
        final_metric_run_avg = run_metric(end);
    end
    row = table(algorithm, time_sec, final_metric_err, final_metric_run_avg, ...
        lambda_value, is_representative, lambda_mean, lambda_below_max_pct);
end

function metric = local_normalize_metric(raw_metric, algorithm_name, expected_length)
    if ~(isnumeric(raw_metric) && isreal(raw_metric) && isvector(raw_metric))
        error('run_r2_waamvid_single:InvalidMetricLayout', ...
            '%s returned a non-real or non-vector metric output.', algorithm_name);
    end
    raw_metric = reshape(raw_metric, 1, []);
    has_initial_placeholder = contains(algorithm_name, 'CP-WOPT') || ...
        strcmp(algorithm_name, 'TeCPSGD') || ...
        startsWith(algorithm_name, 'OLSTEC') || ...
        startsWith(algorithm_name, 'RSI-OLSTEC');

    if has_initial_placeholder
        if numel(raw_metric) ~= expected_length + 1
            error('run_r2_waamvid_single:InvalidMetricLayout', ...
                ['%s returned %d metric values; its documented layout requires ' ...
                 'one initial placeholder followed by %d frame values.'], ...
                algorithm_name, numel(raw_metric), expected_length);
        end
        if ~isnan(raw_metric(1))
            error('run_r2_waamvid_single:InvalidMetricLayout', ...
                '%s did not return the expected initial NaN placeholder.', ...
                algorithm_name);
        end
        metric = raw_metric(2:end);
    else
        if numel(raw_metric) ~= expected_length
            error('run_r2_waamvid_single:InvalidMetricLayout', ...
                '%s returned %d metric values; %d values were expected.', ...
                algorithm_name, numel(raw_metric), expected_length);
        end
        metric = raw_metric;
    end
    metric = validate_complete_nre(metric, expected_length, algorithm_name, 1);
end

function [lambda_mean, lambda_low_pct] = local_lambda_stats(sub_infos, lambda_max)
    if isfield(sub_infos, 'lambda_history') && ~isempty(sub_infos.lambda_history)
        lambda_vals = sub_infos.lambda_history(:);
        lambda_vals = lambda_vals(isfinite(lambda_vals));
        if isempty(lambda_vals)
            lambda_mean = NaN;
            lambda_low_pct = NaN;
        else
            lambda_mean = mean(lambda_vals);
            lambda_low_pct = 100 * mean(lambda_vals < lambda_max);
        end
    else
        lambda_mean = NaN;
        lambda_low_pct = NaN;
    end
end

function local_plot_safe_log(x, y, color, ~)
    y_plot = y;
    y_plot(y_plot <= 0) = NaN;
    valid_idx = (x > 0) & ~isnan(y_plot);
    semilogy(x(valid_idx), y_plot(valid_idx), color, 'linewidth', 2.0);
end

function linespec = local_olstec_linespec(index)
    line_specs = {':c', '--c', '-c', '-.c', ':b', '--b', '-b', '-.b', ...
        ':k', '--k', '-k', '-.k'};
    linespec = line_specs{mod(index - 1, numel(line_specs)) + 1};
end

function metric = local_get_err_metric(sub_info)
    metric = sub_info.err_residual;
end

function metric = local_get_run_metric(sub_info)
    metric = sub_info.err_run_ave;
end

function val = local_metric_at_frame(sub_info, frame)
    metric = local_get_err_metric(sub_info);
    val = NaN;
    if isfield(sub_info, 'inner_iter') && ~isempty(sub_info.inner_iter)
        idx = find(sub_info.inner_iter == frame, 1, 'first');
        if ~isempty(idx) && idx <= numel(metric)
            val = metric(idx);
            return;
        end
    end
    if numel(metric) >= frame + 1 && (isnan(metric(1)) || metric(1) == 0)
        val = metric(frame + 1);
    elseif numel(metric) >= frame
        val = metric(frame);
    end
end

function local_display_images(rows, cols, observe, height, width, ...
    test_idx, frame, sub_infos, algorithm, display_intensity_limits, ...
    display_residual_limits)

    ax = subplot(height, width, test_idx);
    imagesc(ax, reshape(sub_infos.I(:, frame), [rows, cols]));
    local_set_clim(ax, display_intensity_limits);
    colormap(ax, gray); axis(ax, 'image'); axis(ax, 'off');
    title(ax, [algorithm, ': ', num2str(observe), '% obs'], 'Interpreter', 'tex');

    ax = subplot(height, width, width + test_idx);
    imagesc(ax, reshape(sub_infos.L(:,frame), [rows cols]));
    local_set_clim(ax, display_intensity_limits);
    colormap(ax, gray); axis(ax, 'image'); axis(ax, 'off');
    title(ax, ['Low-rank: f = ', num2str(frame)]);

    ax = subplot(height, width, 2*width + test_idx);
    % E is the full-frame evaluation residual Y - L, not a sparse component.
    if ~isfield(sub_infos, 'E') || isempty(sub_infos.E) || ...
            size(sub_infos.E, 1) ~= rows * cols || ...
            size(sub_infos.E, 2) < frame
        error('run_r2_waamvid_single:InvalidResidual', ...
            'A valid full-frame residual is unavailable for %s at frame %d.', ...
            algorithm, frame);
    end
    full_residual = sub_infos.E(:, frame);
    imagesc(ax, reshape(full_residual, [rows cols]));
    local_set_clim(ax, display_residual_limits);
    colormap(ax, gray); axis(ax, 'image'); axis(ax, 'off');

    curr_err = local_metric_at_frame(sub_infos, frame);
    title(ax, sprintf('NRE = %.4f', curr_err));
    if test_idx == 1
        ylabel(ax, 'Full-frame residual', ...
            'Visible', 'on', 'FontWeight', 'bold');
    end
end

function local_make_matrix_figure(Tensor_Y_Original, rows, cols, total_slices, ...
    matrix_plot_infos, result_dir, export_results, display_intensity_limits)

    target_frames = [50, 200, 400, 600];
    target_frames = target_frames(target_frames <= total_slices);
    num_cols = length(target_frames);

    if num_cols == 0
        warning('run_r2_waamvid_single:NoDisplayFrames', ...
            'No frames available for display. Please check target_frames.');
        return;
    end

    fprintf('Generating multi-frame comparison matrix (Frames: %s)...\n', num2str(target_frames));

    algo_plot_list = {};

    if isfield(matrix_plot_infos, 'grouse')
        algo_plot_list(end+1,:) = {matrix_plot_infos.grouse, 'Grouse'};
    end
    if isfield(matrix_plot_infos, 'tecpsgd')
        algo_plot_list(end+1,:) = {matrix_plot_infos.tecpsgd, 'TeCPSGD'};
    end
    if isfield(matrix_plot_infos, 'grasta')
        algo_plot_list(end+1,:) = {matrix_plot_infos.grasta, 'Grasta'};
    end
    if isfield(matrix_plot_infos, 'petrels')
        algo_plot_list(end+1,:) = {matrix_plot_infos.petrels, 'Petrels'};
    end
    if isfield(matrix_plot_infos, 'olstec')
        algo_plot_list(end+1,:) = {matrix_plot_infos.olstec, 'OLSTEC'};
    end
    if isfield(matrix_plot_infos, 'rsi')
        algo_plot_list(end+1,:) = {matrix_plot_infos.rsi, 'RSI-OLSTEC'};
    end

    num_algos = size(algo_plot_list, 1);
    num_rows = num_algos + 1;

    fig_height = 200 * num_rows;
    fig_width  = 200 * num_cols;
    h_matrix = figure('Name', 'Multi-Frame Comparison', 'Position', [50, 50, fig_width, fig_height]);

    use_tiled = exist('tiledlayout', 'file') == 2 || exist('tiledlayout', 'builtin') == 5;
    if use_tiled
        tiledlayout(num_rows, num_cols, 'TileSpacing', 'none', 'Padding', 'compact');
    end

    for j = 1:num_cols
        frame_idx = target_frames(j);

        if use_tiled
            ax = nexttile;
        else
            ax = subplot(num_rows, num_cols, j);
        end

        imagesc(ax, Tensor_Y_Original(:, :, frame_idx));
        local_set_clim(ax, display_intensity_limits);
        colormap(ax, gray); axis(ax, 'image'); axis(ax, 'off');

        title(ax, ['Frame ' num2str(frame_idx)], 'FontSize', 12, 'FontWeight', 'bold');

        if j == 1
            hY = ylabel(ax, 'Original', 'FontSize', 12, ...
                'FontWeight', 'bold', 'Color', 'k');
            set(hY, 'Visible', 'on');
        end
    end

    for i = 1:num_algos
        algo_data = algo_plot_list{i, 1};
        algo_name = algo_plot_list{i, 2};

        for j = 1:num_cols
            frame_idx = target_frames(j);

            if use_tiled
                ax = nexttile;
            else
                curr_row_idx = i + 1;
                subplot_idx = (curr_row_idx - 1) * num_cols + j;
                ax = subplot(num_rows, num_cols, subplot_idx);
            end

            L_frame = reshape(algo_data.L(:, frame_idx), [rows, cols]);

            imagesc(ax, L_frame);
            local_set_clim(ax, display_intensity_limits);
            colormap(ax, gray); axis(ax, 'image'); axis(ax, 'off');

            if j == 1
                hY = ylabel(ax, algo_name, 'FontSize', 12, ...
                    'FontWeight', 'bold', 'Color', 'k');
                set(hY, 'Visible', 'on');
            end
        end
    end

    if export_results
        savefig(h_matrix, fullfile(result_dir, 'Fig_MultiFrame_Matrix_R2.fig'));
    end
    fprintf('Saved multi-frame comparison: Fig_MultiFrame_Matrix_R2.fig\n');
end

function local_set_clim(ax, limits)
    if exist('clim', 'file') == 2 || exist('clim', 'builtin') == 5
        clim(ax, limits);
    else
        set(ax, 'CLim', limits);
    end
end
