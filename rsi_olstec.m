function [Xsol, infos, sub_infos] = rsi_olstec(A_in, Omega_in, Gamma_in, tensor_dims, rank, X_init, options, Aux_Signal)
% RSI_OLSTEC: Robust Side-Informed Online Low-rank Subspace Tracking
%
% =========================================================================
% ACKNOWLEDGEMENT & ATTRIBUTION:
% This code is a modified and extended version of the original OLSTEC algorithm.
%
% Original Author: Hiroyuki Kasai
% Original Paper:  H. Kasai, "Online low-rank tensor subspace tracking from
%                  incomplete data by CP decomposition using recursive least
%                  squares," 2016 IEEE International Conference on Acoustics,
%                  Speech and Signal Processing (ICASSP), 2016, pp. 2519-2523.
%                  DOI: 10.1109/ICASSP.2016.7472131
% Original Code:   https://github.com/hiroyuki-kasai/OLSTEC
%
% MODIFICATIONS IN THIS VERSION (RSI-OLSTEC):
% 1. Introduced Huber loss penalty for robust outlier (spatter) suppression.
% 2. Integrated physical side-information (Aux_Signal) for physics-guided
%    adaptive lambda computation.
% 3. Modified the RLS update steps to incorporate robust weights, utilizing
%    robust linear solves (\) instead of standard SMW updates to maintain
%    numerical stability for small rank sizes.
% =========================================================================
%
% Inputs:
%   A_in: Input tensor/video stream (Rows x Cols x Time)
%   Omega_in: Observation mask (Logical matrix)
%   Gamma_in: Test mask (Optional)
%   tensor_dims: Dimensions of the tensor
%   rank: CP decomposition rank (R)
%   X_init: Initialization structure (contains A, B, C)
%   options: Parameter structure. Relevant optional fields include
%            irls_max_iters (default: 3), irls_tolerance (default: 1e-3),
%            and normalization_epsilon (default: 1e-3). Setting
%            huber_delta = Inf disables robust downweighting; setting
%            lambda_min = lambda_max gives a fixed forgetting factor.
%            Early stopping requires store_subinfo = true. When
%            store_subinfo = false, per-frame error and cost trajectories
%            are empty; adaptive-memory and IRLS histories remain available.
%   Aux_Signal: Auxiliary physical signal (e.g., melt pool width)
%
% Outputs:
%   Xsol: Final decomposition result
%   infos: Global statistical information
%   sub_infos: Frame-by-frame tracking details

    %% 1. Parameter Initialization and Configuration
    if nargin < 7, options = struct; end
    if nargin < 8, Aux_Signal = zeros(tensor_dims(3), 1); end

    lambda_max  = get_option(options, 'lambda_max', 0.80);
    lambda_min  = get_option(options, 'lambda_min', 0.70);
    huber_delta = get_option(options, 'huber_delta', 0.3);
    mu          = get_option(options, 'mu', 0.01);
    min_grad_threshold = get_option(options, 'min_grad_threshold', 0.7);
    grad_ema_alpha = get_option(options, 'grad_ema_alpha', 0.999);
    irls_max_iters = get_option(options, 'irls_max_iters', 3);
    irls_tolerance = get_option(options, 'irls_tolerance', 1e-3);
    normalization_epsilon = get_option(options, 'normalization_epsilon', 1e-3);
    tolcost     = get_option(options, 'tolcost', 1e-12);
    early_stop_on = normalize_early_stop_mode( ...
        get_option(options, 'early_stop_on', 'none'));
    verbose     = get_option(options, 'verbose', 1);
    store_matrix = get_option(options, 'store_matrix', true);
    store_subinfo = get_option(options, 'store_subinfo', true);
    irls_norm_floor = 1e-10;

    if ~store_subinfo && ~strcmp(early_stop_on, 'none')
        error('rsi_olstec:EarlyStopRequiresSubinfo', ...
            ['early_stop_on requires store_subinfo=true because the ' ...
             'selected stopping cost is not evaluated otherwise.']);
    end

    I = tensor_dims(1);
    J = tensor_dims(2);
    slice_length = tensor_dims(3);

    if ~(isnumeric(lambda_min) && isreal(lambda_min) && isscalar(lambda_min) && ...
            isfinite(lambda_min) && lambda_min > 0 && lambda_min <= 1)
        error('rsi_olstec:InvalidLambdaMin', ...
            'lambda_min must be a finite real scalar in the interval (0, 1].');
    end
    if ~(isnumeric(lambda_max) && isreal(lambda_max) && isscalar(lambda_max) && ...
            isfinite(lambda_max) && lambda_max > 0 && lambda_max <= 1)
        error('rsi_olstec:InvalidLambdaMax', ...
            'lambda_max must be a finite real scalar in the interval (0, 1].');
    end
    if lambda_min > lambda_max
        error('rsi_olstec:InvalidLambdaRange', ...
            'lambda_min must not exceed lambda_max.');
    end
    if ~(isnumeric(huber_delta) && isreal(huber_delta) && ...
            isscalar(huber_delta) && ~isnan(huber_delta) && huber_delta > 0)
        error('rsi_olstec:InvalidHuberDelta', ...
            'huber_delta must be a positive real scalar; Inf selects squared loss.');
    end
    if ~(isnumeric(mu) && isreal(mu) && isscalar(mu) && isfinite(mu) && mu > 0)
        error('rsi_olstec:InvalidRegularization', ...
            'mu must be a positive finite real scalar.');
    end
    if ~(isnumeric(min_grad_threshold) && isreal(min_grad_threshold) && ...
            isscalar(min_grad_threshold) && ~isnan(min_grad_threshold) && ...
            min_grad_threshold >= 0)
        error('rsi_olstec:InvalidGradientThreshold', ...
            ['min_grad_threshold must be a nonnegative real scalar; Inf ' ...
             'disables side-information activation.']);
    end
    if ~(isnumeric(grad_ema_alpha) && isreal(grad_ema_alpha) && ...
            isscalar(grad_ema_alpha) && isfinite(grad_ema_alpha) && ...
            grad_ema_alpha >= 0 && grad_ema_alpha < 1)
        error('rsi_olstec:InvalidEnvelopeDecay', ...
            'grad_ema_alpha must be a finite real scalar in the interval [0, 1).');
    end
    if ~(isnumeric(irls_max_iters) && isreal(irls_max_iters) && ...
            isscalar(irls_max_iters) && isfinite(irls_max_iters) && ...
            irls_max_iters >= 1 && irls_max_iters == floor(irls_max_iters))
        error('rsi_olstec:InvalidIRLSIterations', ...
            'irls_max_iters must be a positive integer.');
    end
    if ~(isnumeric(irls_tolerance) && isreal(irls_tolerance) && ...
            isscalar(irls_tolerance) && isfinite(irls_tolerance) && ...
            irls_tolerance > 0)
        error('rsi_olstec:InvalidIRLSTolerance', ...
            'irls_tolerance must be a positive finite real scalar.');
    end
    if ~(isnumeric(normalization_epsilon) && isreal(normalization_epsilon) && ...
            isscalar(normalization_epsilon) && isfinite(normalization_epsilon) && ...
            normalization_epsilon > 0)
        error('rsi_olstec:InvalidNormalizationEpsilon', ...
            'normalization_epsilon must be a positive finite real scalar.');
    end
    if isempty(Aux_Signal)
        Aux_Signal = zeros(slice_length, 1);
    else
        Aux_Signal = Aux_Signal(:);
        if length(Aux_Signal) < slice_length
            error('rsi_olstec:InvalidAuxSignal', 'Aux_Signal must contain at least tensor_dims(3) entries.');
        end
    end

    A_Omega = Omega_in .* A_in;
    if ~isempty(Gamma_in)
        A_Gamma = Gamma_in .* A_in;
        has_test_mask = true;
    else
        A_Gamma = [];
        has_test_mask = false;
    end

    % Auxiliary signal preprocessing (strict causal filling).
    % The first value cannot be forward-filled without using an artificial
    % value, so callers must trim or validate the auxiliary signal first.
    if isnan(Aux_Signal(1))
        error('rsi_olstec:InvalidAuxSignal', ...
            ['Aux_Signal starts with NaN. Provide a valid first side-information ' ...
             'sample or trim the sequence to the first causally available value.']);
    end
    % Manual forward fill for R2016b compatibility (fillmissing requires R2017a+)
    for i = 2:length(Aux_Signal)
        if isnan(Aux_Signal(i))
            Aux_Signal(i) = Aux_Signal(i-1);
        end
    end
    if any(~isfinite(Aux_Signal(1:slice_length)))
        error('rsi_olstec:InvalidAuxSignal', ...
            'Aux_Signal must be finite after causal forward filling.');
    end

    %% 2. Initialization of Factor Matrices and RLS Covariance
    if isempty(X_init)
        A_t0 = randn(I, rank);
        B_t0 = randn(J, rank);
        C_t0 = randn(slice_length, rank);
    else
        A_t0 = X_init.A;
        B_t0 = X_init.B;
        C_t0 = X_init.C;
    end

    % Initialize RLS information matrices
    RA_info = cell(I, 1);
    for i = 1:I, RA_info{i} = mu * eye(rank); end
    RB_info = cell(J, 1);
    for j = 1:J, RB_info{j} = mu * eye(rank); end

    %% 3. Initialize Costs and Monitoring Arrays
    Rec_init = zeros(I, J, slice_length);
    for k = 1:slice_length
        gamma_init = C_t0(k,:)';
        Rec_init(:,:,k) = A_t0 * diag(gamma_init) * B_t0';
    end
    init_train_cost = compute_cost_tensor(Rec_init, Omega_in, A_Omega, tensor_dims);

    if ~isempty(Gamma_in) && ~isempty(A_Gamma)
        init_test_cost = compute_cost_tensor(Rec_init, Gamma_in, A_Gamma, tensor_dims);
    else
        init_test_cost = 0;
    end
    % Reuse the initial reconstruction buffer for later cost evaluations.
    Rec_temp = Rec_init;
    clear Rec_init;

    infos.iter = 0;
    infos.train_cost = init_train_cost;
    infos.test_cost = init_test_cost;
    if store_subinfo
        infos.causal_train_cost = 0;
        infos.causal_test_cost = 0;
    else
        infos.causal_train_cost = NaN;
        infos.causal_test_cost = NaN;
    end
    infos.early_stop_on = early_stop_on;
    infos.time = 0;

    if store_subinfo
        sub_infos.inner_iter = zeros(1, slice_length + 1);
        sub_infos.err_residual = zeros(1, slice_length + 1);
        sub_infos.err_residual_legacy = zeros(1, slice_length + 1);
        sub_infos.err_run_ave = zeros(1, slice_length + 1);
        sub_infos.err_observed = zeros(1, slice_length + 1);
        sub_infos.err_observed_run_ave = zeros(1, slice_length + 1);
        if has_test_mask
            sub_infos.err_test = zeros(1, slice_length + 1);
            sub_infos.err_test_run_ave = zeros(1, slice_length + 1);
        else
            sub_infos.err_test = [];
            sub_infos.err_test_run_ave = [];
        end
        sub_infos.global_train_cost = zeros(1, slice_length + 1);
        sub_infos.global_test_cost = zeros(1, slice_length + 1);
        sub_infos.causal_train_cost = zeros(1, slice_length + 1);
        sub_infos.causal_test_cost = zeros(1, slice_length + 1);

        sub_infos.inner_iter(1) = 0;
        sub_infos.err_residual(1) = NaN;
        sub_infos.err_residual_legacy(1) = NaN;
        sub_infos.err_run_ave(1) = NaN;
        sub_infos.err_observed(1) = NaN;
        sub_infos.err_observed_run_ave(1) = NaN;
        if has_test_mask
            sub_infos.err_test(1) = NaN;
            sub_infos.err_test_run_ave(1) = NaN;
        end
        sub_infos.global_train_cost(1) = init_train_cost;
        sub_infos.global_test_cost(1) = init_test_cost;
        sub_infos.causal_train_cost(1) = 0;
        sub_infos.causal_test_cost(1) = 0;
    else
        sub_infos.inner_iter = [];
        sub_infos.err_residual = [];
        sub_infos.err_residual_legacy = [];
        sub_infos.err_run_ave = [];
        sub_infos.err_observed = [];
        sub_infos.err_observed_run_ave = [];
        sub_infos.err_test = [];
        sub_infos.err_test_run_ave = [];
        sub_infos.global_train_cost = [];
        sub_infos.global_test_cost = [];
        sub_infos.causal_train_cost = [];
        sub_infos.causal_test_cost = [];
    end

    if store_matrix
        sub_infos.I = zeros(I * J, slice_length);
        sub_infos.L = zeros(I * J, slice_length);
        sub_infos.E = zeros(I * J, slice_length);
    end

    prev_aux_val = Aux_Signal(1);
    min_gap = normalization_epsilon;
    max_grad_seen = min_grad_threshold + min_gap;
    lambda_history = zeros(slice_length, 1);
    phys_grad_history = zeros(slice_length, 1);
    norm_grad_history = zeros(slice_length, 1);
    max_grad_history = zeros(slice_length, 1);
    prior_irls_iterations = zeros(slice_length, 1);
    posterior_irls_iterations = zeros(slice_length, 1);
    if store_subinfo
        test_error_sum = 0;
        test_error_count = 0;
    end

    if verbose > 0
        fprintf('Starting RSI-OLSTEC (Robust Side-Informed Online Mode)...\n');
        fprintf('Initial Cost: Train %7.3e, Test %7.3e\n', init_train_cost, init_test_cost);
    end

    %% 4. Main Loop: Frame-by-Frame Online Tracking
    t_begin = tic;

    for k = 1:slice_length
        % --- A. Physics-Guided Adaptive Lambda Computation ---
        curr_aux_val = Aux_Signal(k);
        phys_grad = abs(curr_aux_val - prev_aux_val);

        if phys_grad > max_grad_seen
            max_grad_seen = phys_grad;
        else
            max_grad_seen = grad_ema_alpha * max_grad_seen + (1 - grad_ema_alpha) * phys_grad;
        end

        max_grad_seen = max(max_grad_seen, min_grad_threshold + min_gap);
        denom = max_grad_seen - min_grad_threshold;

        if phys_grad <= min_grad_threshold
            norm_grad = 0.0;
        else
            norm_grad = (phys_grad - min_grad_threshold) / denom;
            norm_grad = max(0, min(1, norm_grad));
        end

        lambda_curr = lambda_max - (lambda_max - lambda_min) * norm_grad;
        lambda_curr = min(lambda_max, max(lambda_min, lambda_curr));
        lambda_history(k) = lambda_curr;
        phys_grad_history(k) = phys_grad;
        norm_grad_history(k) = norm_grad;
        max_grad_history(k) = max_grad_seen;
        prev_aux_val = curr_aux_val;

        % --- B. Data Preparation ---
        y_slice = A_in(:, :, k);
        omega_slice = logical(Omega_in(:, :, k));

        obs_indices = find(omega_slice);
        [ii, jj] = ind2sub([I, J], obs_indices);
        y_vec = y_slice(obs_indices);

        % --- C. Core Algorithm Steps ---
        % Step 1: Robust Estimation of Temporal Factor Gamma (IRLS)
        prior_irls_count = 0;
        if isempty(obs_indices)
            if k > 1
                gamma_vec = C_t0(k-1, :)';
            else
                gamma_vec = C_t0(k, :)';
            end
        else
            H_k = A_t0(ii, :) .* B_t0(jj, :);
            % SVD stable solve for ridge regression
            [U_svd, S_svd, V_svd] = svd(H_k, 'econ');
            s = diag(S_svd);
            gamma_vec = V_svd * ((s ./ (s.^2 + mu)) .* (U_svd' * y_vec));

            for irls_iter = 1:irls_max_iters
                prior_irls_count = irls_iter;
                y_pred = H_k * gamma_vec;
                residuals = abs(y_vec - y_pred);

                weights = ones(size(residuals));
                outlier_mask = residuals > huber_delta;
                weights(outlier_mask) = huber_delta ./ residuals(outlier_mask);

                % SVD stable solve for weighted ridge regression
                sqrt_weights = sqrt(weights);
                H_w = bsxfun(@times, sqrt_weights, H_k);
                [U_svd, S_svd, V_svd] = svd(H_w, 'econ');
                s = diag(S_svd);
                gamma_new = V_svd * ((s ./ (s.^2 + mu)) .* (U_svd' * (sqrt_weights .* y_vec)));

                if norm(gamma_new - gamma_vec) / ...
                        (norm(gamma_vec) + irls_norm_floor) < irls_tolerance
                    gamma_vec = gamma_new;
                    break;
                end
                gamma_vec = gamma_new;
            end
        end
        prior_irls_iterations(k) = prior_irls_count;
        C_t0(k, :) = gamma_vec';

        % Step 2: Robust Update of Spatial Factor A (Robust RLS)
        for i = 1:I
            idx = find(omega_slice(i, :));
            if isempty(idx)
                R_old = RA_info{i};
                Info_mat = lambda_curr * R_old + (mu - lambda_curr * mu) * eye(rank);
                RA_info{i} = Info_mat;
                A_t0(i, :) = A_t0(i, :) - (mu - lambda_curr * mu) * (Info_mat \ A_t0(i, :)')';
                continue;
            end

            U = bsxfun(@times, gamma_vec', B_t0(idx, :));
            y_i = y_slice(i, idx)';

            pred_i = U * A_t0(i, :)';
            err_i = y_i - pred_i;

            w_vec = ones(size(err_i));
            mask_i = abs(err_i) > huber_delta;
            w_vec(mask_i) = huber_delta ./ abs(err_i(mask_i));

            Weighted_U = bsxfun(@times, w_vec, U);

            R_old = RA_info{i};
            Info_mat = lambda_curr * R_old + U' * Weighted_U + (mu - lambda_curr * mu) * eye(rank);
            RA_info{i} = Info_mat;

            grad = U' * (w_vec .* err_i);
            rhs_vec = grad - (mu - lambda_curr * mu) * A_t0(i, :)';
            update_step = Info_mat \ rhs_vec;
            A_t0(i, :) = A_t0(i, :) + update_step';
        end

        % Step 3: Robust Update of Spatial Factor B
        for j = 1:J
            idx = find(omega_slice(:, j));
            if isempty(idx)
                R_old_B = RB_info{j};
                Info_mat_B = lambda_curr * R_old_B + (mu - lambda_curr * mu) * eye(rank);
                RB_info{j} = Info_mat_B;
                B_t0(j, :) = B_t0(j, :) - (mu - lambda_curr * mu) * (Info_mat_B \ B_t0(j, :)')';
                continue;
            end

            V = bsxfun(@times, gamma_vec', A_t0(idx, :));
            y_j = y_slice(idx, j);

            pred_j = V * B_t0(j, :)';
            err_j = y_j - pred_j;

            w_vec_col = ones(size(err_j));
            mask_j = abs(err_j) > huber_delta;
            w_vec_col(mask_j) = huber_delta ./ abs(err_j(mask_j));

            Weighted_V = bsxfun(@times, w_vec_col, V);
            R_old_B = RB_info{j};
            Info_mat_B = lambda_curr * R_old_B + V' * Weighted_V + (mu - lambda_curr * mu) * eye(rank);
            RB_info{j} = Info_mat_B;

            grad_B = V' * (w_vec_col .* err_j);
            rhs_vec_B = grad_B - (mu - lambda_curr * mu) * B_t0(j, :)';
            update_step_B = Info_mat_B \ rhs_vec_B;
            B_t0(j, :) = B_t0(j, :) + update_step_B';
        end

        % --- Supplementary Step: Re-estimate Posterior Gamma ---
        posterior_irls_count = 0;
        if isempty(obs_indices)
            gamma_post = gamma_vec;
        else
            H_k_post = A_t0(ii, :) .* B_t0(jj, :);
            % SVD stable solve for ridge regression
            [U_svd, S_svd, V_svd] = svd(H_k_post, 'econ');
            s = diag(S_svd);
            gamma_post = V_svd * ((s ./ (s.^2 + mu)) .* (U_svd' * y_vec));

            for irls_iter = 1:irls_max_iters
                posterior_irls_count = irls_iter;
                y_pred_post = H_k_post * gamma_post;
                residuals_post = abs(y_vec - y_pred_post);

                weights_post = ones(size(residuals_post));
                outlier_mask_post = residuals_post > huber_delta;
                weights_post(outlier_mask_post) = huber_delta ./ residuals_post(outlier_mask_post);

                % SVD stable solve for weighted ridge regression
                sqrt_weights_post = sqrt(weights_post);
                H_w_post = bsxfun(@times, sqrt_weights_post, H_k_post);
                [U_svd, S_svd, V_svd] = svd(H_w_post, 'econ');
                s = diag(S_svd);
                gamma_new_post = V_svd * ((s ./ (s.^2 + mu)) .* (U_svd' * (sqrt_weights_post .* y_vec)));

                if norm(gamma_new_post - gamma_post) / ...
                        (norm(gamma_post) + irls_norm_floor) < irls_tolerance
                    gamma_post = gamma_new_post;
                    break;
                end
                gamma_post = gamma_new_post;
            end
        end
        posterior_irls_iterations(k) = posterior_irls_count;
        gamma_vec = gamma_post;
        C_t0(k, :) = gamma_vec';

        % Reconstruct the current frame for optional output and diagnostics.
        X_rec_slice = A_t0 * diag(gamma_vec) * B_t0';
        if store_subinfo || verbose > 1
            norm_residual = norm(y_slice(:) - X_rec_slice(:));
            norm_I = norm(y_slice(:));
            if norm_I > 0
                nre = norm_residual / norm_I;
            else
                nre = norm_residual;
            end
        end

        if store_subinfo
            obs_residual = y_slice(omega_slice) - X_rec_slice(omega_slice);
            obs_norm = norm(y_slice(omega_slice));
            if obs_norm > 0
                observed_nre = norm(obs_residual) / obs_norm;
            else
                observed_nre = norm(obs_residual);
            end

            if has_test_mask
                test_mask = logical(Gamma_in(:, :, k));
                if any(test_mask(:))
                    test_residual = y_slice(test_mask) - X_rec_slice(test_mask);
                    test_norm = norm(y_slice(test_mask));
                    if test_norm > 0
                        test_nre = norm(test_residual) / test_norm;
                    else
                        test_nre = norm(test_residual);
                    end
                else
                    test_nre = NaN;
                end
            end

            sub_infos.inner_iter(k+1) = k;
            sub_infos.err_residual(k+1) = nre;
            sub_infos.err_residual_legacy(k+1) = nre;
            sub_infos.err_observed(k+1) = observed_nre;
            if has_test_mask
                sub_infos.err_test(k+1) = test_nre;
            end

            if k == 1
                run_error = nre;
                observed_run_error = observed_nre;
            else
                run_error = ...
                    (sub_infos.err_run_ave(k) * (k-1) + nre) / k;
                observed_run_error = ...
                    (sub_infos.err_observed_run_ave(k) * (k-1) + ...
                    observed_nre) / k;
            end
            sub_infos.err_run_ave(k+1) = run_error;
            sub_infos.err_observed_run_ave(k+1) = observed_run_error;
            if has_test_mask
                if ~isnan(test_nre)
                    test_error_sum = test_error_sum + test_nre;
                    test_error_count = test_error_count + 1;
                end
                if test_error_count > 0
                    sub_infos.err_test_run_ave(k+1) = ...
                        test_error_sum / test_error_count;
                else
                    sub_infos.err_test_run_ave(k+1) = NaN;
                end
            end
        end

        if store_matrix
            sub_infos.I(:, k) = y_slice(:) .* omega_slice(:);
            sub_infos.L(:, k) = X_rec_slice(:);
            sub_infos.E(:, k) = y_slice(:) - X_rec_slice(:);
        end

        if store_subinfo
            for f = 1:slice_length
                g_f = C_t0(f,:)';
                Rec_temp(:,:,f) = A_t0 * diag(g_f) * B_t0';
            end
            g_train_cost = compute_cost_tensor(Rec_temp, Omega_in, A_Omega, tensor_dims);

            if ~isempty(Gamma_in) && ~isempty(A_Gamma)
                g_test_cost = compute_cost_tensor(Rec_temp, Gamma_in, A_Gamma, tensor_dims);
            else
                g_test_cost = 0;
            end

            causal_dims = [I, J, k];
            causal_train_cost = compute_cost_tensor(Rec_temp(:, :, 1:k), ...
                Omega_in(:, :, 1:k), A_Omega(:, :, 1:k), causal_dims);
            if ~isempty(Gamma_in) && ~isempty(A_Gamma)
                causal_test_cost = compute_cost_tensor(Rec_temp(:, :, 1:k), ...
                    Gamma_in(:, :, 1:k), A_Gamma(:, :, 1:k), causal_dims);
            else
                causal_test_cost = 0;
            end

            sub_infos.global_train_cost(k+1) = g_train_cost;
            sub_infos.global_test_cost(k+1)  = g_test_cost;
            sub_infos.causal_train_cost(k+1) = causal_train_cost;
            sub_infos.causal_test_cost(k+1)  = causal_test_cost;

            stop_cost = select_early_stop_cost(early_stop_on, g_train_cost, causal_train_cost);
            if stop_cost < tolcost
                if verbose > 0
                    fprintf('RSI-OLSTEC: %s cost %7.3e < tolcost at frame %d. Stopping early.\n', early_stop_on, stop_cost, k);
                end
                break;
            end
        end

        if verbose > 1 && mod(k, 50) == 0
            fprintf('Iter %d: Lambda=%.3f, Err=%.4f\n', k, lambda_curr, nre);
        end
    end

    % Mark unfilled frames as NaN to prevent artificial flat-line illusion
    if store_subinfo && k < slice_length
        sub_infos.err_residual(k+2:end) = NaN;
        sub_infos.err_residual_legacy(k+2:end) = NaN;
        sub_infos.err_run_ave(k+2:end) = NaN;
        sub_infos.err_observed(k+2:end) = NaN;
        sub_infos.err_observed_run_ave(k+2:end) = NaN;
        if has_test_mask
            sub_infos.err_test(k+2:end) = NaN;
            sub_infos.err_test_run_ave(k+2:end) = NaN;
        end
        sub_infos.global_train_cost(k+2:end) = NaN;
        sub_infos.global_test_cost(k+2:end) = NaN;
        sub_infos.causal_train_cost(k+2:end) = NaN;
        sub_infos.causal_test_cost(k+2:end) = NaN;
        sub_infos.inner_iter(k+2:end) = NaN;
        lambda_history(k+1:end) = NaN;
        phys_grad_history(k+1:end) = NaN;
        norm_grad_history(k+1:end) = NaN;
        max_grad_history(k+1:end) = NaN;
        prior_irls_iterations(k+1:end) = NaN;
        posterior_irls_iterations(k+1:end) = NaN;
        if store_matrix
            sub_infos.I(:, k+1:end) = NaN;
            sub_infos.L(:, k+1:end) = NaN;
            sub_infos.E(:, k+1:end) = NaN;
        end
    end

    total_time = toc(t_begin);

    %% 5. Output Encapsulation
    Xsol.A = A_t0;
    Xsol.B = B_t0;
    Xsol.C = C_t0;

    if ~store_subinfo
        for f = 1:slice_length
            g_f = C_t0(f,:)';
            Rec_temp(:,:,f) = A_t0 * diag(g_f) * B_t0';
        end
        final_train_cost = compute_cost_tensor(Rec_temp, Omega_in, A_Omega, tensor_dims);
        if ~isempty(Gamma_in) && ~isempty(A_Gamma)
            final_test_cost = compute_cost_tensor(Rec_temp, Gamma_in, A_Gamma, tensor_dims);
        else
            final_test_cost = 0;
        end
        final_causal_train_cost = NaN;
        final_causal_test_cost = NaN;
    else
        final_train_cost = sub_infos.global_train_cost(k+1);
        final_test_cost = sub_infos.global_test_cost(k+1);
        final_causal_train_cost = sub_infos.causal_train_cost(k+1);
        final_causal_test_cost = sub_infos.causal_test_cost(k+1);
    end

    infos.iter = [infos.iter; k];
    infos.train_cost = [infos.train_cost; final_train_cost];
    infos.test_cost = [infos.test_cost; final_test_cost];
    infos.causal_train_cost = [infos.causal_train_cost; final_causal_train_cost];
    infos.causal_test_cost = [infos.causal_test_cost; final_causal_test_cost];
    infos.time = [infos.time; total_time];

    sub_infos.lambda_history = lambda_history;
    sub_infos.phys_grad_history = phys_grad_history;
    sub_infos.norm_grad_history = norm_grad_history;
    sub_infos.max_grad_history = max_grad_history;
    sub_infos.min_grad_threshold = min_grad_threshold;
    sub_infos.grad_ema_alpha = grad_ema_alpha;
    sub_infos.lambda_min = lambda_min;
    sub_infos.lambda_max = lambda_max;
    sub_infos.huber_delta = huber_delta;
    sub_infos.mu = mu;
    sub_infos.irls_max_iters = irls_max_iters;
    sub_infos.irls_tolerance = irls_tolerance;
    sub_infos.normalization_epsilon = normalization_epsilon;
    sub_infos.prior_irls_iterations = prior_irls_iterations;
    sub_infos.posterior_irls_iterations = posterior_irls_iterations;

    if verbose > 0
        fprintf('RSI-OLSTEC Completed in %.3fs. Final Cost: %7.3e\n', total_time, final_train_cost);
    end
end

function val = get_option(opts, name, default)
    if isfield(opts, name)
        val = opts.(name);
    else
        val = default;
    end
end

function mode = normalize_early_stop_mode(mode)
    if ~(ischar(mode) || (isstring(mode) && isscalar(mode)))
        error('rsi_olstec:InvalidEarlyStopMode', ...
            'early_stop_on must be none, causal_train, or global_train.');
    end

    mode = lower(strtrim(char(mode)));
    switch mode
        case {'none', 'off', 'false'}
            mode = 'none';
        case {'causal', 'causal_train', 'causal_train_cost'}
            mode = 'causal_train';
        case {'global', 'global_train', 'global_train_cost', 'legacy'}
            mode = 'global_train';
        otherwise
            error('rsi_olstec:InvalidEarlyStopMode', ...
                'early_stop_on must be none, causal_train, or global_train.');
    end
end

function stop_cost = select_early_stop_cost(mode, global_train_cost, causal_train_cost)
    mode = lower(char(mode));
    switch mode
        case {'none', 'off', 'false'}
            stop_cost = Inf;
        case {'causal', 'causal_train', 'causal_train_cost'}
            stop_cost = causal_train_cost;
        case {'global', 'global_train', 'global_train_cost', 'legacy'}
            stop_cost = global_train_cost;
        otherwise
            error('rsi_olstec:InvalidEarlyStopMode', ...
                'early_stop_on must be none, causal_train, or global_train.');
    end
end
