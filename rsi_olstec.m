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
% 3. Modified the RLS update steps to incorporate robust weights and avoid 
%    explicit matrix inversion.
% =========================================================================
%
% Inputs:
%   A_in: Input tensor/video stream (Rows x Cols x Time)
%   Omega_in: Observation mask (Logical matrix)
%   Gamma_in: Test mask (Optional)
%   tensor_dims: Dimensions of the tensor
%   rank: CP decomposition rank (R)
%   X_init: Initialization structure (contains A, B, C)
%   options: Parameter structure
%   Aux_Signal: Auxiliary physical signal (e.g., melt pool width)
%
% Outputs:
%   Xsol: Final decomposition result
%   infos: Global statistical information
%   sub_infos: Frame-by-frame tracking details

    %% 1. Parameter Initialization and Configuration
    if nargin < 7, options = struct; end
    if nargin < 8, Aux_Signal = zeros(tensor_dims(3), 1); end
    
    lambda_max  = get_option(options, 'lambda_max', 0.90); 
    lambda_min  = get_option(options, 'lambda_min', 0.70); 
    huber_delta = get_option(options, 'huber_delta', 0.3); 
    mu          = get_option(options, 'mu', 0.01);
    min_grad_threshold = get_option(options, 'min_grad_threshold', 0.7);
    verbose     = get_option(options, 'verbose', 1);
    store_matrix = get_option(options, 'store_matrix', true);
    store_subinfo = get_option(options, 'store_subinfo', true);
    
    I = tensor_dims(1);
    J = tensor_dims(2);
    slice_length = tensor_dims(3);
    
    A_Omega = Omega_in .* A_in;
    if ~isempty(Gamma_in)
        A_Gamma = Gamma_in .* A_in;
    else 
        A_Gamma = [];
    end
    
    % Auxiliary signal preprocessing (Strict causal filling)
    if isnan(Aux_Signal(1))
        Aux_Signal(1) = 0; 
    end
    Aux_Signal = fillmissing(Aux_Signal, 'previous');
    
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
    
    infos.iter = 0;
    infos.train_cost = init_train_cost;
    infos.test_cost = init_test_cost;
    infos.time = 0;
    
    sub_infos.inner_iter = zeros(1, slice_length + 1);
    sub_infos.err_residual = zeros(1, slice_length + 1);
    sub_infos.err_run_ave = zeros(1, slice_length + 1);
    sub_infos.global_train_cost = zeros(1, slice_length + 1);
    sub_infos.global_test_cost = zeros(1, slice_length + 1);
    
    sub_infos.inner_iter(1) = 0;
    sub_infos.err_residual(1) = 0;
    sub_infos.err_run_ave(1) = 0;
    sub_infos.global_train_cost(1) = 0;
    sub_infos.global_test_cost(1) = 0;
    
    if store_matrix
        sub_infos.I = zeros(I * J, slice_length); 
        sub_infos.L = zeros(I * J, slice_length); 
        sub_infos.E = zeros(I * J, slice_length); 
    end
    
    prev_aux_val = Aux_Signal(1); 
    max_grad_seen = 1e-6; 
    lambda_history = zeros(slice_length, 1);
    
    if verbose > 0
        fprintf('Starting RSI-OLSTEC (Robust Side-Informed Online Mode)...\n');
        fprintf('Initial Cost: Train %7.3e, Test %7.3e\n', init_train_cost, init_test_cost);
    end
    if store_subinfo
        Rec_temp = zeros(I, J, slice_length); 
    end
    
    %% 4. Main Loop: Frame-by-Frame Online Tracking
    t_begin = tic;
    
    for k = 1:slice_length
        % --- A. Physics-Guided Adaptive Lambda Computation ---
        curr_aux_val = Aux_Signal(k);
        phys_grad = abs(curr_aux_val - prev_aux_val);
        min_gap = 1e-3; 
        
        if phys_grad > max_grad_seen
            max_grad_seen = phys_grad;
        else
            max_grad_seen = 0.999 * max_grad_seen + 0.001 * phys_grad;
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
        prev_aux_val = curr_aux_val;
        
        % --- B. Data Preparation ---
        y_slice = A_in(:, :, k);
        omega_slice = Omega_in(:, :, k);
        
        obs_indices = find(omega_slice);
        [ii, jj] = ind2sub([I, J], obs_indices);
        y_vec = y_slice(obs_indices);
        
        % --- C. Core Algorithm Steps ---
        % Step 1: Robust Estimation of Temporal Factor Gamma (IRLS)
        H_k = A_t0(ii, :) .* B_t0(jj, :);
        gamma = (H_k' * H_k + mu * eye(rank)) \ (H_k' * y_vec);
        
        for irls_iter = 1:3
            y_pred = H_k * gamma;
            residuals = abs(y_vec - y_pred);
            
            weights = ones(size(residuals));
            outlier_mask = residuals > huber_delta;
            weights(outlier_mask) = huber_delta ./ residuals(outlier_mask);
            Weighted_H = bsxfun(@times, weights, H_k);
            
            gamma = (H_k' * Weighted_H + mu * eye(rank)) \ (H_k' * (weights .* y_vec));
        end
        C_t0(k, :) = gamma'; 
        
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
            
            U = bsxfun(@times, gamma', B_t0(idx, :)); 
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
            
            V = bsxfun(@times, gamma', A_t0(idx, :));
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
        H_k_post = A_t0(ii, :) .* B_t0(jj, :);
        gamma_post = (H_k_post' * H_k_post + mu * eye(rank)) \ (H_k_post' * y_vec);
        
        for irls_iter = 1:3
            y_pred_post = H_k_post * gamma_post;
            residuals_post = abs(y_vec - y_pred_post);
            
            weights_post = ones(size(residuals_post));
            outlier_mask_post = residuals_post > huber_delta;
            weights_post(outlier_mask_post) = huber_delta ./ residuals_post(outlier_mask_post);
            Weighted_H_post = bsxfun(@times, weights_post, H_k_post);
            
            gamma_post = (H_k_post' * Weighted_H_post + mu * eye(rank)) \ (H_k_post' * (weights_post .* y_vec));
        end
        gamma = gamma_post;    
        C_t0(k, :) = gamma';   
        
        % --- Compute Errors and Store Logs ---
        X_rec_slice = A_t0 * diag(gamma) * B_t0';
        norm_residual = norm(y_slice(:) - X_rec_slice(:));
        norm_I = norm(y_slice(:));
        
        if norm_I > 0
            error = norm_residual / norm_I;
        else
            error = 0;
        end
        
        sub_infos.inner_iter(k+1)   = k;            
        sub_infos.err_residual(k+1) = error;
        
        if k == 1
            run_error = error;
        else
            run_error = (sub_infos.err_run_ave(k) * (k-1) + error) / k;
        end
        sub_infos.err_run_ave(k+1) = run_error;
        
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
            
            sub_infos.global_train_cost(k+1) = g_train_cost; 
            sub_infos.global_test_cost(k+1)  = g_test_cost;
        end
        
        if verbose > 1 && mod(k, 50) == 0
            fprintf('Iter %d: Lambda=%.3f, Err=%.4f\n', k, lambda_curr, error);
        end
    end
    
    total_time = toc(t_begin);
    
    %% 5. Output Encapsulation
    Xsol.A = A_t0;
    Xsol.B = B_t0;
    Xsol.C = C_t0;
    
    if ~store_subinfo
        Rec_temp = zeros(I, J, slice_length);
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
    else
        final_train_cost = sub_infos.global_train_cost(end);
        final_test_cost = sub_infos.global_test_cost(end);
    end
    
    infos.iter = [infos.iter; 1]; 
    infos.train_cost = [infos.train_cost; final_train_cost];
    infos.test_cost = [infos.test_cost; final_test_cost];
    infos.time = [infos.time; total_time];
    
    sub_infos.err_run_avg = sub_infos.err_run_ave;
    sub_infos.lambda_history = lambda_history;
    
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