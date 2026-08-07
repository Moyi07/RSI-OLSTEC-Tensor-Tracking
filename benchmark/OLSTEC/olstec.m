function [Xsol, infos, sub_infos] = olstec(A_in, Omega_in, Gamma_in, tensor_dims, rank, X_init, options)
% OLSTEC algorithm.
%
% Inputs:
%       A_in            full tensor data to be tracked.
%       Omega_in        logical data of traing tensor set to speficy observable/missing elements.
%       Gamma_in        logical data of test tensor set to speficy observable/missing elements.
%       tensor_dims     dimension of tensor.
%       rank            max rank.
%       xinit           initial tensor data.
%       options         structure data of options.
% Output:
%       XSol            solution.
%       infos           information.
%       sub_infos       sub information.
%
% Reference:
%       H.Kasai,
%       "Online low-rank tensor subspace tracking from incomplete data by CP decomposition using recursive least squares,"
%       IEEE International conference on Acoustics, Speech and Signal Processing (ICASSP), 2016.
%
%
% This file is part of OLSTEC package.
%
% Created by H.Kasai on June 07, 2017
% Modified by H.Kasai on Sep. 12, 2017


    % extract options
    if ~isfield(options, 'maxepochs')
        maxepochs = 1;
    else
        maxepochs = options.maxepochs;
    end

    if ~isfield(options, 'tolcost')
        tolcost = 1e-12;
    else
        tolcost = options.tolcost;
    end

    if ~isfield(options, 'early_stop_on')
        early_stop_on = 'none';
    else
        early_stop_on = options.early_stop_on;
    end

    if ~isfield(options, 'permute_on')
        permute_on = false;
    else
        permute_on = options.permute_on;
    end

    if ~isfield(options, 'lambda')
        lambda = 0.7;
    else
        lambda = options.lambda;
    end

    if ~isfield(options, 'mu')
        mu = 0.1;
    else
        mu = options.mu;
    end

    if ~isfield(options, 'tw_flag')
        TW_Flag = false;
    else
        TW_Flag = options.tw_flag;
    end

    if ~isfield(options, 'tw_len')
        TW_LEN = 10;
    else
        TW_LEN = options.tw_len;
    end

    if ~isfield(options, 'store_subinfo')
        store_subinfo = true;
    else
        store_subinfo = options.store_subinfo;
    end

    if ~isfield(options, 'store_matrix')
        store_matrix = false;
    else
        store_matrix = options.store_matrix;
    end

    if ~isfield(options, 'verbose')
        verbose = 2;
    else
        verbose = options.verbose;
    end


    % set tensors
    A               = A_in;             % Full entries
    Omega           = Omega_in;         % Training set 'Omega'
    Gamma           = Gamma_in;         % Test set 'Gamma'
    has_test_mask   = ~isempty(Gamma_in);

    A_Omega         = Omega_in.*A_in;   % Training entries i.e., Omega_in.*A_in
    if ~isempty(Gamma_in)
        A_Gamma         = Gamma_in.*A_in;   % Test entries i.e., Gamma_in.*A_in
    else
        A_Gamma     = [];
    end


    % set tensor dimentions
    rows            = tensor_dims(1);
    cols            = tensor_dims(2);
    slice_length    = tensor_dims(3);


    % initialize X (A_t0 and B_t0) if needed
    if isempty(X_init)
        A_t0 = randn(tensor_dims(1), rank);
        B_t0 = randn(tensor_dims(2), rank);
        C_t0 = randn(tensor_dims(3), rank);
    else
        A_t0 = X_init.A;
        B_t0 = X_init.B;
        C_t0 = X_init.C;
    end


    % prepare Rinv histroy buffers
    RAinv = repmat(100*eye(rank), rows, 1);
    RBinv = repmat(100*eye(rank), cols, 1);

    % prepare
    N_AlphaAlphaT = zeros(rank*rows, rank*(TW_LEN+1));
    N_BetaBetaT   = zeros(rank*cols, rank*(TW_LEN+1));

    % prepare
    N_AlphaResi = zeros(rank*rows, TW_LEN+1);
    N_BetaResi  = zeros(rank*cols, TW_LEN+1);


    % calculate initial cost
    Rec = zeros(rows, cols, slice_length);
    for k=1:slice_length
        gamma = C_t0(k,:)';
        Rec(:,:,k) = A_t0 * diag(gamma) * B_t0';
    end
    train_cost = compute_cost_tensor(Rec, Omega, A_Omega, tensor_dims);
    if ~isempty(Gamma) && ~isempty(A_Gamma)
        test_cost = compute_cost_tensor(Rec, Gamma, A_Gamma, tensor_dims);
    else
        test_cost = 0;
    end


    % initialize infos
    infos.iter = 0;
    infos.train_cost = train_cost;
    infos.test_cost = test_cost;
    infos.causal_train_cost = 0;
    infos.causal_test_cost = 0;
    infos.early_stop_on = early_stop_on;
    infos.time = 0;


    % initialize sub_infos
    sub_infos.inner_iter = 0;
    sub_infos.err_residual = NaN;
    sub_infos.err_residual_legacy = NaN;
    sub_infos.err_run_ave = NaN;
    sub_infos.err_observed = NaN;
    sub_infos.err_observed_run_ave = NaN;
    if has_test_mask
        sub_infos.err_test = NaN;
        sub_infos.err_test_run_ave = NaN;
    else
        sub_infos.err_test = [];
        sub_infos.err_test_run_ave = [];
    end
    sub_infos.global_train_cost = train_cost;
    sub_infos.global_test_cost = test_cost;
    sub_infos.causal_train_cost = 0;
    sub_infos.causal_test_cost = 0;
    if store_matrix
        sub_infos.I = zeros(rows * cols, slice_length);
        sub_infos.L = zeros(rows * cols, slice_length);
        sub_infos.E = zeros(rows * cols, slice_length);
    end

    if verbose > 1
        fprintf('OLSTEC Epoch 000, Cost %7.3e, Cost(test) %7.3e\n', train_cost, test_cost);
    end

    test_error_sum = 0;
    test_error_count = 0;

    % Main loop
    for outiter = 1 : maxepochs

        % permute samples
        if permute_on
            col_order = randperm(slice_length);
        else
            col_order = 1:slice_length;
        end
        processed_slices = false(1, slice_length);

        % Begin the time counter for the epoch
        t_begin = tic();

        for k=1:slice_length

            % Pull out the relevant indices and revealed entries for this column
            % sampled original image
            I_mat = A(:,:, col_order(k));
            Omega_mat   = Omega(:,:, col_order(k));
            I_mat_Omega = A_Omega(:,:, col_order(k));
            processed_slices(col_order(k)) = true;


            %% Gamma (B) Update
            temp3 = 0;
            temp4 = 0;
            for m=1:rows
                alpha_remat = repmat(A_t0(m,:)', 1, cols);
                alpha_beta = alpha_remat .* B_t0';
                I_row = I_mat_Omega(m,:);
                temp3 = temp3 + alpha_beta * I_row';

                omega_row = logical(Omega_mat(m,:));
                alpha_beta_Omega = alpha_beta(:, omega_row);
                temp4 = temp4 + alpha_beta_Omega * (alpha_beta_Omega');
            end

            temp4 = lambda * eye(rank) + temp4;
            gamma = temp4 \ temp3;                                             % equation (18)


            %% update A
            for m=1:rows

                Omega_mat_ind = find(Omega_mat(m,:));
                I_row = I_mat_Omega(m,:);
                I_row_Omega = I_row(Omega_mat_ind);
                C_t0_Omega = B_t0(Omega_mat_ind,:);
                N_alpha_Omega = diag(gamma) * C_t0_Omega';
                N_alpha_alpha_t_Omega = N_alpha_Omega * N_alpha_Omega';

                % Calc TAinv (i.e. RAinv)
                TAinv = lambda^(-1) * RAinv((m-1)*rank+1:m*rank,:);
                if TW_Flag
                    Oldest_alpha_alpha_t = N_AlphaAlphaT((m-1)*rank+1:m*rank,1:rank);
                    TAinv = inv(inv(TAinv) + N_alpha_alpha_t_Omega + (mu - lambda*mu)*eye(rank) - lambda^TW_LEN * Oldest_alpha_alpha_t);
                else
                    TAinv = inv(inv(TAinv) + N_alpha_alpha_t_Omega + (mu - lambda*mu)*eye(rank));
                end

                % Calc delta A_t0(m,:)
                recX_col_Omega = N_alpha_Omega' * A_t0(m,:)';
                resi_col_Omega = I_row_Omega' - recX_col_Omega;
                N_alpha_Resi_Omega = N_alpha_Omega * diag(resi_col_Omega);

                N_resi_Rt_alpha = TAinv * N_alpha_Resi_Omega;
                delta_A_t0_m = sum(N_resi_Rt_alpha,2);

                % Update A
                if TW_Flag
                    % update A
                    Oldest_alpha_resi = N_AlphaResi((m-1)*rank+1:m*rank,1)';
                    %A_t1(m,:) = A_t0(m,:) + delta_A_t0_m' - lambda^TW_LEN * Oldest_alpha_resi;
                    A_t1(m,:) = A_t0(m,:)  - (mu - lambda*mu) * A_t0(m,:) * TAinv' + delta_A_t0_m' - lambda^TW_LEN * Oldest_alpha_resi;

                    % Store data
                    N_AlphaAlphaT((m-1)*rank+1:m*rank,TW_LEN*rank+1:(TW_LEN+1)*rank) = N_alpha_alpha_t_Omega;
                    N_AlphaResi((m-1)*rank+1:m*rank,TW_LEN+1) = sum(N_alpha_Resi_Omega,2);
                else
                    %A_t1(m,:) = A_t0(m,:) + delta_A_t0_m';
                    %A_t1(m,:) = A_t0(m,:) - (mu - lambda*mu) * (TAinv * A_t0(m,:)')' + delta_A_t0_m';
                    A_t1(m,:) = A_t0(m,:) - (mu - lambda*mu) * A_t0(m,:) * TAinv' + delta_A_t0_m';
                end

                % Store RAinv
                RAinv((m-1)*rank+1:m*rank,:) = TAinv;
            end

            % Final update of A
            A_t0 = A_t1;


            %% update B
            for n=1:cols

                Omega_mat_ind = find(Omega_mat(:,n));
                I_col = I_mat_Omega(:,n);
                I_col_Omega = I_col(Omega_mat_ind);
                A_t0_Omega = A_t0(Omega_mat_ind,:);
                N_beta_Omega = A_t0_Omega * diag(gamma);
                N_beta_beta_t_Omega = N_beta_Omega' * N_beta_Omega;

                % Calc TBinv (i.e. RBinv)
                TBinv = lambda^(-1) * RBinv((n-1)*rank+1:n*rank,:);
                if TW_Flag
                    Oldest_beta_beta_t = N_BetaBetaT((n-1)*rank+1:n*rank,1:rank);
                    TBinv = inv(inv(TBinv) + N_beta_beta_t_Omega + (mu - lambda*mu)*eye(rank) - lambda^TW_LEN * Oldest_beta_beta_t);
                else
                    TBinv = inv(inv(TBinv) + N_beta_beta_t_Omega + (mu - lambda*mu)*eye(rank));
                end

                % Calc delta B_t0(n,:)
                recX_col_Omega = B_t0(n,:) * N_beta_Omega';
                resi_col_Omega = I_col_Omega' - recX_col_Omega;
                N_beta_Resi_Omega = N_beta_Omega' * diag(resi_col_Omega);
                N_resi_Rt_beta = TBinv * N_beta_Resi_Omega;
                delta_C_t0_n = sum(N_resi_Rt_beta,2);

                if TW_Flag
                    % Upddate B
                    Oldest_beta_resi = N_BetaResi((n-1)*rank+1:n*rank,1)';
                    B_t1(n,:) = B_t0(n,:) - (mu - lambda*mu) * B_t0(n,:) * TBinv' + delta_C_t0_n' - lambda^TW_LEN * Oldest_beta_resi;

                    % Store data
                    N_BetaBetaT((n-1)*rank+1:n*rank,TW_LEN*rank+1:(TW_LEN+1)*rank) = N_beta_beta_t_Omega;
                    N_BetaResi((n-1)*rank+1:n*rank,TW_LEN+1) = sum(N_beta_Resi_Omega,2);
                else
                    B_t1(n,:) = B_t0(n,:) - (mu - lambda*mu) * B_t0(n,:) * TBinv' + delta_C_t0_n';

                end

                % Store RBinv
                RBinv((n-1)*rank+1:n*rank,:) = TBinv;
            end


            if TW_Flag
                N_AlphaAlphaT(:,1:rank) = [];
                N_BetaBetaT(:,1:rank) = [];
                N_AlphaResi(:,1) = [];
                N_BetaResi(:,1) = [];
            end

            % Final update of B
            B_t0 = B_t1;


            %% Reculculate gamma (B)
            temp3 = 0;
            temp4 = 0;
            for m=1:rows
                alpha_remat = repmat(A_t0(m,:)', 1, cols);
                alpha_beta = alpha_remat .* B_t0';
                I_row = I_mat_Omega(m,:);
                temp3 = temp3 + alpha_beta * I_row';

                omega_row = logical(Omega_mat(m,:));
                alpha_beta_Omega = alpha_beta(:, omega_row);
                temp4 = temp4 + alpha_beta_Omega * alpha_beta_Omega';
            end
            temp4 = lambda * eye(rank) + temp4;
            gamma = temp4 \ temp3;

            % Store gamma into C_t0
            C_t0(col_order(k),:) = gamma';

            % Reconstruct Low-rank Matrix
            L_rec = A_t0 * diag(gamma) * B_t0';

            if store_subinfo
                % Residual Error
                norm_residual   = norm(I_mat(:) - L_rec(:));
                norm_I          = norm(I_mat(:));
                if norm_I > 0
                    error = norm_residual / norm_I;
                else
                    error = 0;
                end
                obs_idx = logical(Omega_mat);
                obs_residual = I_mat(obs_idx) - L_rec(obs_idx);
                obs_norm = norm(I_mat(obs_idx));
                if obs_norm > 0
                    observed_error = norm(obs_residual) / obs_norm;
                else
                    observed_error = norm(obs_residual);
                end
                if has_test_mask
                    test_idx = logical(Gamma(:, :, col_order(k)));
                    if any(test_idx(:))
                        test_residual = I_mat(test_idx) - L_rec(test_idx);
                        test_norm = norm(I_mat(test_idx));
                        if test_norm > 0
                            test_error = norm(test_residual) / test_norm;
                        else
                            test_error = norm(test_residual);
                        end
                    else
                        test_error = NaN;
                    end
                end
                sub_infos.inner_iter    = [sub_infos.inner_iter (outiter-1)*slice_length+k];
                sub_infos.err_residual    = [sub_infos.err_residual error];
                sub_infos.err_residual_legacy = [sub_infos.err_residual_legacy error];
                sub_infos.err_observed = [sub_infos.err_observed observed_error];
                if has_test_mask
                    sub_infos.err_test = [sub_infos.err_test test_error];
                end

                % Running-average Estimation Error
                if k == 1
                    run_error   = error;
                    observed_run_error = observed_error;
                else
                    run_error   = (sub_infos.err_run_ave(end) * (k-1) + error)/k;
                    observed_run_error = (sub_infos.err_observed_run_ave(end) * (k-1) + observed_error)/k;
                end
                sub_infos.err_run_ave     = [sub_infos.err_run_ave run_error];
                sub_infos.err_observed_run_ave = [sub_infos.err_observed_run_ave observed_run_error];
                if has_test_mask
                    if ~isnan(test_error)
                        test_error_sum = test_error_sum + test_error;
                        test_error_count = test_error_count + 1;
                    end
                    if test_error_count > 0
                        sub_infos.err_test_run_ave = [sub_infos.err_test_run_ave test_error_sum / test_error_count];
                    else
                        sub_infos.err_test_run_ave = [sub_infos.err_test_run_ave NaN];
                    end
                end

                % Store reconstruction error
                if store_matrix
                    E_rec = I_mat - L_rec;
                    sub_infos.I(:,col_order(k)) = I_mat_Omega(:);
                    sub_infos.L(:,col_order(k)) = L_rec(:);
                    sub_infos.E(:,col_order(k)) = E_rec(:);
                end

                for f=1:slice_length
                    gamma = C_t0(f,:)';
                    Rec(:,:,f) = A_t0 * diag(gamma) * B_t0';
                end

                % Global train_cost computation
                train_cost = compute_cost_tensor(Rec, Omega, A_Omega, tensor_dims);
                if ~isempty(Gamma) && ~isempty(A_Gamma)
                    test_cost = compute_cost_tensor(Rec, Gamma, A_Gamma, tensor_dims);
                else
                    test_cost = 0;
                end
                causal_dims = [rows, cols, sum(processed_slices)];
                causal_train_cost = compute_cost_tensor(Rec(:, :, processed_slices), ...
                    Omega(:, :, processed_slices), A_Omega(:, :, processed_slices), causal_dims);
                if ~isempty(Gamma) && ~isempty(A_Gamma)
                    causal_test_cost = compute_cost_tensor(Rec(:, :, processed_slices), ...
                        Gamma(:, :, processed_slices), A_Gamma(:, :, processed_slices), causal_dims);
                else
                    causal_test_cost = 0;
                end
                sub_infos.global_train_cost  = [sub_infos.global_train_cost train_cost];
                sub_infos.global_test_cost  = [sub_infos.global_test_cost test_cost];
                sub_infos.causal_train_cost = [sub_infos.causal_train_cost causal_train_cost];
                sub_infos.causal_test_cost = [sub_infos.causal_test_cost causal_test_cost];


                if verbose > 2
                    fnum = (outiter-1)*slice_length + k;
                    fprintf('OLSTEC: fnum = %03d, cost = %e, error = %e\n', fnum, train_cost, error);
                end
            end

        end


        % store infos
        infos.iter = [infos.iter; outiter];
        infos.time = [infos.time; infos.time(end) + toc(t_begin)];

        if ~store_subinfo
            for f=1:slice_length
                gamma = C_t0(f,:)';
                Rec(:,:,f) = A_t0 * diag(gamma) * B_t0';
            end

            train_cost = compute_cost_tensor(Rec, Omega, A_Omega, tensor_dims);
            if ~isempty(Gamma) && ~isempty(A_Gamma)
                test_cost = compute_cost_tensor(Rec, Gamma, A_Gamma, tensor_dims);
            else
                test_cost = 0;
            end
            causal_train_cost = train_cost;
            causal_test_cost = test_cost;
        end
        infos.train_cost = [infos.train_cost; train_cost];
        infos.test_cost = [infos.test_cost; test_cost];
        infos.causal_train_cost = [infos.causal_train_cost; causal_train_cost];
        infos.causal_test_cost = [infos.causal_test_cost; causal_test_cost];

        if verbose > 1
            fprintf('OLSTEC Epoch %0.3d, Cost %7.3e, Cost(test) %7.3e\n', outiter, train_cost, test_cost);
        end

        % stopping criteria: cost tolerance reached
        stop_cost = olstec_select_early_stop_cost(early_stop_on, train_cost, causal_train_cost);
        if stop_cost < tolcost
            fprintf('train_cost sufficiently decreased.\n');
            break;
        end
    end

    Xsol.A = A_t0;
    Xsol.B = B_t0;
    Xsol.C = C_t0;
end

function stop_cost = olstec_select_early_stop_cost(mode, global_train_cost, causal_train_cost)
    mode = lower(char(mode));
    switch mode
        case {'none', 'off', 'false'}
            stop_cost = Inf;
        case {'causal', 'causal_train', 'causal_train_cost'}
            stop_cost = causal_train_cost;
        case {'global', 'global_train', 'global_train_cost', 'legacy'}
            stop_cost = global_train_cost;
        otherwise
            error('olstec:InvalidEarlyStopMode', ...
                'early_stop_on must be none, causal_train, or global_train.');
    end
end
