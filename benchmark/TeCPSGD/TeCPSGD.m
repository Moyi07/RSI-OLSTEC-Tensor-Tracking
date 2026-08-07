function [Xsol, infos, sub_infos] = TeCPSGD(A_in, Omega_in, Gamma_in, tensor_dims, rank, xinit, options)
% TeSPSGD algorithm.
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
%       M. Mardani, G. Mateos, and G.B. Giannakis,
%       "Subspace learning and imputation for streaming big data matrices and tensors,"
%       IEEE Transactions on Signal Processing, vol. 63, no. 10, pp. 266-2677, 2015.
%
%
% This file is part of OLSTEC package.
%
% Created by H.Kasai on June 07, 2017


    A               = A_in;             % Full entries
    Omega           = Omega_in;         % Training set 'Omega'
    Gamma           = Gamma_in;         % Test set 'Gamma'
    has_test_mask   = ~isempty(Gamma_in);

    %
    A_Omega         = Omega_in.*A_in;   % Training entries i.e., Omega_in.*A_in
    if ~isempty(Gamma_in)
        A_Gamma         = Gamma_in.*A_in;   % Test entries i.e., Gamma_in.*A_in
    else
        A_Gamma     = [];
    end

    if isempty(xinit)
        A_t0 = randn(tensor_dims(1), rank);
        B_t0 = randn(tensor_dims(2), rank);
        C_t0 = randn(tensor_dims(3), rank);
    else
        A_t0 = xinit.A;
        B_t0 = xinit.B;
        C_t0 = xinit.C;
    end


    % set tensor size
    rows            = tensor_dims(1);
    cols            = tensor_dims(2);
    slice_length    = tensor_dims(3);


    % set options
    lambda          = options.lambda;
    stepsize_init   = options.stepsize;
    maxepochs       = options.maxepochs;
    tolcost         = options.tolcost;
    if ~isfield(options, 'early_stop_on')
        early_stop_on = 'none';
    else
        early_stop_on = options.early_stop_on;
    end
    store_subinfo   = options.store_subinfo;
    store_matrix    = options.store_matrix;
    verbose         = options.verbose;

    if ~isfield(options, 'permute_on')
        permute_on = 1;
    else
        permute_on = options.permute_on;
    end


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

    % set parameters
    eta = 0;

    if verbose > 0
        fprintf('TeCPSGD [%d] Epoch 000, Cost %7.3e, Cost(test) %7.3e, Stepsize %7.3e\n', stepsize_init, train_cost, test_cost, eta);
    end

    test_error_sum = 0;
    test_error_count = 0;

    % main loop
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

            fnum = (outiter - 1) * slice_length + k;

            % sampled original image
            I_mat = A(:,:,col_order(k));
            Omega_mat = Omega(:,:,col_order(k));
            I_mat_Omega = Omega_mat .* I_mat;
            processed_slices(col_order(k)) = true;

            % Reculculate gamma (C)
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
            gamma = temp4 \ temp3;                                             % equation (18)

            L_rec = A_t0 * diag(gamma) * B_t0';
            diff = Omega_mat.*(I_mat - L_rec);

            eta = stepsize_init/(1+lambda*stepsize_init*fnum);
            A_t1 = (1 - lambda*eta) * A_t0 + eta * diff *  B_t0 * diag(gamma);   % equation (20)&(21)
            B_t1 = (1 - lambda*eta) * B_t0 + eta * diff' * A_t0 * diag(gamma);  % equation (20)&(22)

            % Reculculate weights
            %weights = pinv(A_t1) * I_mat_Omega * pinv(B_t1');
            %t = diag(weights);

            % Update of A and B
            A_t0 = A_t1;
            B_t0 = B_t1;

            % Reculculate gamma (C)
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
            gamma = temp4 \ temp3;                                             % equation (18)

            % Store gamma into C_t0
            C_t0(col_order(k),:) = gamma';

            % Reconstruct Low-rank Matrix
            L_rec = A_t0 * diag(gamma) * B_t0';
%             if disp_flag
%                 L{alg_idx} = [L{alg_idx} L_rec(:)];
%             end

            if store_matrix
                E_rec = I_mat - L_rec;
                %sub_infos.E = [sub_infos.E E_rec(:)];
                sub_infos.I(:,col_order(k)) = I_mat_Omega(:);
                sub_infos.L(:,col_order(k)) = L_rec(:);
                sub_infos.E(:,col_order(k)) = E_rec(:);
            end

            if store_subinfo
                % Residual Error
                norm_residual   = norm(I_mat(:) - L_rec(:));
                norm_I          = norm(I_mat(:));
                if norm_I > 0
                    error = norm_residual / norm_I;
                else
                    error = norm_residual;
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

                if verbose > 1
                    fnum = (outiter-1)*slice_length + k;
                    fprintf('TeCPSGD: fnum = %03d, cost = %e, error = %e\n', fnum, train_cost, error);
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

        if verbose > 0
            fprintf('TeCPSGD [%d] Epoch %0.3d, Cost %7.3e, Cost(test) %7.3e, Stepsize %7.3e\n', stepsize_init, outiter, train_cost, test_cost, eta);
        end

        % stopping criteria: cost tolerance reached
        stop_cost = tecpsgd_select_early_stop_cost(early_stop_on, train_cost, causal_train_cost);
        if stop_cost < tolcost
            fprintf('train_cost sufficiently decreased.\n');
            break;
        end
    end




% Once we have settled on our column space, a single pass over the data
% suffices to compute the weights associated with each column.  You only
% need to compute these weights if you want to make predictions about these
% columns.
% fprintf('Find column weights...');
% R = zeros(numc,maxrank);
% for k=1:numc,
%     % Pull out the relevant indices and revealed entries for this column
%     idx = find(Indicator(:,k));
%     v_Omega = values(idx,k);
%     U_Omega = U(idx,:);
%     % solve a simple least squares problem to populate R
%     R(k,:) = (U_Omega\v_Omega)';
% end

    Xsol.A = A_t0;
    Xsol.B = B_t0;
    Xsol.C = C_t0;
end

function stop_cost = tecpsgd_select_early_stop_cost(mode, global_train_cost, causal_train_cost)
    mode = lower(char(mode));
    switch mode
        case {'none', 'off', 'false'}
            stop_cost = Inf;
        case {'causal', 'causal_train', 'causal_train_cost'}
            stop_cost = causal_train_cost;
        case {'global', 'global_train', 'global_train_cost', 'legacy'}
            stop_cost = global_train_cost;
        otherwise
            error('TeCPSGD:InvalidEarlyStopMode', ...
                'early_stop_on must be none, causal_train, or global_train.');
    end
end
