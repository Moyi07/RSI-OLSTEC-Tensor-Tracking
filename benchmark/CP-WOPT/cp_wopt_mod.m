function [Xsol, infos, sub_infos] = cp_wopt_mod(A_in, Omega_in, Gamma_in, tensor_dims, rank, xinit, options)
% Interface file for CP-WOPT algorithm
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
%                   
% This file is part of OLSTEC package.
%
% Created by H.Kasai on June 07, 2017


    A               = A_in;
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
    store_subinfo   = options.store_subinfo;
    store_matrix    = options.store_matrix; 
    verbose         = options.verbose;
    
    
    % set an example problem with missing data
    X = tensor(Omega .* A(:,:,1:slice_length));
    P = tensor(Omega);

    if isempty(xinit)
        % Create initial guess using 'nvecs'
        M_init = create_guess('Data', X, 'Num_Factors', rank, 'Factor_Generator', 'nvecs');
    else
        M_init = cell(3,1);
        M_init{1} = xinit.A;
        M_init{2} = xinit.B;       
        M_init{3} = xinit.C;          
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
    infos.iter = 1;
    infos.train_cost = train_cost;
    infos.test_cost = test_cost;
    infos.causal_train_cost = train_cost;
    infos.causal_test_cost = test_cost;
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
    sub_infos.causal_train_cost = train_cost;
    sub_infos.causal_test_cost = test_cost;

    if store_matrix
        sub_infos.I = zeros(rows*cols, slice_length);
        sub_infos.L = zeros(rows*cols, slice_length);
        sub_infos.E = zeros(rows*cols, slice_length);
    else
        sub_infos.E = [];
    end
    test_error_sum = 0;
    test_error_count = 0;


    % set up the optimization parameters
    % Get the defaults
    ncg_opts = ncg('defaults');
    % Tighten the stop tolerance (norm of gradient). This is often too large.
    ncg_opts.StopTol = 1.0e-6;
    % Tighten relative change in function value tolerance. This is often too large.
    if isfield(options, 'tolcost')
        ncg_opts.RelFuncTol = options.tolcost;
    else
        ncg_opts.RelFuncTol = 1.0e-9;
    end
    % Increase the number of iterations.
    %ncg_opts.MaxIters = 3*10^2;
    ncg_opts.MaxIters = options.maxepochs;
    % Only display every 10th iteration
    %ncg_opts.DisplayIters = 10;
    ncg_opts.DisplayIters = options.display_iters;
    % Display the final set of options
    %ncg_opts

    
    % Begin the time counter for the epoch
    t_begin = tic();
       
    % Main routine
    [M, ~, ~] = cp_wopt(X, P, rank, 'init', M_init, 'alg', 'ncg', 'alg_options', ncg_opts);
    
    L_rec_all = double(full(M));
    train_cost = compute_cost_tensor(L_rec_all, Omega, A_Omega, tensor_dims);
    if ~isempty(Gamma) && ~isempty(A_Gamma)
        test_cost = compute_cost_tensor(L_rec_all, Gamma, A_Gamma, tensor_dims);
    else
        test_cost = 0;
    end

    if store_subinfo
        for fnum= 1 : slice_length
            % Extract a noiseless original slice
            I_mat_Noiseless = A(:,:,fnum);
            Omega_mat = logical(Omega(:,:,fnum));

            % Extract a reconstructed slice
            L_rec = L_rec_all(:,:,fnum);

    %         if disp_flag
    %             L{alg_idx} = [L{alg_idx} L_rec(:)];
    %         end

            norm_residual   = norm(I_mat_Noiseless(:) - L_rec(:));
            norm_I          = norm(I_mat_Noiseless(:));
            if norm_I > 0
                error = norm_residual / norm_I;
            else
                error = 0;
            end

            obs_residual = I_mat_Noiseless(Omega_mat) - L_rec(Omega_mat);
            obs_norm = norm(I_mat_Noiseless(Omega_mat));
            if obs_norm > 0
                observed_error = norm(obs_residual) / obs_norm;
            else
                observed_error = norm(obs_residual);
            end
            if has_test_mask
                test_mask = logical(Gamma(:,:,fnum));
                if any(test_mask(:))
                    test_residual = I_mat_Noiseless(test_mask) - L_rec(test_mask);
                    test_norm = norm(I_mat_Noiseless(test_mask));
                    if test_norm > 0
                        test_error = norm(test_residual) / test_norm;
                    else
                        test_error = norm(test_residual);
                    end
                else
                    test_error = NaN;
                end
            end

            sub_infos.inner_iter    = [sub_infos.inner_iter fnum];
            sub_infos.err_residual  = [sub_infos.err_residual error];
            sub_infos.err_residual_legacy = [sub_infos.err_residual_legacy error];
            sub_infos.err_observed = [sub_infos.err_observed observed_error];
            if has_test_mask
                sub_infos.err_test = [sub_infos.err_test test_error];
            end

            % Running-average Estimation Error
            if fnum == 1
                run_error   = error;
                observed_run_error = observed_error;
            else
                run_error   = (sub_infos.err_run_ave(end) * (fnum-1) + error)/fnum;
                observed_run_error = (sub_infos.err_observed_run_ave(end) * (fnum-1) + observed_error)/fnum;
            end
            sub_infos.err_run_ave = [sub_infos.err_run_ave run_error];
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

            if store_matrix
                E_rec = I_mat_Noiseless - L_rec;
                sub_infos.I(:, fnum) = I_mat_Noiseless(:) .* Omega_mat(:);
                sub_infos.L(:, fnum) = L_rec(:);
                sub_infos.E(:, fnum) = E_rec(:);
            end

            sub_infos.global_train_cost = [sub_infos.global_train_cost train_cost];
            sub_infos.global_test_cost = [sub_infos.global_test_cost test_cost];
            sub_infos.causal_train_cost = [sub_infos.causal_train_cost train_cost];
            sub_infos.causal_test_cost = [sub_infos.causal_test_cost test_cost];

            if verbose > 1
                fprintf('CP-WOPT: fnum = %03d, error = %e\n', fnum, error);
            end
        end
    end
    
    
    % store infos
    infos.iter = [infos.iter; 2];
    infos.time = [infos.time; infos.time(end) + toc(t_begin)];        

    infos.train_cost = [infos.train_cost; train_cost];
    infos.test_cost = [infos.test_cost; test_cost];
    infos.causal_train_cost = [infos.causal_train_cost; train_cost];
    infos.causal_test_cost = [infos.causal_test_cost; test_cost];
    

    if verbose > 1 && store_subinfo
        fprintf('CP-WOPT: fnum = %03d, error = %e\n', fnum, error);
    end    

    Xsol.A = M.U{1};
    Xsol.B = M.U{2};
    Xsol.C = M.U{3} * diag(M.lambda);
end
