function err_curve = compute_true_nre_tensor(X_true, L_out)
%COMPUTE_TRUE_NRE_TENSOR Per-frame NRE against an external clean reference.
%   err_curve = compute_true_nre_tensor(X_true, L_out) computes
%   ||L_t - X_t||_F / ||X_t||_F for each frame t. X_true must be a 3-D clean
%   reference tensor. L_out may be either an I-by-J-by-T tensor or an
%   (I*J)-by-T matrix of reconstructed frames.

    if ndims(X_true) ~= 3
        error('compute_true_nre_tensor:InvalidReference', ...
            'X_true must be a 3-D tensor.');
    end

    [I, J, T] = size(X_true);
    if ndims(L_out) == 3
        if ~isequal(size(L_out), size(X_true))
            error('compute_true_nre_tensor:SizeMismatch', ...
                '3-D reconstruction size must match X_true.');
        end
        L_tensor = L_out;
    elseif ismatrix(L_out)
        if size(L_out, 1) ~= I * J || size(L_out, 2) ~= T
            error('compute_true_nre_tensor:SizeMismatch', ...
                'Matrix reconstruction must have size (I*J)-by-T.');
        end
        L_tensor = reshape(L_out, [I, J, T]);
    else
        error('compute_true_nre_tensor:InvalidReconstruction', ...
            'L_out must be a 3-D tensor or an (I*J)-by-T matrix.');
    end

    err_curve = NaN(1, T);
    for t = 1:T
        ref_frame = X_true(:, :, t);
        rec_frame = L_tensor(:, :, t);
        denom = norm(ref_frame(:));
        if denom <= eps
            err_curve(t) = norm(rec_frame(:) - ref_frame(:));
        else
            err_curve(t) = norm(rec_frame(:) - ref_frame(:)) / denom;
        end
    end
end
