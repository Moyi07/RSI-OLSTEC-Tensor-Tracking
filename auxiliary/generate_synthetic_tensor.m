function [Tensor_Y_Noiseless, Tensor_Y_Noiseless_Normalized, Tensor_Y_Normalized, OmegaTensor, ...
    Matrix_Y_Noiseless, Matrix_Y_Noiseless_Normalized, Matrix_Y_Normalized, OmegaMatrix, ...
    rows, cols, total_slices, Normalize_Ratio] = generate_synthetic_tensor(tensor_dims, rank, fraction, inverse_snr, data_subtype)
% This file is part of OLSTEC package.
%
% Created by H.Kasai on June 07, 2017
 
    disp('# Generating synthetic dataset ....');

    rows            = tensor_dims(1);
    cols            = tensor_dims(2);
    total_slices    = tensor_dims(3);

    if strcmp(data_subtype, 'Static')

        disp('## Static dataset ....');    

        A=randn(rows, rank);
        B=randn(cols, rank);
        C=randn(total_slices, rank);

        % Create observed tensor that follows PARAFAC model
        Tensor_Y_Noiseless = zeros(rows,cols,total_slices);
        for k=1:total_slices
            Tensor_Y_Noiseless(:,:,k)=A*diag(C(k,:))*B.';
        end

    else

        disp('## Dynamic dataset ....');

        REPEAT_NUM = 4;
        Tensor_Y_Noiseless = zeros(rows, cols, total_slices);
        slice_edges = round(linspace(0, total_slices, REPEAT_NUM + 1));

        for i=1:REPEAT_NUM
            slice_start = slice_edges(i) + 1;
            slice_end = slice_edges(i+1);
            SUB_SLICE = slice_end - slice_start + 1;
            if SUB_SLICE <= 0
                continue;
            end

            A=randn(rows,rank);
            B=randn(cols,rank);
            C=randn(SUB_SLICE,rank);   

            % Create observed tensor that follows PARAFAC model
            sub_tensor = zeros(rows,cols,SUB_SLICE);
            for k=1:SUB_SLICE
                sub_tensor(:,:,k)=A*diag(C(k,:))*B.';
            end  

            Tensor_Y_Noiseless(:,:,slice_start:slice_end) = sub_tensor;
        end

    end

    max_abs_signal = max(abs(Tensor_Y_Noiseless(:)));
    if max_abs_signal > 0
        Normalize_Ratio = 1 / max_abs_signal;
    else
        Normalize_Ratio = 1;
    end

    %% Add scaled Gaussian noise
    Tensor_Noise = randn(size(Tensor_Y_Noiseless));
    Norm_Tensor_Y_Noiseless = norm(reshape(Tensor_Y_Noiseless, rows*cols, total_slices),'fro');
    Norm_Tensor_Noise = norm(reshape(Tensor_Noise, rows*cols, total_slices),'fro');

    Tensor_Y = Tensor_Y_Noiseless + (inverse_snr * Norm_Tensor_Y_Noiseless / Norm_Tensor_Noise) * Tensor_Noise; % entries added with noise

    Tensor_Y_Noiseless_Normalized = Tensor_Y_Noiseless * Normalize_Ratio; 
    Tensor_Y_Normalized = Tensor_Y * Normalize_Ratio;

    % Matrix 
    Matrix_Y_Noiseless = reshape(Tensor_Y_Noiseless,[rows*cols total_slices]);
    Matrix_Y = reshape(Tensor_Y,[rows*cols total_slices]);

    Matrix_Y_Noiseless_Normalized = Matrix_Y_Noiseless * Normalize_Ratio;
    Matrix_Y_Normalized = Matrix_Y * Normalize_Ratio;
    %% Generate observation masks
    OmegaTensor = zeros(rows,cols,total_slices);
    OmegaMatrix = zeros(rows*cols,total_slices);
    for t=1:total_slices
        % Sample observed entries for frame t.
        M = round(fraction * rows * cols);
        p = randperm(rows * cols);
        idx = p(1:M)';

        % Omega Matrix
        OmegaMatrix(:,t) = false;
        OmegaMatrix(idx,t) = true;
        OmegaTensor(:,:,t) = reshape(OmegaMatrix(:,t),[rows,cols]);    
    end


end
