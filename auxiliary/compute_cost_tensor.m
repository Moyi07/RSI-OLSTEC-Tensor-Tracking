function f = compute_cost_tensor(X, P, PA, tensor_dims)
%COMPUTE_COST_TENSOR Half squared Frobenius error on selected entries.
    n1 = tensor_dims(1);
    n2 = tensor_dims(2);
    n3 = tensor_dims(3);

    Diff = P.*X - PA;
    Diff_flat = reshape(Diff, n1*n2, n3);

    f = .5*norm(Diff_flat , 'fro')^2;
end
