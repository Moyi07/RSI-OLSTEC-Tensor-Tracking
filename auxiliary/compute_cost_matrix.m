function f = compute_cost_matrix(X, P, A)
%COMPUTE_COST_MATRIX Half squared Frobenius error on selected entries.
    Diff = P.*X - P.*A;
    f = .5*norm(Diff , 'fro')^2;        
end
