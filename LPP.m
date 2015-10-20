function [ W ] = LPP( Data, n, k)
%LPP Find the locality preserving projection of the data 'Data', with k nearest neighbours and n dimensions.
    % In the eigenanalysis part, each column is a data entry
    % but the KNearestNeighbours command takes rows as data entries
    X = Data';
    
    % 1. Calculate S, the connectivity matrix
    S = KNearestNeighbours(Data, k);

    % 2. Find the matrix U, that whitens XDX'
    D = diag(sum(S,1));
    Ds = sqrt(D);
    
    L = eig(Ds * (X' * X) * Ds);
    Lp = diag(1 ./ sqrt(L));

    U = Lp * Ds * X';
    
    % 3. Project X with U
    Xp = U * X;
    
    % 4. Eigenanalysis
    [Vq, Vl] = eig(Xp * (D - S) * Xp');
    
    % 5. Find best eigenvectors, perform dimensionality reduction if desired
    [~, eig_indices] = sort(diag(Vl), 'ascend');
    top_indices = eig_indices(1:min(size(eig_indices,1), n));
    
    Q = Vq(:,top_indices);
    W = U' * Q;
end
