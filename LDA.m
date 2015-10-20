function [ W ] = LDA( Data, Gnd, N )
%LDA Perform linear discriminant analysis on a dataset 'Data' (samples X features), given a list of class numbers 'Gnd' and number of dimensions to preserve N
    num_clusters = size(unique(Gnd),1);
    cluster_freqs = hist(Gnd, num_clusters);
    
    E_array = cell(num_clusters);
    for i = 1:num_clusters
        E_array{i} = ones(cluster_freqs(i)) * cluster_freqs(i);
    end
    M = blkdiag(E_array{:});
    
    X = Data';

    I = eye(size(M,1));

    % 1. Perform eigenanalysis of Xw
    Xw = X * (I - M);
    Ue = Xw' * Xw;
    [Vw, Lw] = eig(Ue);

    % Remove zero eigenvalues and corresponding vectors
    a = diag(Lw);
    inds = find(a);
    Vw_clean = Vw(inds,:);
    Lw_clean = Lw(inds,:);
    
    % 2. From the eigenvalues compute the transform to whiten Sw (ie, to make w' Sw w = 1).
    U = X * (I - M) * Vw_clean / Lw_clean;
    
    % 3. Project to a new space Xb = U' X M
    Xb = U' * X * M;
    
    % 4. Perform PCA on Wb to find Q
    Q = PCA(Xb, N);

    % 5. Compute final transform
    W = U * Q;
end

