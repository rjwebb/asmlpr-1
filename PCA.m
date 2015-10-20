function [ W, WD ] = PCA( Data, n )
%PCA Returns the principal component analysis transformation matrix
    % de-mean X
    Data_d = bsxfun(@minus, Data, mean(Data,1));
    X = Data_d';
    
    % eigenanalysis
    [W,WD] = eigs(X * X', n);
end

