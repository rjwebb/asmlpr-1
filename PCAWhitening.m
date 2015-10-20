function [ W ] = PCAWhitening( Data, n )
%PCAWHITENING Returns a matrix that applies PCA and whitening to the data
    % de-mean X
    Data_d = bsxfun(@minus, Data, mean(Data,1));
    X = Data_d';
    
    % eigenanalysis
    [V,L] = eigs(X * X', n);

    % Apply whitening    
    eV = 1 ./ sqrt(abs(L));
    eV(isinf(eV)) = 0;
    W = V * eV;

end

