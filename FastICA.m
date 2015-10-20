function [ W, S ] = FastICA( Data, C )
%FASTICA Performs independent component analysis using the FastICA algorithm.
    max_iter = 10;
    
    X_nowhiten = Data';
    
    Whit = PCAWhitening(X_nowhiten, size(Data,1));
    
    % X must be NxM
    X = X_nowhiten * Whit;
    
    % M = number of samples
    % C = number of desired components
    % N = number of dimensions
    
    % ICA
    N = size(X,1);
    M = size(X,2);
    W = zeros(C, N);
    
    for p = 1:C
        % initialise wp, the vector that will express the pth component
        w = rand(N,1);

        for i = 1:max_iter
            %iteratively update wp
            p1 = X * G(w' * X)';
            p2 = Gp(w' * X) * ones(M,1) * w;
            w = (p1 - p2) / M;
            
            err = 0;
            for j = 1:(p-1)
                wj = W(j,:)';
                d = w' * wj * wj;
                err = err + d;
            end
            w = w - err;
            
            w = w / norm(w);
        end
                
        W(p,:) = w;
    end
    
    % return the matrix S
    S = W * X;
end

% definitions of the first and second derivative of a nonquadratic nonlinearity function f
function [X] = G(w)
    X = tanh(w);
end

function [X] = Gp(w)
    X = 1 - pow2(tanh(w));
end
