function [ S ] = KNearestNeighbours( X, k )
%KNEARESTNEIGHBOURS An implementation of the k nearest neighbours clustering algorithm
    N = size(X, 1);
    S = zeros(N, N);
    
    % for every data point
    for i = 1:N
        % calculate all of the distances
        distances = zeros(N);
        for j = 1:N
            distances(j) = L2Norm(X(i,:), X(j,:));
        end
        
        % find the nearest neighbours
        [~, Nearest] = sort(distances);
        kNearestNeighbours = Nearest(2:k+1);
        
        % generate connectivity matrix
        for l = 1:k
            nbr = kNearestNeighbours(l);
            S(i,nbr) = 1;
            S(nbr,i) = 1;
        end
    end
end

