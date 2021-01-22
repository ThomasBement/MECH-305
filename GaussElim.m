% ---------------------------------------- %
% MECH-358 Homework Set 1 Matlab Code
% Written By: Thomas Bement
% Student Number: 24099822
% Date: 01/18/2021
% ---------------------------------------- %
function [P, U, L, N] = GaussElim(A)
    sz = size(A);
    if sz(1) ~= sz(2)
        fprintf('Matrix Input is Not Square\nQuitting...\n\n')
        return
    else
        N = sz(1);
    end    
    % Initialize variables
    U = A;
    L = eye(N);
    P = eye(N);
    % Perform Algorithm for Gaussian Elimination
    for k = 1:(N - 1)
        % find i ≥ k to maximize |U(i, k)|;
        maximum = norm(U(k,k));  % Initial condition
        imax = k;
        for i = k:(N)
            if (norm(U(i,k)) > maximum)
                maximum = norm(U(i,k));
                imax = i;
            end
        end
        temp = U(k, k:N); 
        U(k, k:N) = U(imax, k:N);
        U(imax, k:N) = temp;
        temp = L(k, 1:(k - 1));
        L(k, 1:(k - 1)) = L(imax, 1:(k - 1));
        L(imax, 1:(k - 1)) = temp;
        temp = P(k, :);
        P(k, :) = P(imax, :);
        P(imax, :) = temp;
        for j = (k + 1):N
            L(j, k) = U(j, k)/U(k, k);
            U(j, k:N) = U(j, k:N) - (L(j, k)*U(k, k:N));
        end
    end
end