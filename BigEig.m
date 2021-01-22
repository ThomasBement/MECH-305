% ---------------------------------------- %
% MECH-358 Homework Set 1 Matlab Code
% Written By: Thomas Bement
% Student Number: 24099822
% Date: 01/18/2021
% ---------------------------------------- %
function [V1, lambda1, V2, lambda2] = BigEig(A)
    sz = size(A);
    if sz(1) ~= sz(2)
        fprintf('Matrix Input is Not Square\nQuitting...\n\n')
        return
    else
        N = sz(1);
    end
    % Find largest Eigen value through itteration
    V10 = rand(N,1);
    threshold = 1E-6;
    rErr = threshold+1;
    while rErr > threshold
        V10 = V10/norm(V10);
        V1 = A*V10;
        V1 = V1/norm(V1);
        rErr = 1 - norm(dot(V1,V10)); % Dot product gives better indicator for relative error
        V10 = V1;
    end
    lambda1 = dot((A*V1),V1);
    lambda2 = eigs(A,1);
    V1 = V1;
    V2 = null(A-lambda2*eye(size(A,1)));
end