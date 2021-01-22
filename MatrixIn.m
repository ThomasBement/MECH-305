% ---------------------------------------- %
% MECH-358 Homework Set 1 Matlab Code
% Written By: Thomas Bement
% Student Number: 24099822
% Date: 01/18/2021
% ---------------------------------------- %
function [A] = MatrixIn(N)
    % Get user defined square matrix of size NxN
    A = zeros(N);
    for r = 1:N
        for c = 1:N
            fprintf('Entre Value for Entry: %i, %i   ', r, c)
            A(r,c) = input('');
        end
    end
end