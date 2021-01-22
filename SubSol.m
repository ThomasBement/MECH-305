% ---------------------------------------- %
% MECH-358 Homework Set 1 Matlab Code
% Written By: Thomas Bement
% Student Number: 24099822
% Date: 01/18/2021
% ---------------------------------------- %
function [z, x] = SubSol(P, L, U, b, N)
    z = zeros(N,1);
    res = P*b;
    z(1) = res(1);
    for r = 2:N
        sum = 0;
        for c = 1:(r-1)
            sum = sum + L(r,c)*z(c);
        end
        z(r)=(res(r)-sum)/L(r,r);
    end
    x = zeros(N,1);
    x(N) = z(N)/U(N,N);
    for r = N:-1:2
        sum = 0;
        for c = (r-1):-1:1
            sum = sum + U(r,c)*x(c);
        end
        x(r)=(res(r)-sum)/U(r,r);
    end
end