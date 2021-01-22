% ---------------------------------------- %
% MECH-358 Homework Set 1 Matlab Code
% Written By: Thomas Bement
% Student Number: 24099822
% Date: 01/18/2021
% ---------------------------------------- %

% Clean up any mess
clc
clear all
close all
clearvars global

% QUESTION 1
% ---------------------------------------- %
fprintf('<strong>QUESTION 1:</strong>\n')

% Define random symetric matrix B
N = 10;
B = rand(N,N);
B = B'*B;
[V1, lambda1, V2, lambda2] = BigEig(B);

% Find largest Eigen value through built in function

%Print all results with labels
    %Iterative Results
fprintf('<strong>Iterative Method</strong>')
fprintf('\nLargest Eigen Value: %f', lambda1)
fprintf('\nAssociated Vector: [')
fprintf('%g ', V1);
fprintf(']\n');
    %Built in results
fprintf('\n\n<strong>Built In Method</strong>')
fprintf('\nLargest Eigen Value: %f', lambda2)
fprintf('\nAssociated Vector: [')
fprintf('%g ', V2);
fprintf(']\n');

% QUESTION 2 C
% ---------------------------------------- %
fprintf('\n<strong>QUESTION 2-C:</strong>\n')

% Define A as the Matrix Given in the Assignment
A = [2 1 0; 4 2 1; 0 2 4];
[P, U, L, N] = GaussElim(A);

% Test results by displaying P'*L*U = A
test = inv(P)*L*U;
fprintf('\nP''*L*U:\n|')
for r = 1:N
    for c = 1:N
        fprintf('%g ', test(r,c));
    end
    if r ~= N
        fprintf('|\n|')
    else
        fprintf('|\n')
    end
end
fprintf('\nA:\n|')
for r = 1:N
    for c = 1:N
        fprintf('%g ', A(r,c));
    end
    if r ~= N
        fprintf('|\n|')
    else
        fprintf('|\n')
    end
end
% Display all other relevant matricies
fprintf('\nL:\n|')
for r = 1:N
    for c = 1:N
        fprintf('%g ', L(r,c));
    end
    if r ~= N
        fprintf('|\n|')
    else
        fprintf('|\n')
    end
end
fprintf('\nP:\n|')
for r = 1:N
    for c = 1:N
        fprintf('%g ', P(r,c));
    end
    if r ~= N
        fprintf('|\n|')
    else
        fprintf('|\n')
    end
end
fprintf('\nU:\n|')
for r = 1:N
    for c = 1:N
        fprintf('%g ', U(r,c));
    end
    if r ~= N
        fprintf('|\n|')
    else
        fprintf('|\n')
    end
end

% QUESTION 2 E
% ---------------------------------------- %
fprintf('\n<strong>QUESTION 2-E:</strong>\n')
% Define b as the Vector Giving in the Assignment
b = [1;4;14];
[z, x] = SubSol(P, L, U, b, N);
% Print formatted results
fprintf('\nVector Z: [')
fprintf('%g ', z);
fprintf(']\n');
fprintf('\nVector X: [')
fprintf('%g ', x);
fprintf(']\n');