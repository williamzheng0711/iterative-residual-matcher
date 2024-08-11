% Generate the matrix
m = 2^8;  % =256
N = 1024; % =8192
P = 1;   % P stands for power
H = sqrt(P) * 1/sqrt(m)*randn(m, N);

% Generate the column vector of channels, each entry is exponential distribution with mean 5
Ka = 50; 
x_init = exprnd(15, [Ka, 1]);
x_init = sort(x_init, "descend");  % better channels get decoded first

% Ranomly choose Ka codewords from N
% chosenNums = randperm(N, Ka);
chosenNums = 1:Ka; 
% disp(chosenNums); 
disp(x_init); 

% Generate the true superpositioned signal
y_true = zeros(m,1);
for i = 1:Ka
    y_true = y_true + x_init(i)*H(:,chosenNums(i)); 
end

% Additive noise (variance is 1, normalized)
% z = randn(m, 1);
z = zeros(m,1); 

% Produce the "observable" y: 
y_observe = y_true + z; 
x = x_init; 
y = y_observe; 

cvx_begin
    variable P(N, Ka)

    P >= 0;
    P <= 1;
    sum(P,1) == ones(1, Ka); 
    % sum(P,2) <= ones(N,1); 

    % Objective function
    minimize(norm(y-H*P*x, 2) + sum(sum(abs(P))) );
cvx_end

row_sums = P*ones(Ka,1);
[~, sortedIndex] = sort(row_sums, "descend");
guesses = sortedIndex(1:Ka);

% Display the indices
disp(diag(P)); 
disp(length(setdiff(guesses, chosenNums))); 