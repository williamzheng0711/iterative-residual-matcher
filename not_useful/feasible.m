% Generate the matrix
m = 10;  % =256
N = 20; %
P = 1;   % P stands for power
H = sqrt(P) * 1/sqrt(m)*randn(m, N);

% Generate the column vector of channels, each entry is exponential distribution with mean 5
Ka = 1; 
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

guesses = randperm(N, Ka);
y_guess = H(:,guesses)*x; 

U = zeros(N,Ka);
for i = 1:Ka
    U(guesses(i),i) = 1;
end

% disp(norm(y - H*U*x,2)^2+ ones(1,N)*abs(U)*ones(Ka,1));
U_t = U; 


X0 = U_t;
opts.record = 0;
opts.mxitr  = 20000000000; 
opts.xtol = 1e-20;
opts.gtol = 1e-20;
opts.ftol = 1e-20;

[X, out]= OptStiefelGBB(X0, @sparse_recovery, opts, N, Ka, y, H, x); 
disp(X);

    function [F, G] = sparse_recovery(U, N, Ka, y, H, x)
        F = norm(y - H*U*x,2)^2 + 100000* ones(1,N)*abs(U)*ones(Ka,1);
        G = 2*H'*(H* U *x-y)*x' + sign(U); 
    end
