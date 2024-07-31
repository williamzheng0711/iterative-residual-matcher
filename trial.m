% Generate the matrix
m = 2^8;  % =256
N = 1000; %
P = 1;   % P stands for power
H = sqrt(P) * 1/sqrt(m)*randn(m, N);

% Generate the column vector of channels, each entry is exponential distribution with mean 5
Ka = 20; 
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

disp(norm(y - H*U*x,2)^2+ ones(1,N)*abs(U)*ones(Ka,1));

t = 0; 
U_t = U; 

while true
    lr = 1e-4 * 1/sqrt(t+1); 
    Grad_t = 2*H'*(H* U_t *x-y)*x' + sign(U_t);
    Xi_t = -lr * (Grad_t - 1/2* U_t *(Grad_t'*U_t + U_t'*Grad_t));
    temp_1 = inv(eye(Ka)+Xi_t'*Xi_t);
    temp_2 = sqrtm(temp_1); 
    U_t = (U_t + Xi_t) * temp_2; 
    if mod(t,3000) == 0
        disp(norm(y - H*U_t*x,2)^2 + ones(1,N)*abs(U_t)*ones(Ka,1));
    end
    t = t+ 1;
end

% Display the indices
fprintf('These are the estimated active messages')
disp(guesses);
disp(length(setdiff(guesses, chosenNums))); 