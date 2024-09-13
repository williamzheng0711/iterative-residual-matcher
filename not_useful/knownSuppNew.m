% Generate the matrix
m = 2^8;  % =256
N = 10000; %
P = 100;   % P stands for power
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

while true
    guesses_old = guesses; 
    for j = 1:Ka
        f = @(v) norm(y_guess- x(j)* H(:,guesses(j)) + x(j)*v -y , 2);
        results = arrayfun(@(t) f(H(:, t)), 1:size(H, 2));
        if j > 1
            results(guesses(1:(j-1))) = Inf;
        end
        [~, idx] = min(results);
        guesses(j) = idx; 
    end 
    if prod(guesses_old == guesses) == 1
        break
    end
end


% Display the indices
fprintf('These are the estimated active messages')
disp(guesses);
disp(length(setdiff(guesses, chosenNums))); 