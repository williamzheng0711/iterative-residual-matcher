% Generate the matrix
m = 256;  % =2^8
N = 8192; % =2^13
P = 10;   % P stands for power
H = sqrt(P) * 1/sqrt(m)*randn(m, N);

% Generate the column vector of channels, each entry is exponential distribution with mean 5
Ka = 100; 
x_init = exprnd(15, [Ka, 1]);
x_init = sort(x_init, "descend");  % better channels get decoded first, in general

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

RowCollects = []; 
cdwdToChannel = []; 

u = zeros(N,1);
u(1:N-Ka,1) = 1;

objval = 100; 

j = 0;
while objval > 0.0001
    if j > 50
        break;
    end
    
    fprintf('Now j is: %d.\n', j);

    cvx_precision low
    cvx_begin
        variable P(N, Ka)
    
        P >= 0;
        ones(1,N)*P == ones(1,Ka); 
        P*ones(Ka,1) <= ones(N,1);
    
        % Objective function
        minimize(100*norm(y-H*P*x, 2) + u'*P*ones(Ka,1)) 
    cvx_end

    objval = cvx_optval;
    
    probs = P*ones(Ka,1); 
    [probs, idx] = sort(probs, "ascend"); 
    u = zeros(N,1);
    u(idx(1:N-Ka), 1) = 1;
    j = j + 1;
end


% Display the indices
fprintf('These are the estimated active messages')
disp( find(u'==0) );
fprintf('These are the true active messages     ')
disp(chosenNums);

disp(length(setdiff(RowCollects, chosenNums))); 