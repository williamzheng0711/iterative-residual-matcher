% Generate the matrix
m = 256;  % =2^8
N = 8192; % =2^10
% P = 10;   % P stands for power

H = randn(m, N);
% n = sqrt(sum(H.^2,1));
% H = bsxfun(@rdivide, H,n);

% Generate the column vector of channels, each entry is exponential distribution with mean 5
Ka = 5; 
x_init = exprnd(15, [Ka, 1]); 
x_init = sort(x_init, "descend");  % better channels get decoded first, in general

% Ranomly choose Ka codewords from N
% chosenNums = randperm(N, Ka);
chosenNums = 1:Ka; 
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
u(randperm(N, N-Ka),1) = 1;

objval = 100; 

j = 0;
flag = 1;
while objval > 0.0001 || isnan(objval)
    if j > 50
        break;
    end
    
    fprintf('Now (j,flag) is: (%d, %d)\n', j, flag);


    % cvx_precision low
    cvx_begin
        variable P(N, Ka)
    
        P >= 0;
        ones(1,N)*P == ones(1,Ka); 
        P*ones(Ka,1) <= ones(N,1);
        norm(y-H*P*x,2) + sum(sum(abs(P))) <= Ka; 

        % Objective function
        minimize( u'*P*ones(Ka,1) );  
    cvx_end

    objval = cvx_optval;
    
    [probs, idx] = sort(P*ones(Ka,1), "ascend"); 

    u_temp = u; 
    u = zeros(N,1);
    u(idx(1:N-Ka), 1) = 1; 
    disp(find(u_temp' ==0)); 
    disp(find(u' ==0)); 
    j = j + 1;
    
    if prod(find(u'==0)==find(u_temp'==0)) == 1
       break;
    end
   
end


disp(length(setdiff(find(u'==0), chosenNums))); 