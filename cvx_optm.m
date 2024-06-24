% Generate the matrix
m = 256;  % =2^8
N = 2048; % =2^11
P = 10;   % P stands for power
H = sqrt(P) * 1/sqrt(m)*randn(m, N);

% Generate the column vector of channels, each entry is exponential distribution with mean 5
Ka = 70; 
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
z = randn(m, 1);
% z = zeros(m,1); 

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
while objval > 0.0001 || isnan(objval)
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
        minimize( 1000*Ka/norm(y,2)*norm(y-H*P*x, 2) + u'*P*ones(Ka,1)) 
    cvx_end

    objval = cvx_optval;
    
    probs = P*ones(Ka,1); 
    [probs, idx] = sort(probs, "ascend"); 

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


% Display the indices
fprintf(" /// /// /// /// /// /// ///")
fprintf(" /// /// /// /// /// /// ///")
fprintf(" /// /// /// /// /// /// ///")

fprintf('These are the estimated active messages')
disp( find(u'==0) );
fprintf('These are the true active messages     ')
disp(chosenNums);

disp(length(setdiff(find(u'==0), chosenNums))); 