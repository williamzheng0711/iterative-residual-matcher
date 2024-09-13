% Generate the matrix
m = 2^8;  % =256
N = 2^10; % =8192
P = 10;   % P stands for power
H = sqrt(P) * 1/sqrt(m)*randn(m, N);

% Generate the column vector of channels, each entry is exponential distribution with mean 5
Ka = 100; 
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

RowCollects = []; 
cdwdToChannel = []; 
for j = 1:Ka
    if j < Ka
        
        numCandidates = 5*(Ka-(j-1)); 
        candidates = []; 
        dp = @(v) v'*y;
        dp_results = arrayfun(@(t) dp(H(:,t)), 1:size(H, 2));
        if j > 1
            dp_results(RowCollects) = -Inf;
        end
%         sos = x'*x; 
%         threshold = min( arrayfun(@(t) norminv(0.5, x(t), sqrt( 1+1/m*sos + 1/m*x(t)^2 )), 1:size(x)) );
%         numCandidates = length(dp_results(dp_results>= threshold) ); 
%         disp(numCandidates); 
        [~, guessActive] = maxk(dp_results, numCandidates);

        cvx_begin quiet
            variable P(numCandidates, Ka-(j-1))
            fprintf('Now j is: %d.\n', j);
    
            P >= 0;
            ones(1,numCandidates)*P == ones(1, Ka-(j-1)); 
            P*ones(Ka-(j-1),1) <= ones(numCandidates,1);
            
%             if nnz(RowCollects)>0
%                 P(RowCollects,:) == 0;
%             end
        
            % Objective function
            minimize(norm(y-H(:,guessActive)*P*x, 2))
        cvx_end
        
        [value, index] = max(P(:));
        [row, col] = ind2sub(size(P), index);
        RowCollects(j) = guessActive(row); 
        y = y - H(:,RowCollects(j))*x(col);
        cdwdToChannel(j) = find(x_init == x(col)); 
        x(col) = [];
    end 

    if j == Ka % If 
        f = @(v) norm(y-x(1)*v,2);
        results = arrayfun(@(t) f(H(:, t)), 1:size(H, 2));
        results(RowCollects) = Inf;
        [~, idx] = min(results);
        RowCollects(j) = idx; 
        cdwdToChannel(j) = find(x_init == x(1)); 
    end
end 

% Display the indices
fprintf('These are the estimated active messages')
[~, sortedIndex] = sort(cdwdToChannel);
sorted_RowCollects = RowCollects(sortedIndex);
disp(sorted_RowCollects);
fprintf('These are the true active messages     ')
disp(chosenNums);

disp(length(setdiff(RowCollects, chosenNums))); 