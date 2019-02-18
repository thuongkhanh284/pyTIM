X = [[1,2];[2,3,];[1,4]];
X = double(X);
[~,N] = size(X);

% check linear independency
if rank(X)<N 
    error('TrainPGM: Invalid input.');
end

mu = mean(X,2);   
X = X - repmat(mu,1,N);
[U,S,V] = svd(X,'econ');
S(N,:) = [];
S(:,N) = [];
V(:,N) = [];
U(:,N) = [];
Q = S*V';

G = getPathWeightMatrix(N);
L = getLapacian(G);

[V0,~] = eig(L);
V0(:,1) = [];
W = (Q*Q')\(Q*V0)   % Q'Vk = yk,½â³öVk

m = zeros(N-1,1);
for j = 1 : N-1
    Q(:,1)'*W(:,j)
    sin(1/N*j*pi+pi*(N-j)/(2*N))
    m(j) = Q(:,1)'*W(:,j)/sin(1/N*j*pi+pi*(N-j)/(2*N));
end
model.W = W;
model.U = U;
model.mu = mu;
model.n = N;
model.m = m;
model
