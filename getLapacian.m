
function L = getLapacian(G)
    N = size(G,1);

    D = zeros(N,N);
    D((0:N-1)*N+(1:N)) = sum(G);
    L = D - G;
end