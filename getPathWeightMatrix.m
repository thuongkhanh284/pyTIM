function G = getPathWeightMatrix(N)
    G = zeros(N,N);
    for i= 1:N-1
        G(i,i+1) = 1;
        G(i+1,i) = 1;
    end
end