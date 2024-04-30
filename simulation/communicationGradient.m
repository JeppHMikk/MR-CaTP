function [DLdp,dldp] = communicationGradient(p,A,v2,K,alpha)

    N = size(p,3);
    
    dAdt2 = zeros(N,N,2,N);
    dDdt2 = zeros(N,N,2,N);
    dldp = [];
    for i = 1:N
        for j = 1:N
            if(i ~= j)
                dij = norm(p(:,i) - p(:,j),2);
                dAdt2(i,j,1,i) = -alpha*(1 - A(i,j))*A(i,j)*((p(1,i) - p(1,j))/dij);
                dAdt2(i,j,2,i) = -alpha*(1 - A(i,j))*A(i,j)*((p(2,i) - p(2,j))/dij);
                dAdt2(j,i,1,i) = dAdt2(i,j,1,i);
                dAdt2(j,i,2,i) = dAdt2(i,j,2,i);
            end
        end
        %dAdt2(:,:,1,i) = dAdt2(:,:,1,i);
        %dAdt2(:,:,2,i) = dAdt2(:,:,2,i);
        dDdt2(:,:,1,i) = diag(sum(dAdt2(:,:,1,i),2));
        dDdt2(:,:,2,i) = diag(sum(dAdt2(:,:,2,i),2));
        dldp(end+1) = v2'*(dDdt2(:,:,1,i) - dAdt2(:,:,1,i))*v2;
        dldp(end+1) = v2'*(dDdt2(:,:,2,i) - dAdt2(:,:,2,i))*v2;
    end

    DLdp = kron(tril(ones(K)),dldp);

end
