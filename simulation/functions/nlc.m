function [c,ceq] = nlc(v,n)

    v = reshape(v(1:2*n),2,n);

    c = [];

    for i = 1:n

        c = norm(v(:,i),2) - 1;

    end

    ceq = [];


end
