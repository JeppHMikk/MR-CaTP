function alpha = arrprob(p1,p2,alpha,d50)

    d = norm(p1-p2,2);
    alpha = 1 - 1/(1 + exp(-alpha*(d - d50)));
    
end