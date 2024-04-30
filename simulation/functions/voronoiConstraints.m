function [C_blk,d_cat] = voronoiConstraints(p,r,epsilon)
    
    N = size(p,2);

    % Calculate Voronoi constraints
    D = delaunay(p(1,:)',p(2,:)');
    D = [[D(:,1),D(:,2)];[D(:,1),D(:,3)];[D(:,2),D(:,3)]];
    D = unique(D,'rows');
    C = cell(N,1);
    d = cell(N,1);
    for h = 1:size(D,1)
        i = D(h,1);
        j = D(h,2);
    
        cij = (p(:,j) - p(:,i))/norm((p(:,j) - p(:,i)),2);
        dij = 1/2*cij'*(p(:,j) + p(:,i));
    
        C{i}(end+1,:) = cij;
        d{i}(end+1,:) = dij - (r + epsilon/2);
        C{j}(end+1,:) = -cij;
        d{j}(end+1,:) = -dij - (r + epsilon/2);
    
    end
    
    C_blk = [];
    d_cat = [];
    
    for i = 1:N
        
        C_blk = blkdiag(C_blk,C{i});
        d_cat = [d_cat;d{i}];
    
    end

end
