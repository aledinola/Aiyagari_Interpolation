function x = nonlinspace(lo,hi,n)
    
    x      = NaN(n,length(lo));
    x(1,:) = lo;
    for i = 2:n
        x(i,:) = x(i-1,:) + (hi-x(i-1,:))/(n-i+1);
    end
    
end