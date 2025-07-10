function [indx,omega] = find_loc_vec2(x_grid,xi)

%Find indx s.t. x_grid(indx)<=xi<x_grid(indx+1)
%for indx=1,..,N-1
% omega is the weight on x_grid(indx)

n_x = size(x_grid,1);

% For each 'xi', get the position of the 'x' element bounding it on the left [p x 1]
indx = discretize(xi,x_grid);
indx(xi<x_grid(1))=1;        % Deal with xi<x_grid(1)
indx(xi>x_grid(end))=n_x-1;  % Deal with xi>x_grid(end)

%Weight on x_grid(indx)
omega = (x_grid(indx+1)-xi)./(x_grid(indx+1)-x_grid(indx));
omega = max(min(omega,1),0);

end %end function "find_loc_vec2"