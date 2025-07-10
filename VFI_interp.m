function [V,Policy]=VFI_interp(n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, vfoptions)

a_grid = gather(a_grid);
z_grid = gather(z_grid);
pi_z   = gather(pi_z);


%% Create return function matrix
aprime_gridvals = a_grid;
a_gridvals      = shiftdim(a_grid,-1);
z_gridvals      = shiftdim(z_grid,-2);
ReturnMatrix = Aiyagari1994_ReturnFn_cpu(aprime_gridvals,a_gridvals,z_gridvals,Params.alpha,Params.delta,Params.mu,Params.r);

N_a=prod(n_a);
N_z=prod(n_z);

pi_z_transpose = pi_z';

VKronold = zeros(N_a,N_z);
Policy = zeros(N_a,N_z);

Tolerance = vfoptions.tolerance;
maxiter = vfoptions.maxiter;
Howards2 = vfoptions.Howards2;
n_fine = vfoptions.n_fine;
do_interp= vfoptions.do_interp;
DiscountFactorParamsVec = Params.(DiscountFactorParamNames{1});

tempcounter=1;
currdist=Inf;
while currdist>Tolerance && tempcounter<=maxiter

    % Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV = VKronold*pi_z_transpose; % (a',z)

    VKron = zeros(N_a,N_z);

    for z_c=1:N_z
        EV_z = EV(:,z_c);
        z_vals = z_grid(z_c);

        entireRHS=ReturnMatrix(:,:,z_c)+DiscountFactorParamsVec*EV_z; %aprime by a by z
        [max_val,max_ind]=max(entireRHS,[],1); % (1,N_a)
        VKron(:,z_c) = max_val;
        Policy(:,z_c) = a_grid(max_ind);

        lb = a_grid(max(max_ind-1,1));
        ub = a_grid(min(max_ind+1,N_a));
        aprime_fine = nonlinspace(lb',ub',n_fine);
        ReturnMatrix_fine = Aiyagari1994_ReturnFn_cpu(aprime_fine,a_gridvals,z_vals,Params.alpha,Params.delta,Params.mu,Params.r);
        EV_z_interp = interp1(a_grid,EV_z,aprime_fine,'linear','extrap');
        entireRHS_fine = ReturnMatrix_fine+DiscountFactorParamsVec*EV_z_interp;
        [max_val,max_ind]=max(entireRHS_fine,[],1); % (1,N_a)
        VKron(:,z_c) = max_val;
        tempindex = sub2ind([n_fine,N_a],max_ind',(1:N_a)');
        Policy(:,z_c) = aprime_fine(tempindex);
  
    end % end z

    %---------------------------------------------------------------------%
    % Howard update
    Ftemp = Aiyagari1994_ReturnFn_cpu(Policy,a_grid,z_grid',Params.alpha,Params.delta,Params.mu,Params.r);
     % Find interp indexes and weights
    [aprime_opt,weight_opt] = find_loc_vec2(a_grid,Policy);
    V_howard = zeros(N_a,N_z);
    for Howards_counter=1:Howards2
        EV_howard = VKron*pi_z_transpose;
        for z_c=1:N_z
            EV_howard_z = EV_howard(:,z_c);
            %for a_c=1:N_a
                EVi = weight_opt(:,z_c).*EV_howard_z(aprime_opt(:,z_c))+...
                    (1-weight_opt(:,z_c)).*EV_howard_z(aprime_opt(:,z_c)+1);
                V_howard(:,z_c) = Ftemp(:,z_c)+DiscountFactorParamsVec*EVi;
            %end
        end
        VKron = V_howard;
    end
    %---------------------------------------------------------------------%
    
    VKrondist=VKron(:)-VKronold(:);
    currdist=max(abs(VKrondist))

    VKronold = VKron;
    tempcounter=tempcounter+1;
   
end %end while

Policy=reshape(Policy,[N_a,N_z]);
V = reshape(VKron,[N_a,N_z]);

end