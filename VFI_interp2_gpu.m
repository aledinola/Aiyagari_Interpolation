function [V,Policy]=VFI_interp2_gpu(n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, vfoptions)

a_grid = gpuArray(a_grid);
z_grid = gpuArray(z_grid);
pi_z   = gpuArray(pi_z);


%% Create return function matrix
aprime_gridvals = a_grid;
a_gridvals      = shiftdim(a_grid,-1);
z_gridvals      = shiftdim(z_grid,-2);
ReturnMatrix = arrayfun(@Aiyagari1994_ReturnFn,aprime_gridvals,a_gridvals,z_gridvals,Params.alpha,Params.delta,Params.mu,Params.r);

N_a=prod(n_a);
N_z=prod(n_z);
%NA = gpuArray.colon(1,N_a)';
%NAZ = gpuArray.colon(1,N_a*N_z)';

[a_ind,z_ind] = ndgrid((1:N_a)',(1:N_z)');
a_ind = a_ind(:);
z_ind = z_ind(:);

pi_z_transpose = pi_z';

VKronold = zeros(N_a,N_z,'gpuArray');
Policy = zeros(N_a,N_z,'gpuArray');

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

        if do_interp==1
            lb = a_grid(max(max_ind-1,1));
            ub = a_grid(min(max_ind+1,N_a));
            aprime_fine = nonlinspace(lb',ub',n_fine);
            ReturnMatrix_fine = arrayfun(@Aiyagari1994_ReturnFn,aprime_fine,a_gridvals,z_vals,Params.alpha,Params.delta,Params.mu,Params.r);
            EV_z_interp = interp1(a_grid,EV_z,aprime_fine,'linear','extrap');
            entireRHS_fine = ReturnMatrix_fine+DiscountFactorParamsVec*EV_z_interp;
            [max_val,max_ind]=max(entireRHS_fine,[],1); % (1,N_a)
            VKron(:,z_c) = max_val;
            tempindex = sub2ind([n_fine,N_a],max_ind',(1:N_a)');
            Policy(:,z_c) = aprime_fine(tempindex);
        end

    end % end z

    %---------------------------------------------------------------------%
    % Howard update
    Ftemp = Aiyagari1994_ReturnFn_cpu(Policy,a_grid,z_grid',Params.alpha,Params.delta,Params.mu,Params.r);
    Ftemp_vec = reshape(Ftemp,[N_a*N_z,1]);

    % Find interp indexes and weights
    [aprime_opt,weight_opt] = find_loc_vec2(a_grid,Policy);

    aprime_opt_vec = aprime_opt(:);
    weight_opt_vec = weight_opt(:);

    ind = a_ind+(z_ind-1)*N_a;
    indp = aprime_opt_vec+(z_ind-1)*N_a;
    indpp = aprime_opt_vec+1+(z_ind-1)*N_a;
    Qmat = sparse(ind,indp,weight_opt_vec,N_a*N_z,N_a*N_z)+...
         sparse(ind,indpp,1-weight_opt(:),N_a*N_z,N_a*N_z);

    for h_c=1:Howards2
        EV_howard = VKron*pi_z_transpose; % (a',z)
        EV_howard = reshape(EV_howard,[N_a*N_z,1]);
        VKron = Ftemp_vec+DiscountFactorParamsVec*Qmat*EV_howard;
        VKron = reshape(VKron,[N_a,N_z]);
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