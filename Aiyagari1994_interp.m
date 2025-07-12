%% Aiyagari (1994) with interpolation, only VFI
clear,clc,close all
addpath(genpath('C:\Users\aledi\Documents\GitHub\VFIToolkit-matlab'))
% These codes set up and solve the Aiyagari (1994) model for a given
% parametrization. After solving the model they then show how some of the
% vfitoolkit commands to easily calculate things like the Gini coefficient
% for income, and how to plot the distribution of asset holdings.
%
% VFI Toolkit automatically detects hardware (GPU? Number of CPUs?) and
% sets defaults accordingly. It will run without a GPU, but slowly. It is
% indended for use with GPU.

%% Set some basic variables

% VFI Toolkit thinks of there as being:
% k: an endogenous state variable (assets)
% z: an exogenous state variable (exogenous labor supply)

% Size of the grids
n_k=1000;%2^9;
n_z=21;

% Parameters
Params.beta=0.96; %Model period is one-sixth of a year
Params.alpha=0.36;
Params.delta=0.08;
Params.mu=3;
Params.sigma=0.2;
Params.rho=0.6;

%% Set up the exogenous shock process
% Create markov process for the exogenous labour productivity, l.
Tauchen_q=3; % Footnote 33 of Aiyagari(1993WP, pg 25) implicitly says that he uses q=3
[z_grid,pi_z]=discretizeAR1_Tauchen(0,Params.rho,sqrt((1-Params.rho^2)*Params.sigma^2),n_z,Tauchen_q);
% Note: sigma is standard deviations of s, input needs to be standard deviation of the innovations
% Because s is AR(1), the variance of the innovations is (1-rho^2)*sigma^2

[z_mean,z_variance,z_corr,~]=MarkovChainMoments(z_grid,pi_z);
z_grid=exp(z_grid);
% Get some info on the markov process
[Expectation_l,~,~,~]=MarkovChainMoments(z_grid,pi_z); %Since l is exogenous, this will be it's eqm value 
% Note: Aiyagari (1994) actually then normalizes l by dividing it by Expectation_l (so that the resulting process has expectation equal to 1)
z_grid=z_grid./Expectation_l;
[Expectation_l,~,~,~]=MarkovChainMoments(z_grid,pi_z);
% If you look at Expectation_l you will see it is now equal to 1
Params.Expectation_l=Expectation_l;

%% Grids

% In the absence of idiosyncratic risk, the steady state equilibrium is given by
r_ss=1/Params.beta-1;
K_ss=((r_ss+Params.delta)/Params.alpha)^(1/(Params.alpha-1)); %The steady state capital in the absence of aggregate uncertainty.

% Set grid for asset holdings
k_grid=10*K_ss*(linspace(0,1,n_k).^3)'; % linspace ^3 puts more points near zero, where the curvature of value and policy functions is higher and where model spends more time

% Bring model into the notational conventions used by the toolkit
d_grid=0; %There is no d variable
a_grid=k_grid;
% z_grid
% pi_z;

n_d=0;
n_a=n_k;
% n_z


fprintf('Grid sizes are: %i points for assets, and %i points for exogenous shock \n', n_a,n_z)

%%
DiscountFactorParamNames={'beta'};

ReturnFn=@(aprime, a, z, alpha,delta,mu,r) Aiyagari1994_ReturnFn(aprime, a, z,alpha,delta,mu,r);
% The first inputs must be: next period endogenous state, endogenous state, exogenous state. Followed by any parameters

Params.r=0.038;
% Equilibrium wage
Params.w=(1-Params.alpha)*((Params.r+Params.delta)/Params.alpha)^(Params.alpha/(Params.alpha-1));



%%
% Solve for the stationary general equilbirium
vfoptions.verbose=0; % Use default options for solving the value function (and policy fn)
vfoptions.tolerance=1e-5;
vfoptions.maxiter=1000;
vfoptions.Howards2=40;

fprintf('Calculating various equilibrium objects \n')
tic
[V,Policy]=ValueFnIter_Case1(n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

vfoptions.do_interp=1;
vfoptions.n_fine = 30;
tic
[V_interp,Policy_interp]=VFI_interp2(n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames,vfoptions);
toc

pol_c = zeros(n_a,n_z);
z_grid = gather(z_grid);
for z_c=1:n_z
    for a_c=1:n_a
        pol_c(a_c,z_c)=Params.w*z_grid(z_c)+(1+Params.r)*a_grid(a_c)-Policy_interp(a_c,z_c); % Budget Constraint
    end
end

figure
plot(a_grid,a_grid,'--')
hold on
plot(a_grid,Policy_interp(:,1))
hold on
plot(a_grid,Policy_interp(:,end))

figure
plot(a_grid,pol_c(:,1))
hold on
plot(a_grid,pol_c(:,end))

% V is value function
% Policy is policy function (but as an index of k_grid, not the actual values)


