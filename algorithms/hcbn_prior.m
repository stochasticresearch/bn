function [p_prior] = hcbn_prior(model, rho_s)

if(strcmpi(model,'Gaussian'))
    param = copulaparam('Gaussian',rho_s,'type','Spearman');
    % generate a truncated Normal distribution
    % FYI - SMS paper, they use a Laplacian distribution ... 
    pd = makedist('Normal');
    pd = truncate(pd,-1,1);
elseif(strcmpi(model,'Gumbel'))
    param = copulaparam('Gumbel',rho_s,'type','Spearman');
    % Generate shifted by Unity Exponential
    pd = makedist('Exponential','mu',4);
    % approximate the shift by truncating from 1 - 1000
    pd = truncate(pd,1,1000);
elseif(strcmpi(model,'Clayton'))
    param = copulaparam('Clayton',rho_s,'type','Spearman');
    % Generate Exponential w/ lambda=4
    pd = makedist('Exponential','mu',4);
else
    error('Unrecognized Model!')
end

% compute the prior probability from the probability object
p_prior = pd.pdf(param);

end