function report=landing_check_timing(model,step)
% Validate the plant clock and discrete controller rates before simulation.
assert(isscalar(step) && isfinite(step) && abs(step-.02)<1e-12, ...
    'landing:Timing','The supplied inner loop requires a 0.02 s fixed step');
assert(strcmp(get_param(model,'SolverType'),'Fixed-step'), ...
    'landing:Timing','Model %s must use a fixed-step solver',model);
actual=slResolve(get_param(model,'FixedStep'),model);
assert(isnumeric(actual) && isscalar(actual) && abs(actual-step)<1e-12, ...
    'landing:Timing','Model %s fixed step must equal %.17g s',model,step);
blocks=find_system(model,'LookUnderMasks','all','FollowLinks','on','BlockType','MATLABSystem');
rates={};
for k=1:numel(blocks)
    cls=get_param(blocks{k},'System');
    if strcmp(cls,'CETI')
        parameter='Ts';
    elseif ismember(cls,{'H_inf','LPV_H_inf','NDO','LPV_NDO'})
        parameter='SampleTime';
    else
        continue;
    end
    value=slResolve(get_param(blocks{k},parameter),blocks{k});
    assert(isnumeric(value) && isscalar(value) && isfinite(value) && abs(value-step)<1e-12, ...
        'landing:Timing','%s.%s must equal %.17g s; align its discretization before running', ...
        blocks{k},parameter,step);
    rates{end+1}=struct('block',blocks{k},'parameter',parameter,'seconds',value); %#ok<AGROW>
end
report=struct('plant_step',step,'controllers',{rates});
end
