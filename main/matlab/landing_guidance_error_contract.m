function form=landing_guidance_error_contract(cmd)
% Accept target-r or target+(-r); both implement p_target-p_helicopter.
plus=[cmd '/Plus']; input=[cmd '/cg_relative_estimate'];
p=get_param(plus,'PortHandles'); q=get_param(input,'PortHandles');
line=get_param(p.Inport(1),'Line');
assert(line~=-1,'landing:Feedback','Guidance position input is disconnected');
source=get_param(line,'SrcPortHandle'); signs=get_param(plus,'Inputs');
if strcmp(signs,'-+')
    assert(source==q.Outport(1),'landing:Feedback','Expected direct CG input for target minus CG');
    form='target - CG (direct -+)';
elseif strcmp(signs,'++')
    block=get_param(source,'Parent');
    assert(strcmp(get_param(block,'BlockType'),'Gain'),'landing:Feedback', ...
        'Guidance ++ requires a -1 gain on the CG input');
    gain=slResolve(get_param(block,'Gain'),block);
    assert(isscalar(gain) && gain==-1,'landing:Feedback','Guidance CG gain must be -1');
    gp=get_param(block,'PortHandles'); line=get_param(gp.Inport(1),'Line');
    assert(line~=-1 && get_param(line,'SrcPortHandle')==q.Outport(1), ...
        'landing:Feedback','Guidance -1 gain must use the navigation CG input');
    form='target + (-CG) (-1 gain and ++)';
else
    error('landing:Feedback','Expected guidance error target minus CG');
end
end
