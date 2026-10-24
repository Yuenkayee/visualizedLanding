function info=landing_prepare_control(model,c)
% On the execution copy, latch the host's just-computed CG signal. The
% converter still logs outputs, but cannot update its state again on the
% direct-feedthrough plant loop or introduce a stale sample at a paused tick.
plant=[model '/' c.autonomous_subsystem];
for k=1:2
    q=get_param(plant,'PortHandles'); line=get_param(q.Inport(k),'Line');
    src=get_param(line,'SrcPortHandle');
    source=[extractAfter(get_param(src,'Parent'),[model '/']) '/' num2str(get_param(src,'PortNumber'))];
    delete_line(model,source,[c.autonomous_subsystem '/' num2str(k)]);
    if k==1, name='LandingControlCG'; value='zeros(3,1)'; else, name='LandingControlValid'; value='0'; end
    add_block('simulink/Sources/Constant',[model '/' name],'Value',value,'SampleTime','0');
    add_line(model,[name '/1'],[c.autonomous_subsystem '/' num2str(k)]);
end
info=struct('position_constant','LandingControlCG','valid_constant','LandingControlValid', ...
    'policy','host writes current signal before each fixed-step advance');
end
