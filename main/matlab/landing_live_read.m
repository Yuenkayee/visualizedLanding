function [heli,ship]=landing_live_read(s)
% Runtime ports are available after initialize and while paused.
heli=read_output(s.live_model,'LandingStateTap1');
ship=read_output(s.live_model,'LandingStateTap2');
assert(numel(heli)==12 && numel(ship)==12 && all(isfinite([heli;ship])), ...
    'landing:State','Native heli/ship outputs must each contain 12 finite doubles');
end
function v=read_output(model,name)
runtime=get_param([model '/' char(name)],'RuntimeObject');
assert(~isempty(runtime),'landing:Runtime','State signal must remain available at runtime');
v=double(runtime.OutputPort(1).Data); v=v(:);
end
