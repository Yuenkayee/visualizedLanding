function [heli,ship]=landing_live_read(s)
% Runtime ports are available after initialize and while paused.
if s.stopped
    result=s.live.SimulationOutput;
    heli=last_sample(result.yout.getElement(s.native_indices(1)).Values,s.timestamp);
    ship=last_sample(result.yout.getElement(s.native_indices(2)).Values,s.timestamp);
else
    heli=read_output(s.live_model,'LandingStateTap1');
    ship=read_output(s.live_model,'LandingStateTap2');
end
assert(numel(heli)==12 && numel(ship)==12 && all(isfinite([heli;ship])), ...
    'landing:State','Native heli/ship outputs must each contain 12 finite doubles');
[heli,ship]=landing_normalize_native(heli,ship,s.config);
end
function v=last_sample(ts,time)
assert(abs(double(ts.Time(end))-time)<1e-8,'landing:Stop','Terminal state/time mismatch');
v=double(reshape(ts.getdatasamples(numel(ts.Time)),[],1));
end
function v=read_output(model,name)
runtime=get_param([model '/' char(name)],'RuntimeObject');
assert(~isempty(runtime),'landing:Runtime','State signal must remain available at runtime');
v=double(runtime.OutputPort(1).Data); v=v(:);
end
