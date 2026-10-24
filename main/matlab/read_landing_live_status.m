function result=read_landing_live_status()
s=evalin('base','landingState');
value=struct('stopped',s.stopped,'timestamp',s.timestamp, ...
    'stop_reason',s.stop_reason,'helicopter_state',s.helicopter_state, ...
    'ship_state',s.ship_state);
if isfield(s,'control_used') && ~s.stopped, value.control_used=s.control_used; end
result=jsonencode(value);
end
