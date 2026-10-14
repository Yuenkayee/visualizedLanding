function result=read_gimbal_telemetry()
s=evalin('base','landingState');
assert(isfield(s,'gimbal_measurement'),'landing:Gimbal','Backend has no encoder telemetry');
result=jsonencode(s.gimbal_measurement);
end
