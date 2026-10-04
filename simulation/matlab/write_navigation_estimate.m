function result = write_navigation_estimate(estimateJSON)
s=evalin('base','landingState');
e=jsondecode(estimateJSON);
assert(abs(e.timestamp-s.timestamp)<1e-8,'landing:Timestamp','Feedback time mismatch');
assert(all(size(e.T_deck_camera)==[4 4]),'landing:Shape','Expected 4x4 pose');
assert(all(isfinite(e.T_deck_camera(:))),'landing:Finite','Invalid estimate');
s.estimate=e;
if isfield(s.config,'feedback_callback'), s=feval(s.config.feedback_callback,s,e); end
assignin('base','landingState',s);
result=jsonencode(struct('accepted',true));
end
