function result=read_innerloop_feedback()
s=evalin('base','landingState');
assert(isfield(s,'feedback') && ~isempty(s.feedback),'landing:Feedback','No feedback at current step');
result=jsonencode(s.feedback);
end
