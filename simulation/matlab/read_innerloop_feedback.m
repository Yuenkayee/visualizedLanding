function result=read_innerloop_feedback()
s=evalin('base','landingState');
assert(isfield(s,'feedback') && ~isempty(s.feedback),'landing:Feedback','No feedback at current step');
assert(strcmp(s.phase,'feedback') && abs(s.feedback.timestamp-s.timestamp)<1e-8, ...
    'landing:Feedback','Feedback is not from the current completed transaction');
result=jsonencode(s.feedback);
end
