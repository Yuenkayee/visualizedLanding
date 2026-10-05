function s=innerloop_advance(s,dt)
assert(strcmp(s.phase,'feedback'),'landing:Order','Write feedback before advancing');
oldTime=s.timestamp;
if isfield(s.config,'plant_step_callback') && ~isempty(s.config.plant_step_callback)
    % Implement later: retain plant/controller state inside s, consume
    % s.feedback.valid, return native 12D states (ship attitude is communicated without delay).
    s=feval(s.config.plant_step_callback,s,dt);
    assert(abs(s.timestamp-oldTime-dt)<1e-8,'landing:Time','Plant failed to advance by dt');
elseif isfield(s.config,'allow_state_hold') && s.config.allow_state_hold
    % Explicit interface test only; neither integration nor a closed-loop plant.
    s.timestamp=oldTime+dt;
else
    error('landing:MissingDynamics', ...
        'No dynamics/controller supplied. Configure plant_step_callback; allow_state_hold is test-only.');
end
s.phase='read';
end
