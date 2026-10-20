function s=innerloop_advance(s,dt)
% Sample-and-hold the most recent CONTROL feedback over intervening sensor ticks.
assert(~isempty(s.feedback),'landing:Order','Write initial feedback before advancing');
oldTime=s.timestamp;
if isfield(s.config,'plant_step_callback') && ~isempty(s.config.plant_step_callback)
    % Implement later: retain plant/controller state inside s, consume
    % s.feedback.valid, return native 12D states (ship attitude is communicated without delay).
    s=feval(s.config.plant_step_callback,s,dt);
    if isfield(s,'stopped') && s.stopped
        assert(s.timestamp>=oldTime-1e-8 && s.timestamp<=oldTime+dt+1e-8, ...
            'landing:Time','Terminal plant time is outside the requested interval');
    else
        assert(abs(s.timestamp-oldTime-dt)<1e-8,'landing:Time','Plant failed to advance by dt');
    end
elseif isfield(s.config,'allow_state_hold') && s.config.allow_state_hold
    % Explicit interface test only; neither integration nor a closed-loop plant.
    s.timestamp=oldTime+dt;
else
    error('landing:MissingDynamics', ...
        'No dynamics/controller supplied. Configure plant_step_callback; allow_state_hold is test-only.');
end
s=innerloop_update_gimbal(s);
s.phase='read';
end
