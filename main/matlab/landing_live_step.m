function s=landing_live_step(s,dt)
assert(strcmp(s.phase,'feedback'),'landing:Order','Read -> feedback -> advance required');
assert(abs(s.feedback.timestamp-s.timestamp)<1e-8,'landing:Stale','Stale feedback');
h=s.config.plant_step; n=round(dt/h);
assert(n>=1 && abs(n*h-dt)<1e-8,'landing:Time','dt must be a multiple of plant_step');
fields=fieldnames(s.config.inputs);
for k=1:numel(fields)
    value=s.feedback.(s.config.inputs.(fields{k}));
    % set_param updates the initialized execution immediately, including t=0.
    % Simulation.setBlockParameter before the first step is only a run setting.
    set_param([s.live_model '/' fields{k}],'Value',mat2str(double(value),17));
end
start=s.timestamp;
for k=1:n
    target=start+k*h;
    step(s.live,PauseTime=target);
    assert(abs(s.live.Time-target)<1e-8,'landing:Time','Plant stopped or missed requested time');
    [s.helicopter_state,s.ship_state]=landing_live_read(s);
    s.timestamp=s.live.Time;
    s=innerloop_update_gimbal(s); % sample actual aircraft attitude at inner-loop rate
end
end
