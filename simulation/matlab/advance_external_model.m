function result = advance_external_model(dt)
validateattributes(dt,{'double'},{'scalar','finite','positive'});
s=evalin('base','landingState');
if isfield(s.config,'advance_callback')
    s=feval(s.config.advance_callback,s,dt);
else
    a=-s.velocity;
    if ~isempty(s.estimate) && s.estimate.healthy
        target=.8; kp=.25; kd=1; limit=2;
        if isfield(s.config,'target_height'), target=s.config.target_height; end
        if isfield(s.config,'kp'), kp=s.config.kp; end
        if isfield(s.config,'kd'), kd=s.config.kd; end
        if isfield(s.config,'max_acceleration'), limit=s.config.max_acceleration; end
        a=kp*([0;0;target]-s.estimate.T_deck_camera(1:3,4))-kd*s.estimate.velocity(:);
        a=a*min(1,limit/max(norm(a),1e-9));
        truth=jsondecode(read_sensor_truth()); a=truth.T_world_deck(1:3,1:3)*a;
    end
    s.position=s.position+s.velocity*dt+.5*a*dt^2;
    s.velocity=s.velocity+a*dt;
    s.timestamp=s.timestamp+dt;
end
assignin('base','landingState',s);
result=jsonencode(struct('timestamp',s.timestamp));
end
