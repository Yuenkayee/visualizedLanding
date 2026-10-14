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
        e=s.estimate; p=e.T_deck_camera(1:3,4); v=e.velocity(:);
        if isfield(e,'gimbal') && ~isempty(e.gimbal)
            m=e.gimbal; point=zeros(3,1); if isfield(s.config,'gear_body'), point=s.config.gear_body(:); end
            lever=m.T_body_camera(1:3,1:3)'*(point-m.T_body_camera(1:3,4));
            p=p+e.T_deck_camera(1:3,1:3)*lever;
            v=v+e.T_deck_camera(1:3,1:3)*cross(e.angular_velocity_camera(:)-m.angular_velocity_camera(:),lever);
        end
        a=kp*([0;0;target]-p)-kd*v;
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
