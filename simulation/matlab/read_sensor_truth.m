function result = read_sensor_truth()
s = evalin('base','landingState');
if isfield(s.config,'truth_callback')
    truth = feval(s.config.truth_callback, s);
else
    [eta,etaDot,~] = s.ship(s.timestamp);
    eta = eta(:); etaDot = etaDot(:);
    phi=eta(4); theta=eta(5); psi=eta(6);
    Rx=[1 0 0;0 cos(phi) -sin(phi);0 sin(phi) cos(phi)];
    Ry=[cos(theta) 0 sin(theta);0 1 0;-sin(theta) 0 cos(theta)];
    Rz=[cos(psi) -sin(psi) 0;sin(psi) cos(psi) 0;0 0 1];
    R=Rz*Ry*Rx;
    world=[R eta(1:3);0 0 0 1];
    % Point-mass surrogate with independently imposed aircraft rocking.
    amplitude=[.6;.6;.6]; period=[3;3.1;2.9];
    if isfield(s.config,'attitude_amplitude_rad'), amplitude=s.config.attitude_amplitude_rad(:); end
    if isfield(s.config,'attitude_period_s'), period=s.config.attitude_period_s(:); end
    angles=amplitude.*sin(2*pi*s.timestamp./period+[0;.4;.9]);
    rates=amplitude.*(2*pi./period).*cos(2*pi*s.timestamp./period+[0;.4;.9]);
    r=angles(1);p0=angles(2);y=angles(3);
    Rb=diag([1 -1 -1])*[cos(y) -sin(y) 0;sin(y) cos(y) 0;0 0 1]* ...
        [cos(p0) 0 sin(p0);0 1 0;-sin(p0) 0 cos(p0)]*[1 0 0;0 cos(r) -sin(r);0 sin(r) cos(r)];
    Ebody=[1 0 -sin(p0);0 cos(r) sin(r)*cos(p0);0 -sin(r) cos(r)*cos(p0)];
    wb=Ebody*rates;
    neutral=eye(4); if isfield(s.config,'T_body_camera'), neutral=s.config.T_body_camera; end
    s.gimbal_state=gimbal_step(s.gimbal_state,s.timestamp,Rb,R,neutral,s.config.gimbal);
    s.gimbal_measurement=gimbal_sample(s.gimbal_state,neutral,s.config.gimbal);
    mount=s.gimbal_measurement.T_body_camera;
    point=zeros(3,1); if isfield(s.config,'gear_body'), point=s.config.gear_body(:); end
    reference=zeros(3,1);
    if isfield(s.config,'position_reference') && strcmp(s.config.position_reference,'gear'), reference=point; end
    cameraLever=mount(1:3,4)-reference;
    pc=s.position+Rb*cameraLever; vc=s.velocity+Rb*cross(wb,cameraLever);
    p=R'*(pc-eta(1:3));
    % Euler rates -> body angular velocity, for relative-frame derivative.
    E=[1 0 -sin(theta);0 cos(phi) sin(phi)*cos(theta);0 -sin(phi) cos(phi)*cos(theta)];
    omega=E*etaDot(4:6);
    v=R'*(vc-etaDot(1:3))-cross(omega,p);
    gear=R'*(s.position+Rb*(point-reference)-eta(1:3));
    vgear=R'*(s.velocity+Rb*cross(wb,point-reference)-etaDot(1:3))-cross(omega,gear);
    truth=struct('timestamp',s.timestamp,'T_world_deck',world, ...
        'T_deck_camera',[R'*Rb*mount(1:3,1:3) p;0 0 0 1], ...
        'velocity_deck',v','T_body_camera',mount,'gear_position_deck',gear','gear_velocity_deck',vgear');
    assignin('base','landingState',s);
end
% Record a read of this exact state snapshot for the interface-only backend.
if isfield(s.config,'truth_callback') && strcmp(s.config.truth_callback,'innerloop_truth')
    s.observed_timestamp=s.timestamp;
    assignin('base','landingState',s);
end
result=jsonencode(truth);
end
