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
    p=R'*(s.position-eta(1:3));
    % Euler rates -> body angular velocity, for relative-frame derivative.
    E=[1 0 -sin(theta);0 cos(phi) sin(phi)*cos(theta);0 -sin(phi) cos(phi)*cos(theta)];
    omega=E*etaDot(4:6);
    v=R'*(s.velocity-etaDot(1:3))-cross(omega,p);
    truth=struct('timestamp',s.timestamp,'T_world_deck',world, ...
        'T_deck_camera',[R'*diag([1 -1 -1]) p;0 0 0 1], ...
        'velocity_deck',v');
end
% Record a read of this exact state snapshot for the interface-only backend.
if isfield(s.config,'truth_callback') && strcmp(s.config.truth_callback,'innerloop_truth')
    s.observed_timestamp=s.timestamp;
    assignin('base','landingState',s);
end
result=jsonencode(truth);
end
