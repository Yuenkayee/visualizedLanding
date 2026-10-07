function result=read_navigation_context()
% Stipulated zero-delay ship attitude communication, no aircraft truth output.
s=evalin('base','landingState');
if isfield(s,'ship_state')
    e=s.ship_state(7:9);
    phi=e(1); theta=e(2); psi=e(3);
    Rx=[1 0 0;0 cos(phi) -sin(phi);0 sin(phi) cos(phi)];
    Ry=[cos(theta) 0 sin(theta);0 1 0;-sin(theta) 0 cos(theta)];
    Rz=[cos(psi) -sin(psi) 0;sin(psi) cos(psi) 0;0 0 1];
    R=Rz*Ry*Rx*diag([1 -1 -1]);
elseif isfield(s.config,'navigation_context_callback')
    result=jsonencode(feval(s.config.navigation_context_callback,s));
    return
else
    % Default external ship-motion surrogate uses an ENU/up world frame.
    [eta,~,~]=s.ship(s.timestamp);
    phi=eta(4); theta=eta(5); psi=eta(6);
    Rx=[1 0 0;0 cos(phi) -sin(phi);0 sin(phi) cos(phi)];
    Ry=[cos(theta) 0 sin(theta);0 1 0;-sin(theta) 0 cos(theta)];
    Rz=[cos(psi) -sin(psi) 0;sin(psi) cos(psi) 0;0 0 1];
    R=[0 1 0;1 0 0;0 0 -1]*Rz*Ry*Rx;
end
result=jsonencode(struct('timestamp',s.timestamp,'R_ned_deck',R));
end
