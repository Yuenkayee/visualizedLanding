function s=innerloop_update_gimbal(s)
c=s.config.gimbal;
if isfield(s,'gimbal_reference')
    c.R_deck_camera_reference=s.gimbal_reference.R_deck_camera;
end
s.gimbal_state=gimbal_step(s.gimbal_state,s.timestamp, ...
    rotation(s.helicopter_state(7:9)),rotation(s.ship_state(7:9))*diag([1 -1 -1]), ...
    s.config.geometry.T_body_camera,c);
s.gimbal_measurement=gimbal_sample(s.gimbal_state,s.config.geometry.T_body_camera,c);
end
function R=rotation(e)
r=e(1);p=e(2);y=e(3);
R=[cos(y) -sin(y) 0;sin(y) cos(y) 0;0 0 1]* ...
  [cos(p) 0 sin(p);0 1 0;-sin(p) 0 cos(p)]*[1 0 0;0 cos(r) -sin(r);0 sin(r) cos(r)];
end
