function [T_world_deck,T_deck_camera,velocity_deck,v_heli_ned,v_ship_ned, ...
    r_truth,r_est,distance_est,P_relative,feedback_valid,v_camera_est_ned,t_out] = ...
    convert(t,heli,ship,T_body_camera,H_body,gear_body,nav_T,nav_v,nav_P,nav_ok,nav_t,nav_R)
%#codegen
% BODY is forward/right/down; deck is bow/port/up. Euler Rz Ry Rx.
% nav_R MUST come from available attitude/navigation information, not truth.
assert(isfinite(t) && t>=0);
assert(all(isfinite(heli)) && all(isfinite(ship)));
assert(all(isfinite(T_body_camera(:))) && all(isfinite(H_body)) && all(isfinite(gear_body)));
Rb=euler_rotation(heli(7:9)); Rs=euler_rotation(ship(7:9));
Rc=T_body_camera(1:3,1:3); lc=T_body_camera(1:3,4);
assert(norm(Rc'*Rc-eye(3),'fro')<1e-6 && det(Rc)>0);
assert(norm(T_body_camera(4,:)-[0 0 0 1])<1e-8);
D=diag([1 -1 -1]); A=[0 1 0;1 0 0;0 0 -1];
Rdeck=Rs*D; pc=heli(1:3)+Rb*lc; ph=ship(1:3)+Rs*H_body;
T_world_deck=[A*Rdeck A*ph;0 0 0 1];
T_deck_camera=[Rdeck'*Rb*Rc Rdeck'*(pc-ph);0 0 0 1];
knot=1852/3600;
v_heli_ned=Rb*(heli(4:6)*knot); v_ship_ned=ship(4:6)*knot;
v_camera=v_heli_ned+Rb*cross(heli(10:12),lc);
v_H=v_ship_ned+Rs*cross(ship(10:12),H_body);
omega_deck=D*ship(10:12);
velocity_deck=Rdeck'*(v_camera-v_H)-cross(omega_deck,T_deck_camera(1:3,4));
r_truth=heli(1:3)+Rb*gear_body-ph;
% Navigation covariance: deck translation + camera-local right angle error.
lever_camera=Rc'*(gear_body-lc);
r_deck=nav_T(1:3,4)+nav_T(1:3,1:3)*lever_camera;
r_est=nav_R*r_deck; distance_est=norm(r_est);
J=[nav_R -nav_R*nav_T(1:3,1:3)*skew3(lever_camera)];
P_relative=J*nav_P*J'; P_relative=(P_relative+P_relative')/2;
% Camera velocity expressed in NED, not gear-relative derivative. The latter
% needs estimated angular rates, intentionally not taken from plant truth.
v_camera_est_ned=nav_R*nav_v;
feedback_valid=double(nav_ok~=0 && abs(nav_t-t)<1e-8 && ...
    all(isfinite(nav_T(:))) && all(isfinite(nav_P(:))) && ...
    all(isfinite(nav_v)) && all(isfinite(nav_R(:))) && ...
    norm(nav_R'*nav_R-eye(3),'fro')<1e-6 && det(nav_R)>0 && ...
    norm(nav_T(1:3,1:3)'*nav_T(1:3,1:3)-eye(3),'fro')<1e-6 && ...
    det(nav_T(1:3,1:3))>0 && norm(nav_T(4,:)-[0 0 0 1])<1e-8 && ...
    norm(nav_P-nav_P','fro')<1e-6 && min(eig((nav_P+nav_P')/2))>=-1e-9);
t_out=t;
end
function R=euler_rotation(e)
a=e(1); b=e(2); c=e(3);
R=[cos(c) -sin(c) 0;sin(c) cos(c) 0;0 0 1]* ...
  [cos(b) 0 sin(b);0 1 0;-sin(b) 0 cos(b)]* ...
  [1 0 0;0 cos(a) -sin(a);0 sin(a) cos(a)];
end
function S=skew3(v)
S=[0 -v(3) v(2);v(3) 0 -v(1);-v(2) v(1) 0];
end
