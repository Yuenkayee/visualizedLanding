function [T_world_deck,T_deck_camera,velocity_deck,v_heli_ned,v_ship_ned, ...
    r_truth,r_est,distance_est,P_relative,feedback_valid,v_camera_est_ned,t_out, ...
    actual_mount,nav_T_deck_body,v_gear_est_ned, ...
    cg_truth,cg_est,v_cg_est_ned,P_cg] = ...
    innerLoop_conversion(t,heli,ship,T_body_camera,H_body,gear_body,nav_T,nav_v,nav_P,nav_ok,nav_t, ...
    gimbal_q,nav_Rbc,nav_mount_omega,nav_mount_P,nav_mount_ok,nav_mount_t,nav_omega)
%#codegen
% BODY is forward/right/down; deck is bow/port/up. Euler Rz Ry Rx.
% Ship Euler angles are available through ideal zero-delay communication.
assert(isfinite(t) && t>=0);
assert(all(isfinite(heli)) && all(isfinite(ship)));
assert(all(isfinite(T_body_camera(:))) && all(isfinite(H_body)) && all(isfinite(gear_body)));
Rb=euler_rotation(heli(7:9)); Rs=euler_rotation(ship(7:9));
Rc=euler_rotation(gimbal_q)*T_body_camera(1:3,1:3); lc=T_body_camera(1:3,4);
actual_mount=[Rc lc;0 0 0 1];
assert(norm(Rc'*Rc-eye(3),'fro')<1e-6 && det(Rc)>0);
assert(norm(T_body_camera(4,:)-[0 0 0 1])<1e-8);
D=diag([1 -1 -1]); A=[0 1 0;1 0 0;0 0 -1];
Rdeck=Rs*D; nav_R=Rdeck; pc=heli(1:3)+Rb*lc; ph=ship(1:3)+Rs*H_body;
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
lever_camera=nav_Rbc'*(gear_body-lc);
r_deck=nav_T(1:3,4)+nav_T(1:3,1:3)*lever_camera;
r_est=nav_R*r_deck; distance_est=norm(r_est);
J=[nav_R -nav_R*nav_T(1:3,1:3)*skew3(lever_camera)];
Jg=nav_R*nav_T(1:3,1:3)*skew3(lever_camera);
P_relative=J*nav_P*J'+Jg*nav_mount_P*Jg'; P_relative=(P_relative+P_relative')/2;
% Camera velocity expressed in NED, not gear-relative derivative. The latter
% needs estimated angular rates, intentionally not taken from plant truth.
v_camera_est_ned=nav_R*nav_v;
nav_T_deck_body=nav_T*[nav_Rbc' -nav_Rbc'*lc;0 0 0 1];
v_gear_deck=nav_v+nav_T(1:3,1:3)*cross(nav_omega-nav_mount_omega,lever_camera);
v_gear_est_ned=nav_R*(v_gear_deck+cross(omega_deck,r_deck));
% Helicopter CG MINUS ship CG. Only the truth output reads helicopter truth;
% the estimate uses navigation + encoder + communicated ship attitude/rates.
cg_truth=heli(1:3)-ship(1:3);
lever_cg=-nav_Rbc'*lc;
cg_deck=nav_T_deck_body(1:3,4);
cg_est=nav_R*cg_deck+Rs*H_body;
v_cg_deck=nav_v+nav_T(1:3,1:3)*cross(nav_omega-nav_mount_omega,lever_cg);
v_cg_est_ned=nav_R*(v_cg_deck+cross(omega_deck,cg_deck))+Rs*cross(ship(10:12),H_body);
Jcg=[nav_R -nav_R*nav_T(1:3,1:3)*skew3(lever_cg)];
Jcg_g=nav_R*nav_T(1:3,1:3)*skew3(lever_cg);
P_cg=Jcg*nav_P*Jcg'+Jcg_g*nav_mount_P*Jcg_g'; P_cg=(P_cg+P_cg')/2;
feedback_valid=double(isfinite(nav_ok) && nav_ok~=0 && abs(nav_t-t)<1e-8 && ...
    isfinite(nav_mount_ok) && nav_mount_ok~=0 && abs(nav_mount_t-t)<1e-8 && ...
    all(isfinite(nav_Rbc(:))) && norm(nav_Rbc'*nav_Rbc-eye(3),'fro')<1e-6 && det(nav_Rbc)>0 && ...
    all(isfinite(nav_mount_P(:))) && norm(nav_mount_P-nav_mount_P','fro')<1e-6 && ...
    min(eig((nav_mount_P+nav_mount_P')/2))>=-1e-9 && ...
    all(isfinite(nav_omega)) && all(isfinite(nav_mount_omega)) && ...
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
