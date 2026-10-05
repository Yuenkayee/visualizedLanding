function result=test_innerloop_interfaces()
% Actual SLX execution; offsets below are asymmetric synthetic fixtures,
% NOT certified UH-60 dimensions. No aircraft/controller dynamics tested.
root=fileparts(fileparts(fileparts(mfilename('fullpath')))); setup_paths(root);
c=struct('initialize_callback','innerloop_initialize','allow_state_hold',true);
c.helicopter_state=[10;20;-30;10;0;0;0;0;pi/2;0;0;0];
c.ship_state=[1;2;0;0;20;0;0;0;0;0;0;0];
c.geometry=struct('T_body_camera',[diag([1 -1 -1]) [1;0;2];0 0 0 1], ...
    'H_ship_body',[-5;0;-3],'gear_body',[0;0;2]);
c.nav_R_ned_deck=diag([1 -1 -1]);
cleanup=onCleanup(@()finalize_external_model());
initialize_external_model(jsonencode(c));
% Advancement and feedback must follow a current-state read.
expect_error(@()advance_external_model(.1),'landing:Order');
e=struct('timestamp',0,'T_deck_camera',eye(4),'velocity',[0 0 0], ...
    'covariance',eye(6)*.01,'healthy',true);
expect_error(@()write_navigation_estimate(jsonencode(e)),'landing:Order');
truth=jsondecode(read_sensor_truth());
s=evalin('base','landingState'); o=innerloop_evaluate(s);
assert(norm(o.helicopter_velocity_ned_mps-[0;10*1852/3600;0])<1e-10);
assert(norm(o.ship_velocity_ned_mps-[0;20*1852/3600;0])<1e-10);
assert(norm(o.gear_relative_truth_ned-[14;18;-25])<1e-10);
assert(norm(truth.T_world_deck(1:3,4)-[2;-4;3])<1e-10);
e.T_deck_camera=truth.T_deck_camera;
write_navigation_estimate(jsonencode(e));
f=jsondecode(read_innerloop_feedback());
assert(f.valid && norm(f.gear_relative_ned_m(:)-[14;18;-25])<1e-10);
assert(abs(f.distance_m-norm([14;18;-25]))<1e-10);
assert(min(eig(f.covariance_ned))>0);
expect_error(@()write_navigation_estimate(jsonencode(e)),'landing:Order');
advance_external_model(.1);
expect_error(@()write_navigation_estimate(jsonencode(e)),'landing:Timestamp');
truth2=jsondecode(read_sensor_truth()); assert(abs(truth2.timestamp-.1)<1e-10);
e.timestamp=.1; e.T_deck_camera=truth2.T_deck_camera; e.healthy=false;
write_navigation_estimate(jsonencode(e)); f=jsondecode(read_innerloop_feedback()); assert(~f.valid);
% No hidden surrogate: missing plant raises explicitly.
s=evalin('base','landingState'); s.config.allow_state_hold=false; assignin('base','landingState',s);
expect_error(@()advance_external_model(.1),'landing:MissingDynamics');
% Signed height: negative NED down means wheel plane is above deck.
% Under ship rotation, offsets and velocities must rotate, not swap axes.
s.ship_state(7:9)=[.1;-.2;.3]; s.helicopter_state(7:9)=[-.15;.1;-.4];
s.ship_state(10:12)=[.01;-.02;.03]; s.helicopter_state(10:12)=[-.04;.05;.02];
s.estimate=[]; o=innerloop_evaluate(s);
% The sensor-frame matrices compose back to helicopter camera in ENU.
A=[0 1 0;1 0 0;0 0 -1]; Rb=euler_rot(s.helicopter_state(7:9)); Rs=euler_rot(s.ship_state(7:9));
T=o.T_world_deck*o.T_deck_camera;
assert(norm(T(1:3,4)-A*(s.helicopter_state(1:3)+Rb*c.geometry.T_body_camera(1:3,4)))<1e-9);
assert(norm(T(1:3,1:3)-A*Rb*c.geometry.T_body_camera(1:3,1:3),'fro')<1e-9);
% Finite-difference rotating lever velocity and rotating deck coordinates.
dt=1e-6; w=s.helicopter_state(10:12); ws=s.ship_state(10:12);
Rc2=Rb*expm(skew(w)*dt); Rs2=Rs*expm(skew(ws)*dt); D=diag([1 -1 -1]);
pc2=s.helicopter_state(1:3)+o.helicopter_velocity_ned_mps*dt+Rc2*c.geometry.T_body_camera(1:3,4);
ph2=s.ship_state(1:3)+o.ship_velocity_ned_mps*dt+Rs2*c.geometry.H_ship_body;
p2=(Rs2*D)'*(pc2-ph2);
assert(norm((p2-o.T_deck_camera(1:3,4))/dt-o.velocity_deck)<1e-5);
% Stale navigation packet fails validity directly inside SLX conversion.
s.estimate=e; s.estimate.timestamp=s.timestamp-.05;
o=innerloop_evaluate(s); assert(o.feedback_valid==0);
result=jsonencode(struct('passed',true,'checks', ...
    'SLX compilation, units, rotated frames, reference offsets, covariance, sensor contract, order, stale feedback, absent dynamics'));
end
function expect_error(f,id)
try
    f();
catch ex
    assert(strcmp(ex.identifier,id),'Expected %s but got %s',id,ex.identifier); return
end
error('landing:Test','Expected error %s',id);
end
function R=euler_rot(e)
a=e(1);b=e(2);c=e(3);
R=[cos(c) -sin(c) 0;sin(c) cos(c) 0;0 0 1]*[cos(b) 0 sin(b);0 1 0;-sin(b) 0 cos(b)]*[1 0 0;0 cos(a) -sin(a);0 sin(a) cos(a)];
end
function S=skew(v)
S=[0 -v(3) v(2);v(3) 0 -v(1);-v(2) v(1) 0];
end
