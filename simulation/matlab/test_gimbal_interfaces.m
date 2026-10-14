function result=test_gimbal_interfaces(parityFile)
% Servo physics + actual SLX body/gear feedback with changing mount angles.
root=fileparts(fileparts(fileparts(mfilename('fullpath')))); setup_paths(root);
c=gimbal_defaults(); neutral=[eye(3) [2;.1;1.11];0 0 0 1];
D=diag([1 -1 -1]); g=[]; errors=zeros(601,1); maximumRate=0;
for k=0:600
    t=k*.01;
    body=D*rotation(.6*sin(2*pi*t/3+[0;.7;1.4]));
    deck=rotation([.03*sin(t);.02*sin(t);0]);
    g=gimbal_step(g,t,body,deck,neutral,c); m=gimbal_sample(g,neutral,c);
    errorR=D'*deck'*body*m.T_body_camera(1:3,1:3);
    errors(k+1)=acos(max(-1,min(1,(trace(errorR)-1)/2)));
    assert(norm(m.T_body_camera(1:3,4)-neutral(1:3,4))==0);
    maximumRate=max(maximumRate,max(abs(g.rates)));
end
assert(max(errors)<.03 && max(errors)>1e-5 && maximumRate<=3);
if nargin>0
    fixture=jsondecode(fileread(parityFile)); cfg=gimbal_defaults(fixture.config); state=[];
    for k=1:numel(fixture.rows)
        row=fixture.rows(k);
        state=gimbal_step(state,row.timestamp,row.R_world_body,row.R_world_deck,fixture.neutral,cfg);
        assert(norm(state.angles-row.angles(:))<1e-10);
        assert(norm(state.rates-row.rates(:))<1e-10);
        measured=gimbal_sample(state,fixture.neutral,cfg);
        assert(norm(measured.T_body_camera-row.T_body_camera,'fro')<1e-10);
        assert(norm(measured.angular_velocity_camera(:)-row.angular_velocity_camera(:))<1e-10);
    end
end
config=struct('initialize_callback','innerloop_initialize','allow_state_hold',true, ...
    'helicopter_state',[10;20;-30;1;2;3;.4;-.3;.7;.8;-.5;.4], ...
    'ship_state',[1;2;0;0;0;0;.1;-.05;.2;.02;-.03;.01], ...
    'geometry',struct('T_body_camera',neutral,'H_ship_body',[-5;0;-3],'gear_body',[.2;-.1;2.9]));
cleanup=onCleanup(@()finalize_external_model()); %#ok<NASGU>
initialize_external_model(jsonencode(config));
truth=jsondecode(read_sensor_truth()); s=evalin('base','landingState');
m=jsondecode(read_gimbal_telemetry());
assert(norm(m.T_body_camera(1:3,1:3)-neutral(1:3,1:3),'fro')>.2);
Rb=rotation(s.helicopter_state(7:9)); Rs=rotation(s.ship_state(7:9));
Rc=m.T_body_camera(1:3,1:3); Rdc=truth.T_deck_camera(1:3,1:3);
omegaDeck=D*s.ship_state(10:12);
wc=Rc'*s.helicopter_state(10:12)+m.angular_velocity_camera(:)-Rdc'*omegaDeck;
e=struct('timestamp',s.timestamp,'T_deck_camera',truth.T_deck_camera, ...
    'velocity',truth.velocity_deck,'covariance',eye(6)*.01,'healthy',true, ...
    'angular_velocity_camera',wc,'gimbal',m);
write_navigation_estimate(jsonencode(e)); f=jsondecode(read_innerloop_feedback());
s=evalin('base','landingState'); o=innerloop_evaluate(s);
assert(f.valid && norm(f.gear_relative_ned_m(:)-o.gear_relative_truth_ned)<1e-9);
expected=Rb*(s.helicopter_state(4:6)*1852/3600+cross(s.helicopter_state(10:12),config.geometry.gear_body)) ...
    -s.ship_state(4:6)*1852/3600-Rs*cross(s.ship_state(10:12),config.geometry.H_ship_body);
assert(norm(f.velocity_gear_ned_mps(:)-expected)<1e-9);
% Two sensor-only advances can hold the last control sample.
advance_external_model(.03); read_sensor_truth(); advance_external_model(.02); read_sensor_truth();
s=evalin('base','landingState'); assert(abs(s.timestamp-.05)<1e-10);
% Missing/stale encoder cannot claim healthy BODY feedback.
s.estimate=e; s.estimate.timestamp=s.timestamp; s.estimate.gimbal.timestamp=0;
o=innerloop_evaluate(s); assert(~o.feedback_valid);
s.estimate=rmfield(s.estimate,'gimbal'); o=innerloop_evaluate(s); assert(~o.feedback_valid);
% A measured mount bias affects the estimate, never the rendering truth.
s.estimate=e; s.estimate.timestamp=s.timestamp; s.estimate.gimbal.timestamp=s.timestamp;
s.estimate.gimbal.T_body_camera(1:3,1:3)=Rc*rotation([.1;0;0]);
biased=innerloop_evaluate(s);
assert(norm(biased.T_deck_camera-o.T_deck_camera,'fro')<1e-12);
assert(norm(biased.gear_relative_estimate_ned-o.gear_relative_truth_ned)>.01);
result=jsonencode(struct('passed',true,'max_tracking_error_rad',max(errors), ...
    'max_axis_rate_rad_s',maximumRate,'checks','dynamic SLX extrinsics, gear position/velocity, stale encoders, asynchronous hold, Python parity'));
end
function R=rotation(e)
r=e(1);p=e(2);y=e(3);
R=[cos(y) -sin(y) 0;sin(y) cos(y) 0;0 0 1]* ...
  [cos(p) 0 sin(p);0 1 0;-sin(p) 0 cos(p)]*[1 0 0;0 cos(r) -sin(r);0 sin(r) cos(r)];
end
