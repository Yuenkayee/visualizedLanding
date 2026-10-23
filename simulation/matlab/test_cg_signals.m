function result=test_cg_signals()
% Geometry and velocity checks use nonzero ship/aircraft rotations and rates.
root=fileparts(fileparts(fileparts(mfilename('fullpath')))); setup_paths(root);
Rb=expm(skew([.2;-.1;.4])); Rs=expm(skew([-.1;.15;-.2]));
Rc=[0 -1 0;1 0 0;0 0 1]; lc=[2;0;1.11]; H=[-45;0;-5.5]; gear=[0;0;2.9];
heli=[10;20;-50;8;2;-1;angles(Rb);.03;-.02;.05];
ship=[95;3;0;7;1;0;angles(Rs);-.01;.04;-.02];
mount=[Rc lc;0 0 0 1]; P=diag([.01 .02 .03 .04 .05 .06]);
fields=innerloop_output_fields(); a=cell(1,numel(fields));
[a{:}]=innerLoop_conversion(0,heli,ship,mount,H,gear,eye(4),zeros(3,1),P,0,0, ...
    zeros(3,1),Rc,zeros(3,1),eye(3)*1e-5,1,0,zeros(3,1));
o=cell2struct(a,fields,2); Rdeck=Rs*diag([1 -1 -1]);
omega=Rc'*(heli(10:12)-Rb'*Rs*ship(10:12));
args={0,heli,ship,mount,H,gear,o.T_deck_camera,o.velocity_deck,P,1,0, ...
    zeros(3,1),Rc,zeros(3,1),eye(3)*1e-5,1,0,omega};
[a{:}]=innerLoop_conversion(args{:}); f=cell2struct(a,fields,2);
assert(norm(f.cg_relative_truth_ned-(heli(1:3)-ship(1:3)))<1e-12);
assert(norm(f.cg_relative_estimate_ned-f.cg_relative_truth_ned)<1e-10);
assert(norm(f.navigation_velocity_cg_ned_mps-(f.helicopter_velocity_ned_mps-f.ship_velocity_ned_mps))<1e-10);
assert(min(eig(f.cg_relative_covariance_ned))>0);
J=[Rdeck -Rdeck*o.T_deck_camera(1:3,1:3)*skew(-Rc'*lc)];
Jg=-J(:,4:6);
assert(norm(f.cg_relative_covariance_ned-(J*P*J'+Jg*eye(3)*1e-5*Jg'),'fro')<1e-10);
% Estimate must not change if helicopter truth or ship CG origin is changed.
biased=args; biased{2}(1:3)=[100;200;-300]; biased{2}(7:12)=0;
biased{3}(1:3)=[-10;-20;0];
[a{:}]=innerLoop_conversion(biased{:}); b=cell2struct(a,fields,2);
assert(norm(b.cg_relative_estimate_ned-f.cg_relative_estimate_ned)<1e-12);
assert(norm(b.navigation_velocity_cg_ned_mps-f.navigation_velocity_cg_ned_mps)<1e-12);
assert(norm(b.cg_relative_truth_ned-f.cg_relative_truth_ned)>100);
biased=args; biased{7}(1:3,4)=biased{7}(1:3,4)+[1;2;3];
[a{:}]=innerLoop_conversion(biased{:}); b=cell2struct(a,fields,2);
assert(norm(b.cg_relative_estimate_ned-f.cg_relative_truth_ned-Rdeck*[1;2;3])<1e-10);
% Finite difference of actual CG separation with a rotating ship/deck.
dt=1e-6;
r2=(heli(1:3)+f.helicopter_velocity_ned_mps*dt)-(ship(1:3)+f.ship_velocity_ned_mps*dt);
assert(norm((r2-f.cg_relative_truth_ned)/dt-f.navigation_velocity_cg_ned_mps)<1e-7);
% convert's per-instance state and the host helper must agree at every tick.
object=convert; object.T_body_camera=mount; object.H_ship_body=H; object.gear_body=gear;
cleanup=onCleanup(@()release(object));
state=cg_signal_initial_state(); outputs=cell(1,23); trace=zeros(101,8);
for k=0:100
    t=k*.02; args{1}=t; args{11}=t; args{17}=t;
    args{7}(1:3,4)=o.T_deck_camera(1:3,4)+[.4*(k>=20);0;0];
    args{10}=double(k<40 || k>=70);
    [a{:}]=innerLoop_conversion(args{:}); raw=cell2struct(a,fields,2);
    [state,p,v,ok,age]=cg_signal_step(state,t,raw.cg_relative_estimate_ned, ...
        raw.navigation_velocity_cg_ned_mps,raw.feedback_valid,t,[.04;5;.3;1]);
    stepArgs=[args(1:3) args(7:end)];
    [outputs{:}]=step(object,stepArgs{:});
    assert(norm(outputs{20}-p)<1e-12 && norm(outputs{21}-v)<1e-12);
    assert(outputs{22}==ok && abs(outputs{23}-age)<1e-12);
    trace(k+1,:)=[p' v' ok age];
end
assert(all(trace(41:70,7)==0));
reset(object); [outputs{:}]=step(object,stepArgs{:});
assert(norm(outputs{20}-raw.cg_relative_estimate_ned)<1e-12);
% 10/5 Hz held input packets still produce an exact constant-velocity 50 Hz ramp.
for stride=[1 5 10]
    state=cg_signal_initial_state(); origin=[-95;2;-50]; speed=[4;-1;.2];
    for k=0:100
        t=k*.02; sample=floor(k/stride)*stride*.02;
        [state,p,~,ok,age]=cg_signal_step(state,t,origin+speed*sample,speed,1,sample,[.04;5;.3;1]);
        assert(norm(p-origin-speed*t)<1e-10 && ok && abs(age-(t-sample))<1e-10);
    end
end
for k=101:125
    [state,p,~,ok,~]=cg_signal_step(state,k*.02,origin+speed*2,speed,1,2,[.04;5;.3;1]);
end
assert(~ok && norm(p-origin-speed*2.3)<1e-10);
result=jsonencode(struct('passed',true,'rotated_geometry',true,'cg_velocity_transport',true, ...
    'truth_isolation',true,'system_host_state_parity',true,'reset',true,'multirate_and_expiry',true));
end
function S=skew(v)
S=[0 -v(3) v(2);v(3) 0 -v(1);-v(2) v(1) 0];
end
function e=angles(R)
e=[atan2(R(3,2),R(3,3));asin(-R(3,1));atan2(R(2,1),R(1,1))];
end
