function result=test_landing_live(folder,geometryJSON)
% Real Simulink integrator test, never presented as helicopter flight validation.
g=jsondecode(geometryJSON);
% Existing controller coordinate helpers must resolve inside this repository.
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
assert(startsWith(which('bodyToNedVel'),root) && startsWith(which('nedVelToBody'),root));
forward=bodyToNedVel; reverse=nedVelToBody;
angles=[.3;-.4;.7]; velocity=[17;-8;3];
ned=forward([velocity;angles]);
assert(norm(reverse(ned,angles)-velocity)<1e-12);
release(forward); release(reverse);
assert(norm(forward([1;0;0;0;0;pi/2])-[0;1;0])<1e-12);
release(forward);
file=jsondecode(landing_build_interface_fixture(folder,geometryJSON));
c=struct('initialize_callback','landing_live_initialize','model',file, ...
    'geometry',g,'workdir',folder,'plant_step',.01,'seed',42, ...
    'gimbal',struct('encoder_std_rad',0), ...
    'inputs',struct('nav_position_ned','gear_relative_ned_m','nav_velocity_ned','velocity_gear_ned_mps', ...
        'nav_valid','valid','nav_timestamp','timestamp'), ...
    'outputs',struct('helicopter_state','helicopter_state','ship_state','ship_state'));
% Confirm that an established initialization script feeds model base variables.
script=fullfile(folder,'fixture_setup.m'); fid=fopen(script,'w');
fprintf(fid,'fixture_origin=%s;\n',mat2str(g.H_ship_body(:)-g.gear_body(:)+[0;0;-30],17));
fclose(fid); c.setup_script=script;
load_system(file); set_param('landing_interface_fixture/StateMemory','InitialCondition','fixture_origin');
save_system('landing_interface_fixture'); close_system('landing_interface_fixture',0);
before=fileread_binary(file);
cleanup=onCleanup(@()finalize_external_model());
initialize_external_model(jsonencode(c));
must_fail(@()advance_external_model(.1),'landing:Order');
e=struct('timestamp',0,'T_deck_camera',eye(4),'velocity',zeros(3,1), ...
    'covariance',eye(6)*.01,'healthy',true,'angular_velocity_camera',zeros(3,1));
must_fail(@()write_navigation_estimate(jsonencode(e)),'landing:Order');
truth=jsondecode(read_sensor_truth());
s=evalin('base','landingState'); initial=s.helicopter_state(1:3);
% The optimized conversion path must match all fifteen actual SLX outputs.
reference=s; reference.config.conversion_evaluation='slx';
a=innerloop_evaluate(reference); b=innerloop_evaluate(s);
for field=fieldnames(a)'
    assert(norm(a.(field{1})(:)-b.(field{1})(:))<1e-9,'landing:Kernel','SLX/kernel mismatch');
end
e.T_deck_camera=truth.T_deck_camera; e.gimbal=jsondecode(read_gimbal_telemetry());
write_navigation_estimate(jsonencode(e)); f=jsondecode(read_innerloop_feedback());
assert(f.valid);
must_fail(@()write_navigation_estimate(jsonencode(e)),'landing:Order');
advance_external_model(.1);
s=evalin('base','landingState');
assert(norm(s.helicopter_state(1:3)-(initial-.02*f.gear_relative_ned_m(:)*.1))<1e-9, ...
    'landing:FirstInput','Feedback must affect the very first integration interval');
must_fail(@()advance_external_model(.1),'landing:Order');
must_fail(@()write_navigation_estimate(jsonencode(e)),'landing:Timestamp');
truth=jsondecode(read_sensor_truth()); e.timestamp=.1;
e.T_deck_camera=truth.T_deck_camera; e.T_deck_camera(1,4)=e.T_deck_camera(1,4)+5;
e.gimbal=jsondecode(read_gimbal_telemetry()); previous=s.helicopter_state(1:3);
write_navigation_estimate(jsonencode(e)); f=jsondecode(read_innerloop_feedback());
advance_external_model(.1); s=evalin('base','landingState');
assert(norm(s.helicopter_state(1:3)-(previous-.02*f.gear_relative_ned_m(:)*.1))<1e-9, ...
    'landing:StateMemory','State must persist, and changed feedback must be consumed');
truth=jsondecode(read_sensor_truth()); e.timestamp=.2; e.T_deck_camera=truth.T_deck_camera;
e.gimbal=jsondecode(read_gimbal_telemetry()); e.healthy=false;
previous=s.helicopter_state;
write_navigation_estimate(jsonencode(e)); advance_external_model(.1); s=evalin('base','landingState');
assert(norm(s.helicopter_state(1:3)-previous(1:3))<1e-10,'landing:Invalid','Fixture must consume nav_valid');
assert(isequal(before,fileread_binary(file)),'landing:Source','Source model was modified');
finalize_external_model();
c.model=fullfile(root,'external','innerloop','simu','innerLoop.slx');
must_fail(@()initialize_external_model(jsonencode(c)),'landing:Ports');
% A completed aircraft model may retain the user's requested name innerLoop.
% It must not be re-executed as the old 15-input conversion-only interface.
if bdIsLoaded('innerLoop'), close_system('innerLoop',0); end
load_system(file);
c.model=fullfile(folder,'innerLoop.slx');
save_system('landing_interface_fixture',c.model); close_system('landing_interface_fixture',0);
initialize_external_model(jsonencode(c));
truth=jsondecode(read_sensor_truth());
assert(norm(truth.gear_position_deck(:)-[0;0;30])<1e-9);
e.timestamp=0; e.T_deck_camera=truth.T_deck_camera; e.gimbal=jsondecode(read_gimbal_telemetry());
write_navigation_estimate(jsonencode(e)); advance_external_model(.01);
finalize_external_model();
result=jsonencode(struct('passed',true,'first_feedback_consumed',true, ...
    'state_preserved',true,'changed_feedback_consumed',true,'invalid_feedback_consumed',true, ...
    'source_unchanged',true,'slx_kernel_equivalent',true, ...
    'base_workspace_setup',true,'interface_only_model_rejected',true, ...
    'full_model_named_innerLoop_supported',true,'time',s.timestamp));
end
function bytes=fileread_binary(file)
fid=fopen(file,'rb'); cleanup=onCleanup(@()fclose(fid));
bytes=fread(fid,inf,'*uint8');
end
function must_fail(callback,expected)
try
    callback();
catch err
    assert(strcmp(err.identifier,expected),'Expected %s, got %s',expected,err.identifier); return;
end
error('landing:Test','Expected %s',expected);
end
