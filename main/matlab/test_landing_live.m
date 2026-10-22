function result=test_landing_live(folder,geometryJSON)
% Real Simulink integrator test, never presented as helicopter flight validation.
g=jsondecode(geometryJSON);
timingFile=fullfile(folder,'plant_steps.jsonl'); previousRecords=0;
if isfile(timingFile)
    previousRecords=numel(splitlines(strtrim(fileread(timingFile))));
end
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
    'geometry',g,'workdir',folder,'plant_step',.02,'seed',42, ...
    'gimbal',struct('encoder_std_rad',0), ...
    'inputs',struct('nav_position_ned','gear_relative_ned_m','nav_velocity_ned','velocity_gear_ned_mps', ...
        'nav_valid','valid','nav_timestamp','timestamp'), ...
    'outputs',struct('helicopter_state','helicopter_state','ship_state','ship_state'));
% Confirm that an established initialization script feeds model base variables.
script=fullfile(folder,'fixture_setup.m'); fid=fopen(script,'w');
fprintf(fid,'fixture_origin=%s;\n',mat2str(g.H_ship_body(:)-g.gear_body(:)+[0;0;-30],17));
fclose(fid); c.setup_script=script;
load_system(file);
set_param('landing_interface_fixture','FixedStep','0.01');
must_fail(@()landing_check_timing('landing_interface_fixture',.02),'landing:Timing');
set_param('landing_interface_fixture','FixedStep','0.02');
add_block('simulink/User-Defined Functions/MATLAB System','landing_interface_fixture/TimingProbe', ...
    'System','CETI','SimulateUsing','Interpreted execution');
set_param('landing_interface_fixture/TimingProbe','Ts','0.01');
must_fail(@()landing_check_timing('landing_interface_fixture',.02),'landing:Timing');
set_param('landing_interface_fixture/TimingProbe','Ts','0.02');
landing_check_timing('landing_interface_fixture',.02);
delete_block('landing_interface_fixture/TimingProbe');
set_param('landing_interface_fixture/StateMemory','InitialCondition','fixture_origin');
save_system('landing_interface_fixture'); close_system('landing_interface_fixture',0);
before=fileread_binary(file);
cleanup=onCleanup(@()finalize_external_model());
initialize_external_model(jsonencode(c));
must_fail(@()advance_external_model(.02),'landing:Order');
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
must_fail(@()advance_external_model(.05),'landing:Time');
advance_external_model(.02);
s=evalin('base','landingState');
assert(abs(s.timestamp-.02)<1e-12);
assert(norm(s.helicopter_state(1:3)-(initial-.02*f.gear_relative_ned_m(:)*.02))<1e-9, ...
    'landing:FirstInput','Feedback must affect the very first integration interval');
must_fail(@()advance_external_model(.02),'landing:Order');
must_fail(@()write_navigation_estimate(jsonencode(e)),'landing:Timestamp');
truth=jsondecode(read_sensor_truth()); e.timestamp=.02;
e.T_deck_camera=truth.T_deck_camera; e.T_deck_camera(1,4)=e.T_deck_camera(1,4)+5;
e.gimbal=jsondecode(read_gimbal_telemetry()); previous=s.helicopter_state(1:3);
write_navigation_estimate(jsonencode(e)); f=jsondecode(read_innerloop_feedback());
advance_external_model(.02); s=evalin('base','landingState');
assert(abs(s.timestamp-.04)<1e-12);
assert(norm(s.helicopter_state(1:3)-(previous-.02*f.gear_relative_ned_m(:)*.02))<1e-9, ...
    'landing:StateMemory','State must persist, and changed feedback must be consumed');
truth=jsondecode(read_sensor_truth()); e.timestamp=.04; e.T_deck_camera=truth.T_deck_camera;
e.gimbal=jsondecode(read_gimbal_telemetry()); e.healthy=false;
previous=s.helicopter_state;
write_navigation_estimate(jsonencode(e)); advance_external_model(.02); s=evalin('base','landingState');
assert(abs(s.timestamp-.06)<1e-12);
assert(norm(s.helicopter_state(1:3)-previous(1:3))<1e-10,'landing:Invalid','Fixture must consume nav_valid');
% Exercise the complete 2 s / 100-step clock, including floating-point
% boundaries such as .12+.02 and .58+.02, with state held by invalid feedback.
truth=jsondecode(read_sensor_truth()); e.timestamp=s.timestamp;
e.T_deck_camera=truth.T_deck_camera; e.gimbal=jsondecode(read_gimbal_telemetry());
write_navigation_estimate(jsonencode(e)); advance_external_model(1.94);
s=evalin('base','landingState');
assert(abs(s.timestamp-2)<1e-12 && ~s.stopped);
assert(norm(s.helicopter_state(1:3)-previous(1:3))<1e-10,'landing:StateMemory','State changed under invalid feedback');
records=splitlines(strtrim(fileread(timingFile)));
assert(numel(records)==previousRecords+100);
records=records(previousRecords+1:end);
for k=1:100
    row=jsondecode(records{k});
    assert(abs(row.target-k*.02)<1e-12 && abs(row.actual_time-row.target)<1e-12);
    assert(~row.final_step && isempty(row.error));
end
% Deliberately request a wrong host period from the already compiled plant.
% A real time mismatch must still fail and retain evidence, not be relabeled
% as STOP or repaired by inventing a timestamp/additional integration step.
truth=jsondecode(read_sensor_truth()); e.timestamp=s.timestamp;
e.T_deck_camera=truth.T_deck_camera; e.gimbal=jsondecode(read_gimbal_telemetry());
write_navigation_estimate(jsonencode(e)); bad=evalin('base','landingState');
bad.config.plant_step=.04;
must_fail(@()landing_live_step(bad,.04),'landing:Time');
records=splitlines(strtrim(fileread(timingFile)));
row=jsondecode(records{end});
assert(strcmp(row.error_identifier,'landing:Time') && contains(row.error,'actual='));
assert(abs(row.target-2.04)<1e-12 && abs(row.actual_time-2.02)<1e-12 && ~row.stopped);
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
write_navigation_estimate(jsonencode(e)); advance_external_model(.02);
finalize_external_model();
result=jsonencode(struct('passed',true,'first_feedback_consumed',true, ...
    'state_preserved',true,'changed_feedback_consumed',true,'invalid_feedback_consumed',true, ...
    'source_unchanged',true,'slx_kernel_equivalent',true, ...
    'base_workspace_setup',true,'interface_only_model_rejected',true, ...
    'full_model_named_innerLoop_supported',true,'time',s.timestamp, ...
    'two_second_clock_and_step_diagnostics',true, ...
    'genuine_time_mismatch_rejected_and_logged',true, ...
    'plant_step',.02,'mismatched_solver_and_controller_rejected',true, ...
    'off_grid_advance_rejected',true));
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
