function result=test_landing_stop(folder,geometryJSON)
% Exercise real Stop Simulation: early, boundary, and initially true.
g=jsondecode(geometryJSON);
for threshold=[.06 .04 0]
    file=jsondecode(landing_build_interface_fixture(folder,geometryJSON));
    m='landing_interface_fixture'; load_system(file);
    add_block('simulink/Sources/Clock',[m '/StopClock']);
    add_block('simulink/Logic and Bit Operations/Compare To Constant',[m '/StopCondition'], ...
        'const',num2str(threshold,17),'relop','>=');
    add_block('simulink/Sinks/Stop Simulation',[m '/Stop']);
    add_line(m,'StopClock/1','StopCondition/1');add_line(m,'StopCondition/1','Stop/1');
    save_system(m); close_system(m,0);
    c=struct('initialize_callback','landing_live_initialize','model',file, ...
        'geometry',g,'workdir',folder,'plant_step',.02,'seed',42, ...
        'gimbal',struct('encoder_std_rad',0), ...
        'inputs',struct('nav_position_ned','gear_relative_ned_m','nav_velocity_ned','velocity_gear_ned_mps', ...
            'nav_valid','valid','nav_timestamp','timestamp'), ...
        'outputs',struct('helicopter_state','helicopter_state','ship_state','ship_state'));
    cleanup=onCleanup(@()finalize_external_model());
    initialize_external_model(jsonencode(c));truth=jsondecode(read_sensor_truth());
    s=evalin('base','landingState'); initial=s.helicopter_state(1:3);
    e=struct('timestamp',0,'T_deck_camera',truth.T_deck_camera,'velocity',zeros(3,1), ...
        'covariance',eye(6)*.01,'healthy',true,'angular_velocity_camera',zeros(3,1), ...
        'gimbal',jsondecode(read_gimbal_telemetry()));
    write_navigation_estimate(jsonencode(e)); f=jsondecode(read_innerloop_feedback());
    advance_external_model(.1);
    s=evalin('base','landingState'); status=jsondecode(read_landing_live_status());
    assert(status.stopped && abs(s.timestamp-threshold)<1e-10);
    assert(norm(s.helicopter_state(1:3)-(initial-threshold*.02*f.gear_relative_ned_m(:)))<1e-9);
    assert(~isempty(status.stop_reason));
    read_sensor_truth(); % conversion still works with captured final state
    finalize_external_model();clear cleanup;
end
% Body velocity exported by Ship_motion must rotate into NED exactly once.
v=zeros(12,1);v(4)=10;v(9)=pi/2;
[~,ship]=landing_normalize_native(zeros(12,1),v,struct('ship_velocity_frame','body'));
assert(norm(ship(4:6)-[0;10;0])<1e-10);
result=jsonencode(struct('passed',true,'stop_times',[.06 .04 0], ...
    'terminal_native_state_preserved',true,'ship_velocity_rotation',true, ...
    'stop_reason',status.stop_reason));
end
