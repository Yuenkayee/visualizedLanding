function result=test_landing_batch(configJSON)
% Integration check on a disposable execution copy, never a flight result.
% Requires the actual model and its licensed toolboxes; run from Python test harness.
c=jsondecode(configJSON);
cleanup=onCleanup(@()finalize_external_model());
initialize_external_model(configJSON);
s=evalin('base','landingState');
initial=s.batch_telemetry;
assert(all(isfinite([initial.ship_eta;initial.ship_nu;initial.ship_eta_dot;initial.ship_nu_dot])));
assert(~initial.touchdown_stop,'landing:BatchTest','Unexpected initial touchdown');
% Finite StopTime exercises terminal retained observers without satisfying
% the original landing STOP. It must NOT be classified as a touchdown.
set_param(s.live_model,'StopTime','0.04');
for k=1:4
    truth=jsondecode(read_sensor_truth()); s=evalin('base','landingState');
    e=struct('timestamp',s.timestamp,'T_deck_camera',truth.T_deck_camera, ...
        'velocity',zeros(3,1),'covariance',eye(6)*.01,'healthy',true, ...
        'angular_velocity_camera',zeros(3,1),'gimbal',jsondecode(read_gimbal_telemetry()));
    write_navigation_estimate(jsonencode(e));read_innerloop_feedback();advance_external_model(.02);
    status=jsondecode(read_landing_live_status());
    if status.stopped, break; end
end
assert(status.stopped && abs(status.timestamp-.04)<1e-9);
t=status.batch_telemetry;
assert(~t.touchdown_stop,'landing:BatchTest','Time limit incorrectly sets landing STOP');
assert(all(isfinite([t.ship_eta;t.ship_nu;t.ship_eta_dot;t.ship_nu_dot])));
assert(numel(t.ship_eta)==6 && numel(t.ship_nu_dot)==6);
result=jsonencode(struct('passed',true,'terminal_timestamp',status.timestamp, ...
    'stop_reason',status.stop_reason,'actual_telemetry',t,'initial_telemetry',initial));
end
