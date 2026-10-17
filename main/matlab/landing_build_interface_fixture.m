function filename=landing_build_interface_fixture(folder,geometryJSON)
% TEST ONLY: tunable feedback drives an integrator. Not an aircraft model.
g=jsondecode(geometryJSON); model='landing_interface_fixture';
if bdIsLoaded(model), close_system(model,0); end
new_system(model);
set_param(model,'SolverType','Fixed-step','Solver','ode4','FixedStep','0.01');
inputs={'nav_position_ned',3;'nav_velocity_ned',3;'nav_valid',1;'nav_timestamp',1};
for k=1:size(inputs,1)
    add_block('simulink/Sources/In1',[model '/' inputs{k,1}], ...
        'Port',num2str(k),'PortDimensions',num2str(inputs{k,2}));
end
add_block('simulink/Math Operations/Gain',[model '/TestGain'],'Gain','-0.02');
add_block('simulink/Math Operations/Product',[model '/ValidGate'],'Inputs','**');
add_block('simulink/Continuous/Integrator',[model '/StateMemory'], ...
    'InitialCondition',mat2str(g.H_ship_body(:)-g.gear_body(:)+[0;0;-30],17));
add_block('simulink/Math Operations/Gain',[model '/Knots'],'Gain','3600/1852');
add_block('simulink/Sources/Constant',[model '/ZeroAnglesRates'],'Value','zeros(6,1)');
add_block('simulink/Signal Routing/Mux',[model '/NativeHelicopter'],'Inputs','3');
add_block('simulink/Sources/Constant',[model '/NativeShip'],'Value','zeros(12,1)');
add_block('simulink/Sinks/Out1',[model '/helicopter_state'],'Port','1');
add_block('simulink/Sinks/Out1',[model '/ship_state'],'Port','2');
add_block('simulink/Sinks/Terminator',[model '/UnusedVelocity']);
add_block('simulink/Sinks/Terminator',[model '/UnusedTime']);
links={'nav_position_ned/1','TestGain/1';'TestGain/1','ValidGate/1'; ...
    'nav_valid/1','ValidGate/2';'ValidGate/1','StateMemory/1'; ...
    'ValidGate/1','Knots/1';'StateMemory/1','NativeHelicopter/1'; ...
    'Knots/1','NativeHelicopter/2';'ZeroAnglesRates/1','NativeHelicopter/3'; ...
    'NativeHelicopter/1','helicopter_state/1';'NativeShip/1','ship_state/1'; ...
    'nav_velocity_ned/1','UnusedVelocity/1';'nav_timestamp/1','UnusedTime/1'};
for k=1:size(links,1), add_line(model,links{k,1},links{k,2}); end
Simulink.Annotation(model,'INTERFACE ACCEPTANCE FIXTURE ONLY; not helicopter dynamics or guidance.');
filename=fullfile(folder,[model '.slx']); save_system(model,filename); close_system(model,0);
% MatlabSession parses JSON strings; return an encoded path.
filename=jsonencode(filename);
end
