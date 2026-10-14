function out=innerloop_evaluate(s)
% Pure snapshot evaluation. Running StopTime=0 never advances a plant.
persistent loaded
% Generated Simulink artifacts stay outside the source tree.
previous=Simulink.fileGenControl('getConfig');
cache=fullfile(tempdir,'visualizedLanding_innerloop_cache');
Simulink.fileGenControl('set','CacheFolder',cache, ...
    'CodeGenFolder',fullfile(cache,'codegen'),'createDir',true);
restore=onCleanup(@()Simulink.fileGenControl('setConfig','config',previous)); %#ok<NASGU>
if isempty(loaded) || ~bdIsLoaded('innerLoop')
    load_system(fullfile(s.config.interface_root,'innerLoop.slx')); loaded=true;
end
g=s.config.geometry;
if isempty(s.estimate)
    e=struct('T_deck_camera',eye(4),'velocity',zeros(3,1), ...
        'covariance',eye(6),'healthy',false,'timestamp',s.timestamp);
else
    e=s.estimate;
end
% Missing telemetry is explicitly invalid, never substituted by simulated truth.
m=struct('T_body_camera',g.T_body_camera,'angular_velocity_camera',zeros(3,1), ...
    'covariance',zeros(3),'valid',false,'timestamp',s.timestamp);
if isfield(e,'gimbal') && ~isempty(e.gimbal), m=e.gimbal; end
m.valid=m.valid && norm(m.T_body_camera(1:3,4)-g.T_body_camera(1:3,4))<1e-8;
omega=zeros(3,1);
if isfield(e,'angular_velocity_camera'), omega=e.angular_velocity_camera(:); end
values={s.timestamp,s.helicopter_state(:),s.ship_state(:), ...
    e.T_deck_camera,e.velocity(:),e.covariance,double(e.healthy), ...
    e.timestamp,s.gimbal_state.angles,m.T_body_camera(1:3,1:3), ...
    m.angular_velocity_camera(:),m.covariance,double(m.valid),m.timestamp,omega};
names={'timestamp','helicopter_state','ship_state', ...
    'nav_T_deck_camera','nav_velocity_deck', ...
    'nav_covariance','nav_healthy','nav_timestamp','gimbal_angles_rad', ...
    'nav_R_body_camera','nav_gimbal_omega_camera','nav_gimbal_covariance', ...
    'nav_gimbal_valid','nav_gimbal_timestamp','nav_angular_velocity_camera'};
ds=Simulink.SimulationData.Dataset;
for k=1:numel(values)
    v=values{k};
    if ismatrix(v) && size(v,2)>1
        data=cat(3,v,v);
    else
        data=[v(:)';v(:)'];
    end
    ds=ds.addElement(timeseries(data,[0;0.01]),names{k});
end
input=Simulink.SimulationInput('innerLoop');
input=input.setExternalInput(ds);
input=input.setVariable('T_body_camera',g.T_body_camera,'Workspace','innerLoop');
input=input.setVariable('H_ship_body',g.H_ship_body(:),'Workspace','innerLoop');
input=input.setVariable('gear_body',g.gear_body(:),'Workspace','innerLoop');
input=input.setModelParameter('StopTime','0');
result=sim(input); outputs=result.yout;
fields={'T_world_deck','T_deck_camera','velocity_deck', ...
    'helicopter_velocity_ned_mps','ship_velocity_ned_mps', ...
    'gear_relative_truth_ned','gear_relative_estimate_ned', ...
    'relative_distance_estimate_m','relative_covariance_ned','feedback_valid', ...
    'navigation_velocity_camera_ned_mps','state_timestamp','T_body_camera_actual', ...
    'nav_T_deck_body','navigation_velocity_gear_ned_mps'};
out=struct();
for k=1:numel(fields)
    signal=outputs.getElement(k).Values;
    data=squeeze(signal.Data);
    if isvector(data), data=data(:); end
    out.(fields{k})=data;
end
end
