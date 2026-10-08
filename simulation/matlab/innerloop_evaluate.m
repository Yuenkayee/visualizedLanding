function out=innerloop_evaluate(s)
% Pure snapshot evaluation. Running StopTime=0 never advances a plant.
% Generated Simulink artifacts stay outside the source tree.
previous=Simulink.fileGenControl('getConfig');
cache=fullfile(tempdir,'visualizedLanding_innerloop_cache');
Simulink.fileGenControl('set','CacheFolder',cache, ...
    'CodeGenFolder',fullfile(cache,'codegen'),'createDir',true);
restore=onCleanup(@()Simulink.fileGenControl('setConfig','config',previous)); %#ok<NASGU>
if ~bdIsLoaded('innerLoop')
    validate_innerloop_model(s.config.interface_root);
end
assert(strcmp(get_param('innerLoop','FileName'),fullfile(s.config.interface_root,'innerLoop.slx')), ...
    'landing:Model','Loaded model differs from the configured interface');
g=s.config.geometry;
if isempty(s.estimate)
    e=struct('T_deck_camera',eye(4),'velocity',zeros(3,1), ...
        'covariance',eye(6),'healthy',false,'timestamp',s.timestamp);
else
    e=s.estimate;
end
values={s.timestamp,s.helicopter_state(:),s.ship_state(:), ...
    e.T_deck_camera,e.velocity(:),e.covariance,double(e.healthy), ...
    e.timestamp};
names={'timestamp','helicopter_state','ship_state', ...
    'nav_T_deck_camera','nav_velocity_deck', ...
    'nav_covariance','nav_healthy','nav_timestamp'};
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
    'navigation_velocity_camera_ned_mps','state_timestamp'};
out=struct();
assert(numElements(outputs)==numel(fields),'landing:Model','Unexpected SLX output dataset size');
for k=1:numel(fields)
    % Simulink may leave Dataset signal labels empty. Root Outport numbering
    % and names are checked at initialization, so that contract maps this index.
    signal=outputs.getElement(k).Values;
    data=squeeze(signal.Data);
    if isvector(data), data=data(:); end
    out.(fields{k})=data;
end
end
