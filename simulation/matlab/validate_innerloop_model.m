function contract=validate_innerloop_model(folder)
% Load the exact interface SLX and check its named port/parameter contract.
file=fullfile(folder,'innerLoop.slx');
assert(isfile(file),'landing:Model','innerLoop.slx is missing');
if ~bdIsLoaded('innerLoop'), load_system(file); end
assert(strcmp(get_param('innerLoop','FileName'),file), ...
    'landing:Model','A different innerLoop model is already loaded');
inputNames={'timestamp','helicopter_state','ship_state', ...
    'nav_T_deck_camera','nav_velocity_deck','nav_covariance','nav_healthy','nav_timestamp'};
inputDims={'1','12','12','[4 4]','3','[6 6]','1','1'};
outputNames={'T_world_deck','T_deck_camera','velocity_deck', ...
    'helicopter_velocity_ned_mps','ship_velocity_ned_mps', ...
    'gear_relative_truth_ned','gear_relative_estimate_ned', ...
    'relative_distance_estimate_m','relative_covariance_ned','feedback_valid', ...
    'navigation_velocity_camera_ned_mps','state_timestamp'};
check_ports('Inport',inputNames,inputDims);
check_ports('Outport',outputNames,{});
block='innerLoop/Frame_Unit_ReferencePoint_Conversion';
assert(strcmp(get_param(block,'System'),'convert'),'landing:Model','Expected convert MATLAB System');
for key={'T_body_camera','H_ship_body','gear_body'}
    assert(strcmp(strtrim(get_param(block,key{1})),key{1}), ...
        'landing:Model','convert parameter %s must reference its Model Workspace variable',key{1});
end
contract=struct('model_file',file,'input_names',{inputNames}, ...
    'output_names',{outputNames},'geometry_parameters',{{'T_body_camera','H_ship_body','gear_body'}}, ...
    'navigation_reference','fixed_reference_camera','snapshot_stop_time',0);
end
function check_ports(type,names,dims)
blocks=find_system('innerLoop','SearchDepth',1,'BlockType',type);
assert(numel(blocks)==numel(names),'landing:Model','Unexpected %s count',type);
ports=cellfun(@(b)str2double(get_param(b,'Port')),blocks);
assert(isequal(sort(ports(:))',(1:numel(names))),'landing:Model','Unexpected port numbering');
for k=1:numel(names)
    block=blocks{find(ports==k,1)};
    assert(strcmp(get_param(block,'Name'),names{k}),'landing:Model','Port %d must be %s',k,names{k});
    if ~isempty(dims)
        assert(strcmp(strtrim(get_param(block,'PortDimensions')),dims{k}), ...
            'landing:Model','Input dimensions differ: %s',names{k});
    end
end
end
