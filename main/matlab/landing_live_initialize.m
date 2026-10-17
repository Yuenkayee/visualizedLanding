function s=landing_live_initialize(s)
% Keep one live Simulink execution, including integrator/System-object states.
c=s.config; source=char(c.model);
rng(c.seed,'twister');
assert(isfile(source),'landing:Model','Model file does not exist: %s',source);
assert(~isMATLABReleaseOlderThan('R2024a'),'landing:Version','Requires MATLAB R2024a or newer');
if isfield(c,'setup_script') && ~isempty(c.setup_script)
    % Existing model parameter scripts normally populate the base workspace.
    % Keep their load/clear operations out of this adapter's local variables.
    script=strrep(char(c.setup_script),'''','''''');
    evalin('base',['run(''' script ''')']);
end
[folder,original]=fileparts(source); addpath(folder);
load_system(source);
assert(strcmp(get_param(original,'Dirty'),'off'),'landing:Unsaved','Save the source model before running');
inputs=find_system(original,'SearchDepth',1,'BlockType','Inport');
actual=cellfun(@(p)get_param(p,'Name'),inputs,'UniformOutput',false);
assert(isequal(sort(actual(:)),sort(fieldnames(c.inputs))), 'landing:Ports', ...
    'Root inputs must exactly match config.inputs. The current innerLoop is a conversion interface, not a complete plant. See main/README.md.');
for key={'helicopter_state','ship_state'}
    p=[original '/' char(c.outputs.(key{1}))];
    assert(getSimulinkBlockHandle(p)>0 && strcmp(get_param(p,'BlockType'),'Outport'), ...
        'landing:Ports','Missing native-state root Outport %s',p);
end
% Work on a copy, preserving the user's source blocks, wiring and callbacks.
[~,unique]=fileparts(tempname(c.workdir)); copy=['landing_live_' unique];
target=fullfile(c.workdir,[copy '.slx']);
save_system(original,target); close_system(original,0); load_system(target);
try
    set_param(copy,'SimulationMode','normal','StartTime','0','StopTime','inf', ...
        'SolverType','Fixed-step','FixedStep',num2str(c.plant_step,17), ...
        'BlockReduction','off','SignalLogging','off','SaveOutput','off','SaveTime','off');
    % Keep the original solver if fixed-step; otherwise select the standard
    % continuous fixed-step integrator. Never replace any model subsystem.
    solver=get_param(copy,'Solver');
    if ~ismember(solver,{'ode1','ode2','ode3','ode4','ode5','ode8','ode14x','FixedStepDiscrete'})
        set_param(copy,'Solver','ode4');
    end
    ws=get_param(copy,'ModelWorkspace');
    for key={'H_ship_body','gear_body','T_body_camera'}
        v=c.geometry.(key{1}); if numel(v)==3, v=v(:); end
        assignin(ws,key{1},v);
    end
    fields=fieldnames(c.inputs);
    for k=1:numel(fields)
        p=[copy '/' fields{k}]; ph=get_param(p,'PortHandles');
        line=get_param(ph.Outport,'Line'); destinations=[];
        if line~=-1
            destinations=get_param(line,'DstPortHandle'); delete_line(line);
        end
        position=get_param(p,'Position'); delete_block(p);
        value=initial_value(c.inputs.(fields{k}));
        add_block('simulink/Sources/Constant',p,'Value',mat2str(value,17), ...
            'SampleTime','0','Position',position);
        ph=get_param(p,'PortHandles');
        for dest=destinations(:)', add_line(copy,ph.Outport,dest,'autorouting','on'); end
    end
    % Root outputs can originate from virtual Mux/subsystem blocks with no
    % RuntimeObject. Add nonvirtual observers on branches in the working copy.
    names={'helicopter_state','ship_state'};
    for k=1:2
        p=[copy '/' char(c.outputs.(names{k}))]; ports=get_param(p,'PortHandles');
        line=get_param(ports.Inport,'Line');
        assert(line~=-1,'landing:Ports','Native-state output is unconnected');
        src=get_param(line,'SrcPortHandle');
        tap=[copy '/LandingStateTap' num2str(k)]; sink=[copy '/LandingStateSink' num2str(k)];
        add_block('simulink/Math Operations/Gain',tap,'Gain','1');
        add_block('simulink/Sinks/Terminator',sink);
        tp=get_param(tap,'PortHandles'); sp=get_param(sink,'PortHandles');
        add_line(copy,src,tp.Inport,'autorouting','on');
        add_line(copy,tp.Outport,sp.Inport,'autorouting','on');
    end
    % initialize() compiles but does not evaluate outputs/initial conditions.
    % Obtain the t=0 native state in a ZERO-duration run (no integration), then
    % initialize the persistent execution with the same model and random state.
    % This is performed once, never between navigation updates. A stochastic
    % aircraft model must use its own seeded streams, as in a normal repeat run.
    randomState=rng;
    initialInput=Simulink.SimulationInput(copy);
    initialInput=initialInput.setModelParameter('StopTime','0','SaveOutput','on', ...
        'OutputSaveName','yout','SaveFormat','Dataset','ReturnWorkspaceOutputs','on');
    initialResult=sim(initialInput);
    native=cell(1,2);
    for k=1:2
        index=str2double(get_param([copy '/' char(c.outputs.(names{k}))],'Port'));
        data=initialResult.yout.getElement(index).Values.Data;
        native{k}=double(reshape(data,[],1));
        assert(numel(native{k})==12 && all(isfinite(native{k})), ...
            'landing:InitialState','Invalid native output at t=0');
    end
    rng(randomState);
    live=simulation(copy); initialize(live);
    s.live=live; s.live_model=copy;
    s.config.helicopter_state=native{1}; s.config.ship_state=native{2};
    % Use the tested conversion kernel directly. The user may extend the
    % source innerLoop.slx into a complete plant, so runtime must not assume
    % that it remains the old 15-port, zero-duration conversion-only model.
    s.config.conversion_evaluation='kernel';
    s=innerloop_initialize(s);
    s.config.plant_step_callback='landing_live_step';
    s.config.finalize_callback='landing_live_finalize';
catch err
    if exist('live','var')
        try
            terminate(live);
        catch
        end
    end
    close_system(copy,0); rethrow(err);
end
end
function v=initial_value(field)
switch char(field)
    case {'gear_relative_ned_m','velocity_gear_ned_mps'}, v=zeros(3,1);
    case 'covariance_ned', v=eye(3);
    case 'T_deck_body', v=eye(4);
    case {'valid','timestamp','distance_m'}, v=0;
    otherwise, error('landing:Mapping','Unknown feedback field: %s',field);
end
end
