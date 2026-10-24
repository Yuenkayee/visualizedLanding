function s=landing_live_step(s,dt)
assert(~s.stopped,'landing:Stopped','Cannot advance a terminated plant');
assert(strcmp(s.phase,'feedback'),'landing:Order','Read -> feedback -> advance required');
assert(abs(s.feedback.timestamp-s.timestamp)<1e-8,'landing:Stale','Stale feedback');
h=s.config.plant_step; n=round(dt/h);
assert(n>=1 && abs(n*h-dt)<1e-8,'landing:Time','dt must be a multiple of plant_step');
fields=fieldnames(s.config.inputs);
cgControl=isfield(s.config,'control_interface') && strcmp(s.config.control_interface,'cg_position');
if cgControl
    assert(abs(dt-h)<1e-8,'landing:Time','CG control requires a fresh feedback at every 0.02 s step');
    set_param([s.live_model '/LandingControlCG'],'Value',mat2str(s.feedback.cg_relative_estimate_50hz_ned(:),17));
    set_param([s.live_model '/LandingControlValid'],'Value',mat2str(double(s.feedback.cg_signal_valid),17));
end
values=s.feedback;
if isfield(s.config,'input_contract') && strcmp(s.config.input_contract,'conversion')
    values=landing_conversion_inputs(s);
end
for k=1:numel(fields)
    value=values.(s.config.inputs.(fields{k}));
    % set_param updates the initialized execution immediately, including t=0.
    % Simulation.setBlockParameter before the first step is only a run setting.
    set_param([s.live_model '/' fields{k}],'Value',mat2str(double(value),17));
end
start=s.timestamp; tick=round(start/h);
assert(abs(tick*h-start)<1e-8,'landing:Time','Plant time is not on the fixed-step grid');
for k=1:n
    target=(tick+k)*h;
    record=struct('tick',tick+k,'from',s.timestamp,'target',target, ...
        'actual_time',double(s.live.Time),'status_before',char(s.live.Status), ...
        'status_after','','final_step',false,'stopped',false,'stop_reason','', ...
        'error_identifier','','error','');
    try
        assert(ismember(char(s.live.Status),{'initialized','paused'}), ...
            'landing:Time','Cannot step plant with status %s at %.17g s',char(s.live.Status),double(s.live.Time));
        assert(abs(s.live.Time-s.timestamp)<1e-8,'landing:Time', ...
            'Plant/host time mismatch before step: host=%.17g, plant=%.17g',s.timestamp,double(s.live.Time));
        % The model uses a validated fixed step: request exactly one major
        % step, not an absolute floating-point pause boundary. step() is
        % synchronous and returns true after the final step even if normal
        % mode leaves the Simulation object paused rather than inactive.
        record.final_step=logical(step(s.live,NumberOfSteps=1));
        record.actual_time=double(s.live.Time);
        record.status_after=char(s.live.Status);
        if record.final_step || strcmp(s.live.Status,'inactive')
            if ~strcmp(s.live.Status,'inactive')
                stop(s.live); % Finalize retained logging before reading it.
            end
            result=s.live.SimulationOutput;
            assert(isempty(result.ErrorMessage),'landing:Simulation','%s',result.ErrorMessage);
            assert(~isempty(result.tout),'landing:Stop','Stopped without a recorded state');
            terminal=double(result.tout(end));
            assert(terminal>=s.timestamp-1e-8 && terminal<=target+1e-8, ...
                'landing:Time','Terminal time %.17g is outside requested interval [%.17g, %.17g]', ...
                terminal,s.timestamp,target);
            s.timestamp=terminal;
            s.stopped=true;
            s.stop_reason=char(string(result.SimulationMetadata.ExecutionInfo.StopEvent));
            record.stopped=true; record.stop_reason=s.stop_reason;
            record.terminal_time=terminal;
        else
            assert(strcmp(s.live.Status,'paused') && abs(s.live.Time-target)<1e-8, ...
                'landing:Time', ...
                'Plant missed fixed step: from=%.17g, target=%.17g, actual=%.17g, status=%s, finalstep=%d', ...
                record.from,target,double(s.live.Time),char(s.live.Status),record.final_step);
            s.timestamp=double(s.live.Time);
        end
        [s.helicopter_state,s.ship_state]=landing_live_read(s);
        if cgControl && ~s.stopped
            s.control_used=landing_control_used(s);
            record.control=s.control_used;
        end
        s=innerloop_update_gimbal(s); % sample actual attitude at inner-loop rate
    catch err
        record.actual_time=double(s.live.Time);
        record.status_after=char(s.live.Status);
        record.error_identifier=err.identifier; record.error=err.message;
        write_record(s,record);
        rethrow(err);
    end
    write_record(s,record);
    if s.stopped, break; end
end
end
function write_record(s,record)
file=fullfile(s.config.workdir,'plant_steps.jsonl');
fid=fopen(file,'a');
assert(fid~=-1,'landing:Log','Cannot open plant timing log: %s',file);
cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',jsonencode(record));
end
