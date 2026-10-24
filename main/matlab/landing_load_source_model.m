function model=landing_load_source_model(source)
% A shared Engine may still hold the version loaded before git pull.
% Refresh clean, stopped models only; never discard interactive edits.
assert(isfile(source),'landing:Model','Model file does not exist: %s',source);
[ok,info]=fileattrib(source); assert(ok,'landing:Model','Cannot resolve model: %s',source);
source=info.Name; [folder,model]=fileparts(source); addpath(folder);
if bdIsLoaded(model)
    assert(strcmp(get_param(model,'SimulationStatus'),'stopped'), ...
        'landing:Busy','Stop the existing simulation before reloading %s',model);
    assert(strcmp(get_param(model,'Dirty'),'off'),'landing:Unsaved', ...
        'Model %s has unsaved edits. Save them to a separate file or close it explicitly before reloading %s',model,source);
    close_system(model,0);
end
load_system(source);
fprintf('[landing] Loaded source model from disk: %s\n',get_param(model,'FileName'));
end
