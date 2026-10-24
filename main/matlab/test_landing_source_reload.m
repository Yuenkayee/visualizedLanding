function result=test_landing_source_reload(workdir)
% Reproduce git replacing a source file while a shared session holds it open.
if ~isfolder(workdir), mkdir(workdir); end
[~,name]=fileparts(tempname(workdir)); model=['reload_' name];
source=fullfile(workdir,[model '.slx']);
old=fullfile(workdir,'one_input.slx'); fresh=fullfile(workdir,'two_inputs.slx');
new_system(model); cleanup=onCleanup(@()close_system(model,0));
add_block('simulink/Sources/In1',[model '/position']);
save_system(model,source); copyfile(source,old);
add_block('simulink/Sources/In1',[model '/valid'],'Port','2');
save_system(model); copyfile(source,fresh); close_system(model,0);
copyfile(old,source); load_system(source);
assert(numel(find_system(model,'SearchDepth',1,'BlockType','Inport'))==1);
copyfile(fresh,source);
landing_load_source_model(source);
assert(numel(find_system(model,'SearchDepth',1,'BlockType','Inport'))==2);
set_param([model '/valid'],'Name','unsaved_valid');
try
    landing_load_source_model(source);
    error('landing:Test','Expected unsaved-edit rejection');
catch err
    assert(strcmp(err.identifier,'landing:Unsaved'));
end
assert(getSimulinkBlockHandle([model '/unsaved_valid'])>0 && strcmp(get_param(model,'Dirty'),'on'));
result=jsonencode(struct('passed',true,'stale_clean_model_reloaded',true,'unsaved_edits_preserved',true));
end
