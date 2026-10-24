function result=test_navigation_connection(requestJSON)
c=jsondecode(requestJSON); [~,model]=fileparts(c.model); load_system(c.model);
cleanup=onCleanup(@()close_system(model,0));
report=landing_check_feedback(model,c); assert(report.enabled);
% The former observer contract must refuse to mislabel this as a bypass run.
observer=c; observer.evaluation_mode='observer';
must_fail(@()landing_check_observer(model,observer),'landing:Observer');
cmd=[model '/' c.autonomous_subsystem '/LAND_CMD'];
set_param([cmd '/Plus'],'Inputs','++');
must_fail(@()landing_check_feedback(model,c),'landing:Feedback');
set_param([cmd '/Plus'],'Inputs','-+');
set_param([cmd '/ZeroRelativeVelocity'],'Value','[1;0;0]');
must_fail(@()landing_check_feedback(model,c),'landing:Feedback');
set_param([cmd '/ZeroRelativeVelocity'],'Value','zeros(3,1)');
conversion='Frame_Unit_ReferencePoint_Conversion';
delete_line(model,[conversion '/20'],[c.autonomous_subsystem '/1']);
add_line(model,[conversion '/16'],[c.autonomous_subsystem '/1']);
must_fail(@()landing_check_feedback(model,c),'landing:Feedback');
delete_line(model,[conversion '/16'],[c.autonomous_subsystem '/1']);
add_line(model,[conversion '/20'],[c.autonomous_subsystem '/1']);
delete_line(model,[conversion '/22'],[c.autonomous_subsystem '/2']);
add_line(model,[conversion '/10'],[c.autonomous_subsystem '/2']);
must_fail(@()landing_check_feedback(model,c),'landing:Feedback');
result=jsonencode(struct('passed',true,'wrong_sign_rejected',true,'truth_feedback_rejected',true, ...
    'wrong_validity_rejected',true,'invalid_velocity_fallback_checked',true,'observer_rejected',true));
end
function must_fail(callback,id)
try, callback(); catch err, assert(strcmp(err.identifier,id)); return; end
error('landing:Test','Expected %s',id);
end
