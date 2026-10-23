function out=innerloop_snapshot(s)
% Same pure kernel as convert/innerLoop.slx, for an already validated live loop.
g=s.config.geometry;
e=s.estimate;
if isempty(e)
    e=struct('T_deck_camera',eye(4),'velocity',zeros(3,1), ...
        'covariance',eye(6),'healthy',false,'timestamp',s.timestamp);
end
m=struct('T_body_camera',g.T_body_camera,'angular_velocity_camera',zeros(3,1), ...
    'covariance',zeros(3),'valid',false,'timestamp',s.timestamp);
if isfield(e,'gimbal') && ~isempty(e.gimbal), m=e.gimbal; end
m.valid=m.valid && norm(m.T_body_camera(1:3,4)-g.T_body_camera(1:3,4))<1e-8;
omega=zeros(3,1);
if isfield(e,'angular_velocity_camera'), omega=e.angular_velocity_camera(:); end
fields=innerloop_output_fields();
values=cell(1,numel(fields));
[values{:}]=innerLoop_conversion(s.timestamp,s.helicopter_state(:),s.ship_state(:), ...
    g.T_body_camera,g.H_ship_body(:),g.gear_body(:),e.T_deck_camera,e.velocity(:), ...
    e.covariance,double(e.healthy),e.timestamp,s.gimbal_state.angles, ...
    m.T_body_camera(1:3,1:3),m.angular_velocity_camera(:),m.covariance, ...
    double(m.valid),m.timestamp,omega);
out=cell2struct(values,fields,2);
end
