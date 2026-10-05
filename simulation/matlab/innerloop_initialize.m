function s=innerloop_initialize(s)
% Configure explicitly: no guessed UH-60 dimensions or fabricated dynamics.
c=s.config;
assert(isfield(c,'geometry'),'landing:Geometry','geometry is required');
for field={'T_body_camera','H_ship_body','gear_body'}
    assert(isfield(c.geometry,field{1}),'landing:Geometry','Missing calibrated geometry');
end
assert(isequal(size(c.geometry.T_body_camera),[4 4]) && ...
    numel(c.geometry.H_ship_body)==3 && numel(c.geometry.gear_body)==3, ...
    'landing:Geometry','Expected 4x4 camera transform and two 3D offsets');
assert(isfield(c,'helicopter_state') && numel(c.helicopter_state)==12, ...
    'landing:State','12 helicopter state values required');
assert(isfield(c,'ship_state') && numel(c.ship_state)==12, ...
    'landing:State','12 ship state values required');
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
s.config.interface_root=fullfile(root,'external','innerloop','simu');
addpath(s.config.interface_root);
s.config.truth_callback='innerloop_truth';
s.config.feedback_callback='innerloop_feedback';
s.config.advance_callback='innerloop_advance';
s.config.finalize_callback='innerloop_finalize';
s.helicopter_state=c.helicopter_state(:); s.ship_state=c.ship_state(:);
s.phase='read'; s.feedback=[];
innerloop_evaluate(s); % Validate/compile actual SLX and contracts now.
end
