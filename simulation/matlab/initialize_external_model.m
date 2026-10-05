function result = initialize_external_model(configJSON)
% Default: external ship-motion System object + point-mass aircraft surrogate.
% Optional callback returns/advances state to integrate a verified controller.
if nargin == 0, configJSON = '{}'; end
c = jsondecode(configJSON);
s = struct('timestamp', 0, 'position', [2;-1;30], 'velocity', zeros(3,1), ...
    'estimate', [], 'config', c, 'ship', []);
if isfield(c,'initial_position'), s.position = c.initial_position(:); end
if isfield(c,'initialize_callback')
    s = feval(c.initialize_callback, s);
elseif exist('ship_motion_model', 'class') == 8
    params = struct();
    if isfield(c,'ship'), params = c.ship; end
    s.ship = ship_motion_model(params);
else
    error('landing:MissingShip', 'external ship_motion_model class is required');
end
assignin('base','landingState',s);
result = jsonencode(struct('initialized',true,'timestamp',s.timestamp));
end
