function [heli,ship]=landing_normalize_native(heli,ship,c)
% Existing Ship_motion exports body velocity, whereas the bridge expects NED.
heli=double(heli(:)); ship=double(ship(:));
assert(numel(heli)==12 && numel(ship)==12 && all(isfinite([heli;ship])), ...
    'landing:State','Native heli/ship outputs must each contain 12 finite doubles');
if isfield(c,'ship_velocity_frame') && strcmp(c.ship_velocity_frame,'body')
    r=ship(7);p=ship(8);y=ship(9);
    R=[cos(y) -sin(y) 0;sin(y) cos(y) 0;0 0 1]* ...
        [cos(p) 0 sin(p);0 1 0;-sin(p) 0 cos(p)]* ...
        [1 0 0;0 cos(r) -sin(r);0 sin(r) cos(r)];
    ship(4:6)=R*ship(4:6); % knots in and out, no translation of CG
end
end
