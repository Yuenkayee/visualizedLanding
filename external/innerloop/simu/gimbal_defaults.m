function c=gimbal_defaults(overrides)
% Same defaults/units as configs/gimbal.yaml; Python sends resolved values.
c=struct('enabled',true,'natural_frequency_hz',10,'damping_ratio',.9, ...
    'max_rate_rad_s',[3;3;3],'max_acceleration_rad_s2',[40;40;40], ...
    'angle_limits_rad',[1.3962634016;1.3962634016;2.0943951024], ...
    'integration_step_s',.002,'rate_feedforward',true,'encoder_std_rad',.0005, ...
    'R_deck_camera_reference',[],'initial_angles_rad',[]);
if nargin>0
    names=fieldnames(overrides);
    for k=1:numel(names), c.(names{k})=overrides.(names{k}); end
end
for name={'natural_frequency_hz','damping_ratio','integration_step_s'}
    validateattributes(c.(name{1}),{'double'},{'scalar','finite','positive'});
end
for name={'max_rate_rad_s','max_acceleration_rad_s2','angle_limits_rad'}
    c.(name{1})=c.(name{1})(:);
    validateattributes(c.(name{1}),{'double'},{'numel',3,'finite','positive'});
end
assert(max(c.angle_limits_rad)<pi && c.angle_limits_rad(2)<pi/2,'landing:Gimbal','Travel crosses Euler singularity');
validateattributes(c.encoder_std_rad,{'double'},{'scalar','finite','nonnegative'});
end
