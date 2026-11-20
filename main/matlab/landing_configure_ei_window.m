function info=landing_configure_ei_window(phase, wind, hs, dt, limits)
% Configure only the disposable simulation copy, using the actual forcing.
if nargin==5
    limits=reshape(limits,1,[]);
    assert(numel(limits)==2 && all(isfinite(limits)) && limits(1)>=0 && diff(limits)>0 && all(abs(limits/dt-round(limits/dt))<1e-8),'landing:Window','Invalid configured prediction interval');
elseif wind==12 && hs==2
    limits=[4.5 5.5];
elseif wind==15 && hs==3
    limits=[4 5.5];
else
    limits=[3.5 6]; % original custom 10 m/s, Hs=2 m preset
end
assert(abs(dt-.02)<1e-12,'landing:Window','DeckEIWindow requires 50 Hz');
realRamp=[phase '/Ship_motion/Ramp']; predRamp=[phase '/PredictModel/Ramp'];
gate=[phase '/PredictModel/Deck EI window maximum'];
assert(slResolve(get_param(realRamp,'start'),realRamp)==5 && slResolve(get_param(realRamp,'InitialOutput'),realRamp)==0 ...
    && slResolve(get_param(realRamp,'slope'),realRamp)==1 && slResolve(get_param(predRamp,'start'),predRamp)==0 ...
    && slResolve(get_param(predRamp,'slope'),predRamp)==1,'landing:Window','Unexpected wave clocks');
offset=limits(2)-5; samples=1+round(diff(limits)/dt);
set_param(predRamp,'InitialOutput',num2str(offset,17)); set_param(gate,'WindowSamples',num2str(samples));
assert(slResolve(get_param(predRamp,'InitialOutput'),predRamp)==offset && slResolve(get_param(gate,'WindowSamples'),gate)==samples, ...
    'landing:Window','Prediction window readback mismatch');
info=struct('lead_interval_s',limits,'samples_inclusive',samples,'sample_period_s',dt,'predictor_clock_offset_s',offset,'max_allowed_ei',1);
end
