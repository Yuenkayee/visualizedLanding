function [s,p,v,valid,age]=cg_signal_step(s,t,position,velocity,healthy,sample_t,parameters)
% Causal velocity-aided resampling with bounded exponential position correction.
% parameters = [time_constant_s; max_correction_speed_mps; max_age_s;
%               max_tracking_error_m]. No truth, future samples or covariance update.
% A held packet is propagated, never assimilated repeatedly. An explicit unhealthy
% packet invalidates output immediately. Expired output freezes and stays invalid.
assert(numel(parameters)==4 && all(isfinite(parameters)) && all(parameters>0), ...
    'landing:CgSignal','Invalid CG signal parameters');
assert(isfinite(t) && t>=0 && t>=s.time-1e-8,'landing:CgSignal','Nonmonotonic output clock');
tau=parameters(1); speed=parameters(2); max_age=parameters(3); max_error=parameters(4);
dt=0; if s.time>=0, dt=max(0,t-s.time); end
packet_ok=isfinite(sample_t) && sample_t>=0 && sample_t<=t+1e-8;
good=logical(healthy) && all(isfinite(position)) && all(isfinite(velocity));
fresh=packet_ok && (sample_t>s.sample_time+1e-8 || (~s.initialized && good && sample_t>=s.sample_time));
if ~good || ~packet_ok, s.healthy=false; end
if fresh
    s.sample_time=sample_t;
    s.healthy=good && t-sample_t<=max_age+1e-8;
end
usable=fresh && s.healthy;
if usable && ~s.initialized
    s.position=position(:)+velocity(:)*(t-sample_t);
    s.feed_velocity=velocity(:); s.velocity=velocity(:);
    s.reference_position=position(:);
    s.valid_time=sample_t; s.initialized=true; s.tracking_error=0;
elseif s.initialized
    previous=s.position;
    % Coast only up to the expiry boundary, including an explicit invalid packet.
    coast=max(0,min(t,s.valid_time+max_age)-max(0,s.time));
    predicted=s.position+s.feed_velocity*coast;
    if usable
        % Trapezoidal velocity prediction avoids a half-step acceleration lag.
        predicted=predicted+.5*(velocity(:)-s.feed_velocity)*coast;
        s.reference_position=position(:);
        s.feed_velocity=velocity(:); s.valid_time=sample_t;
    end
    if s.healthy && t-s.valid_time<=max_age+1e-8
        residual=s.reference_position+s.feed_velocity*(t-s.valid_time)-predicted;
        correction=-expm1(-dt/tau)*residual;
        correction=correction*min(1,speed*dt/max(norm(correction),eps));
        s.position=predicted+correction;
        s.tracking_error=norm(residual-correction);
    else
        s.position=predicted;
    end
    if dt>1e-8, s.velocity=(s.position-previous)/dt; end
end
s.time=t;
age=max_age+1; if s.initialized, age=max(0,t-s.valid_time); end
valid=double(s.initialized && s.healthy && age<=max_age+1e-8 && s.tracking_error<=max_error);
p=s.position; v=s.velocity;
end
