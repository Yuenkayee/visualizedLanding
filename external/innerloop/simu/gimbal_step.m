function g=gimbal_step(g,t,Rbody,Rdeck,neutral,c)
% Co-located XYZ servo, identical substepping/limits to Python Gimbal.advance.
if ~isempty(g) && t==g.timestamp, return; end
if isempty(c.R_deck_camera_reference)
    reference=diag([1 -1 -1])*neutral(1:3,1:3);
else
    reference=c.R_deck_camera_reference;
end
assert(norm(reference'*reference-eye(3),'fro')<1e-6 && det(reference)>0);
R=Rbody'*Rdeck*reference*neutral(1:3,1:3)';
target=[atan2(R(3,2),R(3,3));asin(max(-1,min(1,-R(3,1))));atan2(R(2,1),R(1,1))];
lim=c.angle_limits_rad(:); command=max(-lim,min(lim,target));
saturated=any(abs(target)>lim);
if isempty(g)
    assert(isfinite(t) && t>=0);
    q=command; if ~isempty(c.initial_angles_rad), q=c.initial_angles_rad(:); end
    assert(numel(q)==3 && all(isfinite(q)) && all(abs(q)<=lim));
    g=struct('timestamp',t,'angles',q,'rates',zeros(3,1),'command',command,'saturated',false);
else
    dt=t-g.timestamp; assert(isfinite(dt) && dt>=-1e-10,'landing:Time','Gimbal time reversed');
    if dt>0 && c.enabled
        wn=2*pi*c.natural_frequency_hz;
        n=ceil(dt/min(c.integration_step_s,.1/wn)); h=dt/n;
        speed=(command-g.command)/dt;
        ff=zeros(3,1); if c.rate_feedforward, ff=speed; end
        for k=0:n-1
            desired=g.command+speed*(k+.5)*h;
            a=wn^2*(desired-g.angles)+2*c.damping_ratio*wn*(ff-g.rates);
            ac=max(-c.max_acceleration_rad_s2,min(c.max_acceleration_rad_s2,a));
            rawv=g.rates+ac*h;
            g.rates=max(-c.max_rate_rad_s,min(c.max_rate_rad_s,rawv));
            rawq=g.angles+g.rates*h;
            g.angles=max(-lim,min(lim,rawq));
            stop=(abs(rawq)>=lim) & (rawq.*g.rates>0); g.rates(stop)=0;
            saturated=saturated || any(a~=ac) || any(rawv~=g.rates) || any(stop);
        end
    end
end
if ~c.enabled, g.angles=zeros(3,1); g.rates=zeros(3,1); saturated=false; end
g.timestamp=t; g.command=command; g.saturated=saturated;
end
