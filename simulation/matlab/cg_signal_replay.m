function result=cg_signal_replay(requestJSON)
% Lightweight batch entry point for cross-language and recorded-signal checks.
c=jsondecode(requestJSON); s=cg_signal_initial_state(); n=numel(c.time);
values=zeros(n,8);
for k=1:n
    [s,p,v,ok,age]=cg_signal_step(s,c.time(k),c.position(k,:)',c.velocity(k,:)', ...
        c.healthy(k),c.sample_time(k),c.parameters(:));
    values(k,:)=[p' v' ok age];
end
result=jsonencode(values);
end
