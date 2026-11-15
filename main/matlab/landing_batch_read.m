function data=landing_batch_read(~,info,result)
names=fieldnames(info.taps); data=struct();
for k=1:numel(names)
    name=names{k};
    if nargin==3
        ts=result.get(info.variables.(name));
        if isfield(info,'evaluation')
            assert(~isempty(ts.Time) && abs(double(ts.Time(end))-double(result.tout(end)))<1e-8, ...
                'landing:Batch','Terminal observer %s is not aligned with the terminal state',name);
        end
        value=double(reshape(ts.getdatasamples(numel(ts.Time)),[],1));
    else
        r=get_param(info.taps.(name),'RuntimeObject');
        assert(~isempty(r),'landing:Batch','Missing runtime tap %s',name);
        value=double(r.OutputPort(1).Data); value=value(:);
    end
    assert(all(isfinite(value)),'landing:Batch','Nonfinite %s',name);
    data.(name)=value;
end
assert(ismember(data.stage,[0 1 2]),'landing:Batch','Unexpected guidance stage');
if isfield(info,'evaluation')
    for name={'ship_eta','ship_nu','ship_eta_dot','ship_nu_dot'}
        assert(numel(data.(name{1}))==6,'landing:Batch','Expected SI six-vector %s',name{1});
    end
    assert(isscalar(data.touchdown_stop) && ismember(data.touchdown_stop,[0 1]), ...
        'landing:Batch','Invalid original STOP input');
end
end
