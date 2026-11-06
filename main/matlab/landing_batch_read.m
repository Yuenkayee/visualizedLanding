function data=landing_batch_read(~,info,result)
names=fieldnames(info.taps); data=struct();
for k=1:numel(names)
    name=names{k};
    if nargin==3
        ts=result.get(info.variables.(name));
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
end
