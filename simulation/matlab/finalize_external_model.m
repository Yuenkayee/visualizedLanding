function result = finalize_external_model()
if evalin('base',"exist('landingState','var')")
    s=evalin('base','landingState');
    if isfield(s.config,'finalize_callback'), feval(s.config.finalize_callback,s); end
    if ~isempty(s.ship), release(s.ship); end
    evalin('base','clear landingState');
end
result=jsonencode(struct('finalized',true));
end
