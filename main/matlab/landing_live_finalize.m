function landing_live_finalize(s)
if isfield(s,'live') && ~strcmp(s.live.Status,'inactive')
    try
        stop(s.live);
    catch
        terminate(s.live);
    end
end
if isfield(s,'live_model') && bdIsLoaded(s.live_model), close_system(s.live_model,0); end
innerloop_finalize(s);
end
