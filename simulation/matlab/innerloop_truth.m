function truth=innerloop_truth(s)
assert(strcmp(s.phase,'read') || strcmp(s.phase,'feedback'), ...
    'landing:Order','Invalid state-read phase');
o=innerloop_evaluate(s);
truth=struct('timestamp',s.timestamp,'T_world_deck',o.T_world_deck, ...
    'T_deck_camera',o.T_deck_camera,'velocity_deck',o.velocity_deck');
end
