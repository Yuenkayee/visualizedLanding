function test_deck_ei_window()
% Exercise the actual System block at 50 Hz, including both window edges.
name='deck_ei_window_fixture'; new_system(name); cleanup=onCleanup(@()close_system(name,0)); %#ok<NASGU>
add_block('simulink/Sources/From Workspace',[name '/input'],'VariableName','input','Interpolate','off','OutputAfterFinalValue','Holding final value');
add_block('simulink/User-Defined Functions/MATLAB System',[name '/gate'],'System','DeckEIWindow','SimulateUsing','Interpreted execution');
add_block('simulink/Sinks/To Workspace',[name '/output'],'VariableName','peak','SaveFormat','Timeseries');
add_line(name,'input/1','gate/1'); add_line(name,'gate/1','output/1');
t=(0:300)'*.02; u=ones(size(t)); u(141)=1.01;
s=Simulink.SimulationInput(name); s=s.setVariable('input',[t,u]);
s=s.setModelParameter('Solver','FixedStepDiscrete','FixedStep','0.02','StopTime','6');
r=sim(s); y=r.peak.Data(:);
assert(numel(y)==numel(t) && all(y(1:125)>1) && all(y(126:140)==1));
assert(all(y(141:266)>1) && all(y(267:end)==1));
end
