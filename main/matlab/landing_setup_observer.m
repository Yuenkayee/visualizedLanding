% Existing controller/ship parameters; no new dynamics or controller gains.
landingRepo=fileparts(fileparts(fileparts(mfilename('fullpath'))));
load(fullfile(landingRepo,'external','inner_loop','datas','landingParam.mat'));
clear landingRepo;
