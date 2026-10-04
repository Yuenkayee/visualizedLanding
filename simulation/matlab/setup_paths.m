function setup_paths(root)
% Read-only inclusion of existing external model/algorithm directories.
if nargin == 0
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
addpath(fullfile(root, 'simulation', 'matlab'));
externalRoot = fullfile(root, 'external', 'inner_loop');
if isfolder(externalRoot)
    addpath(genpath(externalRoot));
end
end
