% Add folders to path.

root_dir = fileparts(mfilename('fullpath'));
addpath(root_dir);
addpath(genpath(fullfile(root_dir, 'tool')));
addpath(genpath(fullfile(root_dir, 'auxiliary')));
addpath(genpath(fullfile(root_dir, 'benchmark')));
