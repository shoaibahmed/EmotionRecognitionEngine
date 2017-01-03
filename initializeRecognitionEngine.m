% Setup path for MatConvNet and VLFeat
setup;

% Setup options
global DAG;
DAG = false; % Defines whether to load DAG or normal network

global opts;
if DAG
    opts.model = 'F:/ConvLayers/src/models/imagenet-resnet-152-dag.mat';
    opts.layerName = 'prob';
else
    opts.model = 'F:/ConvLayers/src/models/imagenet-vgg-verydeep-19.mat';
end

opts.imageDim = 224;

% Load model
global net;
if DAG
    net = dagnn.DagNN.loadobj(load(opts.model));
    net.mode = 'test';
else
    net = load(opts.model);
	net = vl_simplenn_tidy(net); % makes the model format compatible
end

% Define global variables
global cam;
global labelList;
global choosenLabel;
choosenLabel = 1;
global im;
global generatePredictions;
generatePredictions = false;
global training;
training = false;

% Load labels
fileID = fopen('labels.txt');
fileText = textscan(fileID, '%s');
labelList = fileText{1};
fclose(fileID);
