% Setup path for MatConvNet and VLFeat
setup;

% Setup options
global DAG;
DAG = false; % Defines whether to load DAG or normal network

global opts;
opts.imageDim = 224;
if DAG
    opts.model = 'F:/ConvLayers/src/models/imagenet-resnet-152-dag.mat';
    opts.layerName = 'prob';
else
    opts.model = '../../MATLAB/ImageRetrieval/ConvNet/models/imagenet-vgg-verydeep-19.mat';
end

% Training params
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0 ;
opts.randomSeed = 0 ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.gpus = [];
opts.conserveMemory = true ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.cudnn = false ;
opts.errorFunction = 'multiclass' ;

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
global newModelFileName;
newModelFileName = 'sas-custom.mat';
global state;
state.learningRate = opts.learningRate;
global updateFunctionTimer;

% Load labels
fileID = fopen('labels.txt');
fileText = textscan(fileID, '%s');
labelList = fileText{1};
fclose(fileID);
