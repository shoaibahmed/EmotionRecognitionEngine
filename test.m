% Setup path for MatConvNet and VLFeat
setup;

% Setup options
global DAG;
DAG = false; % Defines whether to load DAG or normal network

if DAG
    opts.model = 'F:/ConvLayers/src/models/imagenet-resnet-152-dag.mat';
    opts.layerName = 'prob';
else
    opts.model = 'F:/ConvLayers/src/models/imagenet-vgg-verydeep-19.mat';
end

opts.imageDim = 224;

% Load model
if DAG
    net = dagnn.DagNN.loadobj(load(opts.model));
    net.mode = 'test';
else
    net = load(opts.model);
	net = vl_simplenn_tidy(net); % makes the model format compatible
end

% Load labels
fileID = fopen('labels.txt');
fileText = textscan(fileID, '%s');
labelList = fileText{1};
fclose(fileID);

% Add new classification layer
numClasses = length(labelList);
net = addClassificationLayer(net, numClasses);
net.meta.classes.name = labelList;
net.meta.classes.description = labelList;

% Load webcam
cam = webcam(1);

% Start camera loop
finished = false;
while ~finished
    % Acquire a single image.
    img = snapshot(cam);
    
    % Show image
    imshow(img);
	
    % Input image to recognization engine
    im = single(img);
    im = imresize(im, [opts.imageDim, opts.imageDim], 'bilinear');

    %averageImage = mean(mean(im));
    im = bsxfun(@minus, im, net.meta.normalization.averageImage);
    
   % Feed the image through the network
    if DAG
        net.eval({'data', im});

        % Extract probabilities from the network
        activation = net.vars(net.getVarIndex(opts.layerName)).value;
        activation = squeeze(gather(activation));
    else
        res = vl_simplenn(net, im);
        activation = squeeze(gather(res(end).x));
    end
    
    [conf, ind] = max(activation);
    title(net.meta.classes.description{ind});
    
    %pause(10) % pauses for 100 milliseconds.
%     
%     text_str = cell(3,1);
%     conf_val = [85.212 98.76 78.342];
%     for ii=1:3
%        text_str{ii} = ['Confidence: ' num2str(conf_val(ii),'%0.2f') '%'];
%     end
end

clear('cam');