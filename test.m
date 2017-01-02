% Setup path for MatConvNet and VLFeat
setup;

% Setup options
%opts.model = 'models/imagenet-vgg-verydeep-19.mat';
opts.model = 'F:/ConvLayers/src/models/imagenet-resnet-152-dag.mat';
opts.imageDim = 224;
opts.layerName = 'prob';

% Load model
net = dagnn.DagNN.loadobj(load(opts.model));
net.mode = 'test';

% Load webcam
cam = webcam(1);

% Start camera loop
finished = false;
while ~finished
    % Acquire a single image.
    im = snapshot(cam);
	
    % Input image to recognization engine
    im = single(im);
    im = imresize(im, [opts.imageDim, opts.imageDim], 'bilinear');

    averageImage = mean(mean(im));
    im = bsxfun(@minus, im, averageImage);
    
    % Feed the image through the network
    net.eval({'data', im});

    % Extract probabilities from the network
    activation = net.vars(net.getVarIndex(opts.layerName)).value;
    activation = squeeze(gather(activation));
    
    [conf, ind] = max(activation);
    net.meta.classes.description{ind};
    
    % Overlay results on image
    imshow(im);
    
    text_str = cell(3,1);
    conf_val = [85.212 98.76 78.342];
    for ii=1:3
       text_str{ii} = ['Confidence: ' num2str(conf_val(ii),'%0.2f') '%'];
    end
end

clear('cam');