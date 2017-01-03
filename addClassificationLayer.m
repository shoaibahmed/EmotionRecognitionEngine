function [ net ] = addClassificationLayer( net, numClasses, varargin )
%ADDREGRESSIONLAYER Takes a network as input and reinitializes the classification layer along with addition of softmax loss at the end

opts.init_bias = [0, 0, 0];
opts.classificationLayers = 3;
opts.outputSize = {[14, 14, 512, numClasses * 4], [1, 1, numClasses * 4, numClasses * 2], [1, 1, 32, numClasses]};
opts.learningRate = 1e-5;
opts.weightsScale = [1e4, 1e4, 1e4];
opts.biasesScale = [0, 0, 0];
opts.weightDecay = [1e-12 , 1e-12];

[opts, varargin] = vl_argparse(opts,varargin) ;

for i = 1 : opts.classificationLayers
    % Randomly initialize the weights and biases for regression layer
    %initialW = opts.scal(i) * randn(opts.outputSize{i}(1), opts.outputSize{i}(2), opts.outputSize{i}(3), opts.outputSize{i}(4), 'single');
    initialW = init_weights(opts.outputSize{i}(1), opts.outputSize{i}(2), opts.outputSize{i}(3), opts.outputSize{i}(4), 'gaussian', opts.weightsScale(i));
    %initialBias = opts.init_bias(i) .* ones(1, opts.outputSize{i}(4), 'single');
    initialBias = rand(opts.outputSize{i}(4), 1, 'single') .* opts.biasesScale(i);
    
    net.layers{end+1} = struct('type', 'conv', ...
        'weights', {{initialW, initialBias}}, ...
        'stride', [1, 1], ...
        'pad', [0, 0, 0, 0], ...
        'learningRate', [opts.learningRate, opts.learningRate]) ;
    
    % Add non-linearity if not the last layer
    if i ~= opts.classificationLayers
        net.layers{end+1} = struct('type', 'relu') ;
        
        % Add regularization
        net.layers{end+1} = struct('type', 'dropout', 'rate', 0.5) ;
    end
    
end

% Add softmax loss on top of the network
net.layers{end+1} = struct('type', 'softmaxloss') ;

end

% -------------------------------------------------------------------------
function weights = init_weights(h, w, in, out, weightInitMethod, scale)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

%weightInitMethod = 'xavierimproved';
type = 'single';
switch lower(weightInitMethod)
  case 'gaussian'
    sc = 0.01/scale ;
    weights = randn(h, w, in, out, type)*sc;
  case 'xavier'
    sc = sqrt(3/(h*w*in)) ;
    weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
  case 'xavierimproved'
    sc = sqrt(2/(h*w*out)) ;
    weights = randn(h, w, in, out, type)*sc ;
  otherwise
    error('Unknown weight initialization method''%s''', weightInitMethod) ;
end
end