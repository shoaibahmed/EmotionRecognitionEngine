function [ net ] = addRegressionLayer( net, varargin )
%ADDREGRESSIONLAYER Takes a network as input and adds a regression layer at the end

opts.init_bias = [0, 0, 0];
opts.regressionLayers = 3;
opts.outputSize = {[14, 14, 512, 4], [1, 1, 64, 32], [1, 1, 32, 4]};
opts.fineTuneWholeNetwork = false;
opts.learningRate = 1e-5;
opts.weightsScale = [1e4, 1e2, 1e2];
opts.biasesScale = [0, 0, 0];
opts.learningRateFT = [1e-10, 0];
opts.weightDecay = [1e-3 , 1e-3];
%opts.weightDecay = [0, 0];
opts.outputSize = {[14, 14, 512, 64], [1, 1, 64, 64], [1, 1, 64, 4]};

[opts, varargin] = vl_argparse(opts,varargin) ;

if opts.fineTuneWholeNetwork
    % Set Hyper-parameters for Network
    for i=1:numel(net.layers)
        if ~strcmp(net.layers{i}.type,'conv'), continue; end
        if ~isfield(net.layers{i}, 'learningRate')
            net.layers{i}.learningRate = [opts.learningRateFT(1), opts.learningRateFT(2)];
        end
        if ~isfield(net.layers{i}, 'weightDecay')
            net.layers{i}.weightDecay = opts.weightDecay;
        end

        %   net.layers{i}.filtersMomentum = zeros(size(net.layers{i}.weights{1}), ...
        %     class(net.layers{i}.weights{1})) ;
        %   net.layers{i}.biasesMomentum = zeros(size(net.layers{i}.weights{2}), ...
        %     class(net.layers{i}.weights{2})) ; 
        %   if ~isfield(net.layers{i}, 'filtersLearningRate')
        %     net.layers{i}.filtersLearningRate = 1 ;
        %   end
        %   if ~isfield(net.layers{i}, 'biasesLearningRate')
        %     net.layers{i}.biasesLearningRate = 1 ;
        %   end
        %   if ~isfield(net.layers{i}, 'filtersWeightDecay')
        %     net.layers{i}.filtersWeightDecay = 1 ;
        %   end
        %   if ~isfield(net.layers{i}, 'biasesWeightDecay')
        %     net.layers{i}.biasesWeightDecay = 1 ;
        %   end
    end
end

for i = 1 : opts.regressionLayers
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
    if i ~= opts.regressionLayers
        net.layers{end+1} = struct('type', 'relu') ;
        
        % Add regularization
        net.layers{end+1} = struct('type', 'dropout', 'rate', 0.5) ;
    end
    
end

% Add tanh layer
%net.layers{end+1} = struct('type', 'tanh') ;

% Add L2 loss on top of the network
net.layers{end+1} = struct('type', 'nnL2loss') ;

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