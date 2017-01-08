function varargout = emotionRecognitionEngineGUI(varargin)
% EMOTIONRECOGNITIONENGINEGUI MATLAB code for emotionRecognitionEngineGUI.fig
%      EMOTIONRECOGNITIONENGINEGUI, by itself, creates a new EMOTIONRECOGNITIONENGINEGUI or raises the existing
%      singleton*.
%
%      H = EMOTIONRECOGNITIONENGINEGUI returns the handle to a new EMOTIONRECOGNITIONENGINEGUI or the handle to
%      the existing singleton*.
%
%      EMOTIONRECOGNITIONENGINEGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in EMOTIONRECOGNITIONENGINEGUI.M with the given input arguments.
%
%      EMOTIONRECOGNITIONENGINEGUI('Property','Value',...) creates a new EMOTIONRECOGNITIONENGINEGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before emotionRecognitionEngineGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to emotionRecognitionEngineGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help emotionRecognitionEngineGUI

% Last Modified by GUIDE v2.5 05-Jan-2017 05:33:15

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @emotionRecognitionEngineGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @emotionRecognitionEngineGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before emotionRecognitionEngineGUI is made visible.
function emotionRecognitionEngineGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to emotionRecognitionEngineGUI (see VARARGIN)

% Choose default command line output for emotionRecognitionEngineGUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes emotionRecognitionEngineGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% Set status and output labels
set(handles.outputLabel, 'string', '');
set(handles.statusBar, 'string', 'Initializing');

% Disable all buttons
set(handles.startCam, 'Enable', 'off');
set(handles.stopCam, 'Enable', 'off');
set(handles.startPredictionsButton, 'Enable', 'off');
set(handles.stopPredictionsButton, 'Enable', 'off');
set(handles.startTrainingButton, 'Enable', 'off');
set(handles.stopTrainingButton, 'Enable', 'off');


% Initialize the recognition engine
initializeRecognitionEngine;
set(handles.statusBar, 'string', 'Initialization completed');

% Create a callback for updating the system
global updateFunctionTimer;
updateFunctionTimer = timer(...
    'ExecutionMode', 'fixedSpacing', ...       % Run timer repeatedly.
    'Period', 0.001, ...                       % Period between executions (5 mili-sec.)
    'BusyMode', 'drop', ...                    % Drop the callback if execution queue is busy
    'TimerFcn', {@updateDisplay, handles});    % Specify callback function.

global labelList;
set(handles.labelListBox, 'string', labelList);

% Enable all start buttons
set(handles.startCam, 'Enable', 'on');
set(handles.startPredictionsButton, 'Enable', 'on');
set(handles.startTrainingButton, 'Enable', 'on');


% --- Outputs from this function are returned to the command line.
function varargout = emotionRecognitionEngineGUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


function updateDisplay(hTimerObj, timerEvent, handles)
global cam;
if isempty(cam), return, end;

global im;
global net;
global opts;
global generatePredictions;
global training;
global DAG;
global state;
global choosenLabel;

im = snapshot(cam);
imshow(im, 'Parent', handles.imagePlaceholder);

% If evaluation mode
if generatePredictions
    
    img = single(im);
    img = imresize(img, [opts.imageDim, opts.imageDim], 'bilinear');
    %averageImage = mean(mean(img));
    img = bsxfun(@minus, img, net.meta.normalization.averageImage);
    %evalMode = 'test' ;
    
    % Feed the image through the network
    if DAG
        net.eval({'data', img});

        % Extract probabilities from the network
        activation = net.vars(net.getVarIndex(opts.layerName)).value;
        activation = squeeze(gather(activation));
    else
        %res = vl_simplenn(net, img, 'mode', evalMode);
        res = vl_simplenn(net, img);
        activation = squeeze(gather(res(end).x));        
    end

    [conf, ind] = max(activation);
    myText = net.meta.classes.description{ind};
    myText = strcat([myText, '(', num2str(conf), ')']);

    set(handles.outputLabel, 'string', myText);
    
% If training mode
elseif training
    
    img = single(im);
    img = imresize(img, [opts.imageDim, opts.imageDim], 'bilinear');
    %averageImage = mean(mean(img));
    img = bsxfun(@minus, img, net.meta.normalization.averageImage);
    
    dzdy = 1 ;
    evalMode = 'normal' ;
    res = [] ;
    error = [];
    mmap = [];
    batchSize = 1;
    
    net.layers{end}.class = choosenLabel ;
    res = vl_simplenn(net, img, dzdy, res, ...
                      'accumulate', false, ... % Not sure
                      'mode', evalMode, ...
                      'conserveMemory', opts.conserveMemory, ...
                      'backPropDepth', opts.backPropDepth, ...
                      'sync', opts.sync, ...
                      'cudnn', opts.cudnn) ;
                  
%     % accumulate errors
%     error = sum([error, [...
%                 sum(double(gather(res(end).x))) ;
%                 reshape(opts.errorFunction(opts, choosenLabel, res),[],1) ; ]],2) ;
            
    [state, net] = accumulate_gradients(state, net, res, opts, batchSize, mmap) ;
end


% -------------------------------------------------------------------------
function [state, net] = accumulate_gradients(state, net, res, opts, batchSize, mmap)
% -------------------------------------------------------------------------
numGpus = numel(opts.gpus) ;
otherGpus = [];%= setdiff(1:numGpus, labindex) ;

for l=numel(net.layers):-1:1
  for j=1:numel(res(l).dzdw)

    % accumualte gradients from multiple labs (GPUs) if needed
    if numGpus > 1
      tag = sprintf('l%d_%d',l,j) ;
      for g = otherGpus
        tmp = gpuArray(mmap.Data(g).(tag)) ;
        res(l).dzdw{j} = res(l).dzdw{j} + tmp ;
      end
    end

    if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
      % special case for learning bnorm moments
      thisLR = net.layers{l}.learningRate(j) ;
      net.layers{l}.weights{j} = ...
        (1 - thisLR) * net.layers{l}.weights{j} + ...
        (thisLR/batchSize) * res(l).dzdw{j} ;
    else
      % standard gradient training
      thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
      thisLR = state.learningRate * net.layers{l}.learningRate(j) ;
      state.layers{l}.momentum{j} = opts.momentum * state.layers{l}.momentum{j} ...
        - thisDecay * net.layers{l}.weights{j} ...
        - (1 / batchSize) * res(l).dzdw{j} ;
      net.layers{l}.weights{j} = net.layers{l}.weights{j} + ...
        thisLR * state.layers{l}.momentum{j} ;
    end
  end
end


% --- Executes on button press in startCam.
function startCam_Callback(hObject, eventdata, handles)
% hObject    handle to startCam (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global cam;
if ~isempty(cam), return, end;
cam = webcam(1);
set(handles.statusBar, 'string', 'Camera started');

% Start the update function timer call
global updateFunctionTimer;
start(updateFunctionTimer);

% Disable start button and enable stop button
set(handles.startCam, 'Enable', 'off');
set(handles.stopCam, 'Enable', 'on');


% --- Executes on button press in stopCam.
function stopCam_Callback(hObject, eventdata, handles)
% hObject    handle to stopCam (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Stop the update function timer call
global updateFunctionTimer;
stop(updateFunctionTimer);

global cam;
if isempty(cam), return, end;
cam = [];
set(handles.statusBar, 'string', 'Camera stopped');

% if strcmp(get(handles.timer, 'Running'), 'on')
%     stop(handles.timer);
% end

% Disable stop button and enable start button
set(handles.startCam, 'Enable', 'on');
set(handles.stopCam, 'Enable', 'off');


% --- Executes on button press in startTrainingButton.
function startTrainingButton_Callback(hObject, eventdata, handles)
% hObject    handle to startTrainingButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global net;
% Add softmax loss on top of the network
net.layers = net.layers(1 : end-1); % Remove softmax layer
net.layers{end+1} = struct('type', 'softmaxloss') ;

% Make the network format compatible
net = vl_simplenn_tidy(net);

% Setup options for training
net.layers{end-1}.precious = 1; % do not remove predictions, used for error

for i=1:numel(net.layers)
    if isfield(net.layers{i}, 'weights')
        J = numel(net.layers{i}.weights) ;
        if ~isfield(net.layers{i}, 'learningRate')
            net.layers{i}.learningRate = ones(1, J, 'single') ;
        end
        if ~isfield(net.layers{i}, 'weightDecay')
            net.layers{i}.weightDecay = ones(1, J, 'single') ;
        end
    end
end

global state;
state.momentum = {} ;
for i = 1:numel(net.layers)
    if isfield(net.layers{i}, 'weights')
        for j = 1:numel(net.layers{i}.weights)
            state.layers{i}.momentum{j} = 0 ;
        end
    end
end

% % Add error function
% global opts;
% opts.errorFunction = @error_multiclass ;

set(handles.statusBar, 'string', 'Training started');

% Disable start button and enable stop button
set(handles.startTrainingButton, 'Enable', 'off');
set(handles.stopTrainingButton, 'Enable', 'on');

% Disable predictions button
set(handles.startPredictionsButton, 'Enable', 'off');

% Mark training flag to be true
global training;
training = true;


% --- Executes on button press in stopTrainingButton.
function stopTrainingButton_Callback(hObject, eventdata, handles)
% hObject    handle to stopTrainingButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global net;
global training;
training = false;

% Add softmax layer on top of the network
net.layers = net.layers(1 : end-1); % Remove softmax loss layer
net.layers{end+1} = struct('type', 'softmax') ;

% Make the network format compatible
net = vl_simplenn_tidy(net);

set(handles.statusBar, 'string', 'Training stopped');

% Disable stop button and enable start button
set(handles.startTrainingButton, 'Enable', 'on');
set(handles.stopTrainingButton, 'Enable', 'off');

% Enable predictions button
set(handles.startPredictionsButton, 'Enable', 'on');


% --- Executes on button press in startPredictionsButton.
function startPredictionsButton_Callback(hObject, eventdata, handles)
% hObject    handle to startPredictionsButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.statusBar, 'string', 'Predictions turned on');

% Disable start button and enable stop button
set(handles.startPredictionsButton, 'Enable', 'off');
set(handles.stopPredictionsButton, 'Enable', 'on');

% Disable training button
set(handles.startTrainingButton, 'Enable', 'off');

global generatePredictions;
generatePredictions = true;


% --- Executes on button press in stopPredictionsButton.
function stopPredictionsButton_Callback(hObject, eventdata, handles)
% hObject    handle to stopPredictionsButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global generatePredictions;
generatePredictions = false;
set(handles.outputLabel, 'string', '');
set(handles.statusBar, 'string', 'Predictions turned off');

% Disable stop button and enable start button
set(handles.startPredictionsButton, 'Enable', 'on');
set(handles.stopPredictionsButton, 'Enable', 'off');

% Enable training button
set(handles.startTrainingButton, 'Enable', 'on');


% --- Executes on button press in reinitializeNetwork.
function reinitializeNetwork_Callback(hObject, eventdata, handles)
% hObject    handle to reinitializeNetwork (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global net;
global DAG;
global labelList;
if DAG
    disp('Method not supported for DAG');
    set(handles.statusBar, 'string', 'Method not supported for DAG');
else
    numClasses = length(labelList);
    net = addClassificationLayer(net, numClasses);
    net.meta.classes.name = labelList;
    net.meta.classes.description = labelList;
    set(handles.statusBar, 'string', 'Network reinitialization complete');
end


% --- Executes on button press in saveNetworkButton.
function saveNetworkButton_Callback(hObject, eventdata, handles)
% hObject    handle to saveNetworkButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global newModelFileName;
global net;
save(newModelFileName, 'net') ;
set(handles.statusBar, 'string', 'Model saved successfully');


% --- Executes on button press in loadNetworkButton.
function loadNetworkButton_Callback(hObject, eventdata, handles)
% hObject    handle to loadNetworkButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global newModelFileName;
global net;
load(newModelFileName) ;
net = vl_simplenn_tidy(net) ;
set(handles.statusBar, 'string', 'Model loaded successfully');


% --- Executes on selection change in labelListBox.
function labelListBox_Callback(hObject, eventdata, handles)
% hObject    handle to labelListBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns labelListBox contents as cell array
%        contents{get(hObject,'Value')} returns selected item from labelListBox
global choosenLabel;
choosenLabel = get(hObject,'Value');


% -------------------------------------------------------------------------
function err = error_multiclass(opts, labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
[~,predictions] = sort(predictions, 3, 'descend') ;

% be resilient to badly formatted labels
if numel(labels) == size(predictions, 4)
  labels = reshape(labels,1,1,1,[]) ;
end

% skip null labels
mass = single(labels(:,:,1,:) > 0) ;
if size(labels,3) == 2
  % if there is a second channel in labels, used it as weights
  mass = mass .* labels(:,:,2,:) ;
  labels(:,:,2,:) = [] ;
end

m = min(5, size(predictions,3)) ;

error = ~bsxfun(@eq, predictions, labels) ;
err(1,1) = sum(sum(sum(mass .* error(:,:,1,:)))) ;
err(2,1) = sum(sum(sum(mass .* min(error(:,:,1:m,:),[],3)))) ;