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

% Last Modified by GUIDE v2.5 04-Jan-2017 02:41:57

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

% Initialize the recognition engine
initializeRecognitionEngine;
set(handles.statusBar, 'string', 'Initialization completed');

% Create a callback for updating the system
handles.timer = timer(...
    'ExecutionMode', 'fixedRate', ...       % Run timer repeatedly.
    'Period', 0.033, ...                    % Initial period is 33 mili-sec.
    'TimerFcn', {@updateDisplay, handles}); % Specify callback function.

start(handles.timer);

global labelList;
set(handles.labelListBox, 'string', labelList);


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
global DAG;

im = snapshot(cam);
imshow(im, 'Parent', handles.imagePlaceholder);

if generatePredictions
    img = single(im);
    img = imresize(img, [opts.imageDim, opts.imageDim], 'bilinear');
    averageImage = mean(mean(img));
    img = bsxfun(@minus, img, averageImage);

    % Feed the image through the network
    if DAG
        net.eval({'data', img});

        % Extract probabilities from the network
        activation = net.vars(net.getVarIndex(opts.layerName)).value;
        activation = squeeze(gather(activation));

        [conf, ind] = max(activation);
        myText = net.meta.classes.description{ind};
    else
        res = vl_simplenn(net, img);
        activation = squeeze(gather(res(end).x));
        
        [conf, ind] = max(activation);
        myText = net.meta.classes.description{ind};
    end

    set(handles.outputLabel, 'string', myText);
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


% --- Executes on button press in stopCam.
function stopCam_Callback(hObject, eventdata, handles)
% hObject    handle to stopCam (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global cam;
if isempty(cam), return, end;
%clear cam;
cam = [];
set(handles.statusBar, 'string', 'Camera stopped');

% if strcmp(get(handles.timer, 'Running'), 'on')
%     stop(handles.timer);
% end


% --- Executes on button press in startTrainingButton.
function startTrainingButton_Callback(hObject, eventdata, handles)
% hObject    handle to startTrainingButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global training;
training = true;
set(handles.statusBar, 'string', 'Training started');


% --- Executes on button press in stopTrainingButton.
function stopTrainingButton_Callback(hObject, eventdata, handles)
% hObject    handle to stopTrainingButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global training;
training = false;
set(handles.statusBar, 'string', 'Training stopped');


% --- Executes on button press in startPredictionsButton.
function startPredictionsButton_Callback(hObject, eventdata, handles)
% hObject    handle to startPredictionsButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global generatePredictions;
generatePredictions = true;
set(handles.statusBar, 'string', 'Predictions turned on');


% --- Executes on button press in stopPredictionsButton.
function stopPredictionsButton_Callback(hObject, eventdata, handles)
% hObject    handle to stopPredictionsButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global generatePredictions;
generatePredictions = false;
set(handles.outputLabel, 'string', '');
set(handles.statusBar, 'string', 'Predictions turned off');


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
    set(handles.statusBar, 'string', 'Network reinitialization complete');
end


% --- Executes on button press in saveNetworkButton.
function saveNetworkButton_Callback(hObject, eventdata, handles)
% hObject    handle to saveNetworkButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
