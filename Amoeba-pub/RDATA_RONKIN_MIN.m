function [mux, muy] = RDATA_RONKIN_MIN(obj, ESeed, varargin)

% Given the eigenvalue list, get the corresponding Ronkin minimum of obj
% Solving the full Amoeba GBZ will be done in another function RDATA_AMOEBA
%
% INPUTS
%     obj: A MonoArray object, see the MonoArray class for its properties
%     ESeed: a vector, usually spectra obtained from diagonalization or ML-GBZ.
%
% VARARGINS
%     DataID / DataTag: dataset name (DataTag takes precedence)
%     ResetData: false (default) resumes a compatible dataset; true resets it
%     mux,yRange: search range of mux,y, default [-1, 1]
%     NIntRes: optimization integral resolution, default 201
%     UseJensen: false (default) uses Ronkin; true uses RonkinJensen
%          Negative NIntRes is supported only by the original Ronkin path.
%     tol: fmincon StepTolerance, default 1e-4
%     NumStarts: number of deterministic initial points, default 3
%     NCheckRes: Jensen resolution for checking every candidate, default 2048
%          Candidates are checked without a second optimization.
%     EIndex: original indices of ESeed, default (1:numel(ESeed))'
%     NumSegments: number of consecutive energy segments, default 1
%     MaxSegments: maximum segments calculated in this call, default Inf
%          Attempted energies are skipped on resume, including failed attempts.
%
% OUTPUTS
%     mux,y: successful candidate with the lowest finite checked value
%          If every start fails, a finite fallback is diagnostic only.
%          Completed is true if any start succeeds; Status is complete/partial/retry.
%          Attempted records whether an energy has been calculated and saved.


%%%%% parse inputs
f = inputParser;
f.KeepUnmatched = true;
addRequired(f, 'obj', @(x) validateattributes(x,{'MonoArray'},{'scalar'}));
addRequired(f, 'ESeed', @(x) validateattributes(x,{'numeric'},{'vector', 'nonempty', 'finite'}));
addParameter(f, 'DataID', 0, @(x) validateattributes(x,{'numeric'},{'scalar', 'real', 'positive'}));
addParameter(f, 'DataTag', '', @(x) validateattributes(x,{'char', 'string'},{'nonempty'}));
addParameter(f, 'ResetData', false, @(x) validateattributes(x,{'logical'},{'scalar'}));
addParameter(f, 'muxRange', [-1, 1], @(x) validateattributes(x,{'numeric'},{'vector', 'numel', 2, 'real', 'finite'}));
addParameter(f, 'muyRange', [-1, 1], @(x) validateattributes(x,{'numeric'},{'vector', 'numel', 2, 'real', 'finite'}));
addParameter(f, 'NIntRes', 201, @(x) validateattributes(x,{'numeric'},{'scalar', 'real', 'finite'}));
addParameter(f, 'UseJensen', false, @(x) validateattributes(x,{'logical'},{'scalar'}));
addParameter(f, 'tol', 1e-4, @(x) validateattributes(x,{'numeric'},{'scalar', 'real', 'positive', 'finite'}));
addParameter(f, 'NumStarts', 3, @(x) validateattributes(x,{'numeric'},{'scalar', 'integer', 'positive', 'finite'}));
addParameter(f, 'NCheckRes', 2048, @(x) validateattributes(x,{'numeric'},{'scalar', 'integer', 'positive', 'finite'}));
addParameter(f, 'EIndex', [], @(x) validateattributes(x,{'numeric'},{'integer', 'real', 'positive', 'finite'}));
addParameter(f, 'NumSegments', 1, @(x) validateattributes(x,{'numeric'},{'scalar', 'integer', 'positive', 'finite'}));
addParameter(f, 'MaxSegments', Inf, @(x) validateattributes(x,{'numeric'},{'scalar', 'real', 'positive'}));
parse(f, obj, ESeed, varargin{:});

ESeed = ESeed(:);
NE = numel(ESeed);
EIndex = f.Results.EIndex;
if isempty(EIndex)
    EIndex = (1:NE)';
elseif ~isvector(EIndex) || numel(EIndex) ~= NE || numel(unique(EIndex)) ~= NE
    error('RDATA_RONKIN_MIN:InvalidEIndex', 'EIndex must contain one distinct index for each energy.');
end
EIndex = EIndex(:);
muxRange = f.Results.muxRange(:).';
muyRange = f.Results.muyRange(:).';
if muxRange(1) > muxRange(2) || muyRange(1) > muyRange(2)
    error('RDATA_RONKIN_MIN:InvalidRange', 'Each search range must be in ascending order.');
end
if f.Results.UseJensen
    validateattributes(f.Results.NIntRes, {'numeric'}, {'integer', 'positive'});
elseif f.Results.NIntRes >= 0 && round(f.Results.NIntRes) < 1
    error('RDATA_RONKIN_MIN:InvalidResolution', 'Nonnegative NIntRes must round to at least one.');
end
if isfinite(f.Results.MaxSegments)
    validateattributes(f.Results.MaxSegments, {'numeric'}, {'integer'});
end
StartMu = GetStartMu(muxRange, muyRange, f.Results.NumStarts);


%%%%% data path and exact calculation identity
FolderPath = './Revision2_Data';
DataTag = strtrim(string(f.Results.DataTag));
if strlength(DataTag) > 0
    DataPath = fullfile(FolderPath, sprintf('RonkinMin_%s.mat', DataTag));
else
    DataPath = fullfile(FolderPath, sprintf('RonkinMin_%d.mat', f.Results.DataID));
end

Config = struct();
Config.SchemaVersion = 3;
Config.CoupleMat = obj.CoupleMat;
Config.CoupleCenter = obj.CoupleCenter;
Config.ESeed = ESeed;
Config.EIndex = EIndex;
Config.muxRange = muxRange;
Config.muyRange = muyRange;
Config.NIntRes = f.Results.NIntRes;
Config.UseJensen = f.Results.UseJensen;
Config.tol = f.Results.tol;
Config.NumStarts = f.Results.NumStarts;
Config.NCheckRes = f.Results.NCheckRes;
Config.StartMu = StartMu;

EmptyDiagnostic = struct('EIndex', NaN, 'StartMu', StartMu, ...
    'CandidateMu', NaN(f.Results.NumStarts, 2), ...
    'Fval', NaN(f.Results.NumStarts, 1), ...
    'CheckValue', NaN(f.Results.NumStarts, 1), ...
    'CheckDifference', NaN(f.Results.NumStarts, 1), ...
    'ExitFlag', NaN(f.Results.NumStarts, 1), ...
    'FuncCount', NaN(f.Results.NumStarts, 1), ...
    'Succeeded', false(f.Results.NumStarts, 1), ...
    'BestStart', NaN, 'BestCheckValue', NaN, 'CandidateGap', NaN, ...
    'AtBoundary', false(f.Results.NumStarts, 1), 'Elapsed', NaN, ...
    'Status', 'pending', 'ErrorID', {repmat({''}, f.Results.NumStarts, 1)}, ...
    'ErrorMessage', {repmat({''}, f.Results.NumStarts, 1)});

if isfile(DataPath) && ~f.Results.ResetData
    data = load(DataPath);
    if ~isfield(data, 'Config')
        error('RDATA_RONKIN_MIN:MissingConfig', ...
            'Existing data has no Config. Use ResetData=true to replace this tag explicitly.');
    end
    if ~isequaln(data.Config, Config)
        error('RDATA_RONKIN_MIN:ConfigMismatch', ...
            'Existing Config does not match the model, energies, indices or calculation settings.');
    end
    RequiredFields = {'CoupleMat', 'CoupleCenter', 'ESeed', 'EIndex', ...
        'mux', 'muy', 'Attempted', 'Completed', 'Diagnostics', 'RunInfo'};
    if ~all(isfield(data, RequiredFields)) || ...
            ~isequaln(data.CoupleMat, obj.CoupleMat) || ...
            ~isequaln(data.CoupleCenter, obj.CoupleCenter) || ...
            ~isequaln(data.ESeed, ESeed) || ~isequaln(data.EIndex, EIndex) || ...
            ~isequal(size(data.mux), [NE, 1]) || ~isequal(size(data.muy), [NE, 1]) || ...
            ~islogical(data.Attempted) || ~isequal(size(data.Attempted), [NE, 1]) || ...
            ~islogical(data.Completed) || ~isequal(size(data.Completed), [NE, 1]) || ...
            any(data.Completed & ~data.Attempted) || ...
            ~isstruct(data.Diagnostics) || ~isequal(size(data.Diagnostics), [NE, 1]) || ...
            ~isequal(fieldnames(data.Diagnostics), fieldnames(EmptyDiagnostic)) || ...
            ~isstruct(data.RunInfo) || ~all(isfield(data.RunInfo, {'Segments', 'Calls'}))
        error('RDATA_RONKIN_MIN:InvalidCheckpoint', 'Existing data does not have the required checkpoint layout.');
    end
    mux = data.mux;
    muy = data.muy;
    Attempted = data.Attempted;
    Completed = data.Completed;
    Diagnostics = data.Diagnostics;
    RunInfo = data.RunInfo;
    fprintf('Resume %s: %d/%d attempted; %d/%d completed.\n', ...
        DataPath, nnz(Attempted), NE, nnz(Completed), NE);
else
    mux = NaN(NE, 1);
    muy = NaN(NE, 1);
    Attempted = false(NE, 1);
    Completed = false(NE, 1);
    Diagnostics = repmat(EmptyDiagnostic, NE, 1);
    for p = 1:NE
        Diagnostics(p).EIndex = EIndex(p);
    end
    RunInfo = struct();
    RunInfo.Segments = struct('SegmentIndex', {}, 'Index', {}, 'EIndex', {}, ...
        'NumSegments', {}, 'CallIndex', {}, 'Elapsed', {}, 'WorkerCount', {}, ...
        'AttemptedCount', {}, 'CompletedCount', {});
    RunInfo.Calls = struct('NumSegments', {}, 'MaxSegments', {}, 'Elapsed', {}, ...
        'WorkerCount', {}, 'ProcessedSegments', {});
end

if all(Attempted)
    fprintf('All %d energies are already attempted; %d/%d completed.\n', NE, nnz(Completed), NE);
    return;
end
if ~isfolder(FolderPath)
    mkdir(FolderPath);
end


%%%%% calculate and checkpoint one segment at a time
CallTimer = tic;
pool = gcp;
WorkerCount = pool.NumWorkers;
RunInfo.NumSegments = f.Results.NumSegments;
RunInfo.MaxSegments = f.Results.MaxSegments;
RunInfo.MATLABVersion = version;
RunInfo.WorkerCount = WorkerCount;
CallIndex = numel(RunInfo.Calls) + 1;
RunInfo.Calls(CallIndex) = struct('NumSegments', f.Results.NumSegments, ...
    'MaxSegments', f.Results.MaxSegments, 'Elapsed', 0, ...
    'WorkerCount', WorkerCount, 'ProcessedSegments', 0);

q = parallel.pool.DataQueue;
NRemaining = nnz(~Attempted);
UpdateParforProgress('init', [NE, nnz(Attempted), nnz(Completed)], max(1, round(NRemaining / 10)));
afterEach(q, @(Succeeded) UpdateParforProgress('tick', Succeeded));
ProgressCleanup = onCleanup(@() UpdateParforProgress('reset'));
Edges = floor((0:f.Results.NumSegments) .* NE ./ f.Results.NumSegments);
NProcessed = 0;
for s = 1:f.Results.NumSegments
    Index = (Edges(s)+1:Edges(s+1))';
    Index = Index(~Attempted(Index));
    if isempty(Index)
        continue;
    end
    if NProcessed >= f.Results.MaxSegments
        break;
    end

    SegmentTimer = tic;
    SegmentMu = NaN(numel(Index), 2);
    SegmentCompleted = false(numel(Index), 1);
    SegmentDiagnostics = repmat(EmptyDiagnostic, numel(Index), 1);
    SegmentEnergy = ESeed(Index);
    SegmentEIndex = EIndex(Index);
    parfor p = 1:numel(Index)
        [SegmentMu(p, :), SegmentCompleted(p), SegmentDiagnostics(p)] = ...
            GetRonkinMin(SegmentEnergy(p), SegmentEIndex(p), Config, EmptyDiagnostic);
        send(q, SegmentCompleted(p));
    end
    mux(Index) = SegmentMu(:, 1);
    muy(Index) = SegmentMu(:, 2);
    Completed(Index) = SegmentCompleted;
    Diagnostics(Index) = SegmentDiagnostics;
    Attempted(Index) = true;
    NProcessed = NProcessed + 1;
    RunInfo.Segments(end+1) = struct('SegmentIndex', s, 'Index', Index, ...
        'EIndex', EIndex(Index), 'NumSegments', f.Results.NumSegments, ...
        'CallIndex', CallIndex, 'Elapsed', toc(SegmentTimer), ...
        'WorkerCount', WorkerCount, 'AttemptedCount', numel(Index), ...
        'CompletedCount', nnz(SegmentCompleted));
    RunInfo.Calls(CallIndex).Elapsed = toc(CallTimer);
    RunInfo.Calls(CallIndex).ProcessedSegments = NProcessed;
    SaveCheckpoint(DataPath, obj, ESeed, EIndex, mux, muy, Attempted, Completed, Diagnostics, Config, RunInfo);
    fprintf('Saved segment %d/%d: %d/%d attempted; %d/%d completed.\n', ...
        s, f.Results.NumSegments, nnz(Attempted), NE, nnz(Completed), NE);
end
clear ProgressCleanup;
RunInfo.Calls(CallIndex).Elapsed = toc(CallTimer);
SaveCheckpoint(DataPath, obj, ESeed, EIndex, mux, muy, Attempted, Completed, Diagnostics, Config, RunInfo);
fprintf('Calculation paused/finished: %d/%d attempted; %d/%d completed; %d unattempted.\n', ...
    nnz(Attempted), NE, nnz(Completed), NE, nnz(~Attempted));

end




function StartMu = GetStartMu(muxRange, muyRange, NumStarts)

UnitMu = zeros(NumStarts, 2);
UnitMu(1, :) = [1/2, 1/2];
if NumStarts >= 2
    UnitMu(2, :) = [1/4, 3/4];
end
% Deterministic radical-inverse points in bases 2 and 3, without random state.
for p = 3:NumStarts
    for d = 1:2
        Base = d + 1;
        Index = p;
        Weight = 1 / Base;
        while Index > 0
            UnitMu(p, d) = UnitMu(p, d) + mod(Index, Base) * Weight;
            Index = floor(Index / Base);
            Weight = Weight / Base;
        end
    end
end
lb = [muxRange(1), muyRange(1)];
ub = [muxRange(2), muyRange(2)];
StartMu = lb + UnitMu .* (ub - lb);

end




function [mu, Completed, Diagnostic] = GetRonkinMin(E0, EIndex, Config, Diagnostic)

EnergyTimer = tic;
Diagnostic.EIndex = EIndex;
% Reconstruct on the worker: broadcasting this value class can restore defaults.
obj = MonoArray(Config.CoupleMat, Config.CoupleCenter);
if Config.UseJensen
    fun = @(mu) obj.RonkinJensen(mu(1), mu(2), E0, 'NIntRes', Config.NIntRes);
else
    fun = @(mu) obj.Ronkin(mu(1), mu(2), E0, 'NIntRes', Config.NIntRes);
end
lb = [Config.muxRange(1), Config.muyRange(1)];
ub = [Config.muxRange(2), Config.muyRange(2)];
opts = optimoptions('fmincon', 'Display', 'off', 'StepTolerance', Config.tol);

for p = 1:Config.NumStarts
    try
        [CandidateMu, Fval, ExitFlag, Output] = fmincon(fun, Config.StartMu(p, :), ...
            [], [], [], [], lb, ub, [], opts);
        Diagnostic.CandidateMu(p, :) = CandidateMu(:).';
        Diagnostic.Fval(p) = Fval;
        Diagnostic.ExitFlag(p) = ExitFlag;
        Diagnostic.FuncCount(p) = Output.funcCount;
        if any(~isfinite(CandidateMu)) || ~isfinite(Fval) || ~isfinite(ExitFlag)
            Diagnostic.ErrorID{p} = 'RDATA_RONKIN_MIN:NonfiniteCandidate';
            Diagnostic.ErrorMessage{p} = 'The optimizer returned a nonfinite candidate, value or exit flag.';
        elseif ExitFlag <= 0
            Diagnostic.ErrorID{p} = 'RDATA_RONKIN_MIN:OptimizationNotConverged';
            Diagnostic.ErrorMessage{p} = Output.message;
        end
        if all(isfinite(CandidateMu))
            Diagnostic.AtBoundary(p) = any(abs(CandidateMu(:).'-lb) <= Config.tol | ...
                abs(CandidateMu(:).'-ub) <= Config.tol);
            Diagnostic.CheckValue(p) = obj.RonkinJensen(CandidateMu(1), CandidateMu(2), E0, ...
                'NIntRes', Config.NCheckRes);
            Diagnostic.CheckDifference(p) = Diagnostic.CheckValue(p) - Fval;
            if ~isfinite(Diagnostic.CheckValue(p))
                Diagnostic.ErrorID{p} = 'RDATA_RONKIN_MIN:NonfiniteCheck';
                Diagnostic.ErrorMessage{p} = 'The Jensen candidate check returned a nonfinite value.';
            end
        end
    catch Error
        Diagnostic.ErrorID{p} = Error.identifier;
        Diagnostic.ErrorMessage{p} = Error.message;
    end
end

FiniteCheck = isfinite(Diagnostic.CheckValue) & all(isfinite(Diagnostic.CandidateMu), 2);
Diagnostic.Succeeded = FiniteCheck & isfinite(Diagnostic.Fval) & ...
    isfinite(Diagnostic.ExitFlag) & Diagnostic.ExitFlag > 0 & cellfun(@isempty, Diagnostic.ErrorID);
Completed = any(Diagnostic.Succeeded);
Selected = Diagnostic.Succeeded;
if ~Completed
    Selected = FiniteCheck;
end
mu = [NaN, NaN];
if any(Selected)
    CandidateIndex = find(Selected);
    [Diagnostic.BestCheckValue, BestIndex] = min(Diagnostic.CheckValue(Selected));
    Diagnostic.BestStart = CandidateIndex(BestIndex);
    mu = Diagnostic.CandidateMu(Diagnostic.BestStart, :);
end
if all(FiniteCheck)
    Diagnostic.CandidateGap = max(Diagnostic.CheckValue) - min(Diagnostic.CheckValue);
end
if all(Diagnostic.Succeeded)
    Diagnostic.Status = 'complete';
elseif Completed
    Diagnostic.Status = 'partial';
else
    Diagnostic.Status = 'retry';
end
Diagnostic.Elapsed = toc(EnergyTimer);

end




function SaveCheckpoint(DataPath, obj, ESeed, EIndex, mux, muy, Attempted, Completed, Diagnostics, Config, RunInfo)

data = struct('CoupleMat', obj.CoupleMat, 'CoupleCenter', obj.CoupleCenter, ...
    'ESeed', ESeed, 'EIndex', EIndex, 'mux', mux, 'muy', muy, ...
    'Attempted', Attempted, 'Completed', Completed, ...
    'Diagnostics', Diagnostics, 'Config', Config, 'RunInfo', RunInfo);
TempPath = [tempname(fileparts(DataPath)), '.mat'];
TempCleanup = onCleanup(@() DeleteTemporaryFile(TempPath));
save(TempPath, '-struct', 'data', '-v7.3');
[Success, Message] = movefile(TempPath, DataPath, 'f');
if ~Success
    error('RDATA_RONKIN_MIN:CheckpointSaveFailed', 'Checkpoint replacement failed: %s', Message);
end
clear TempCleanup;

end




function DeleteTemporaryFile(TempPath)

if isfile(TempPath)
    delete(TempPath);
end

end




function UpdateParforProgress(action, Value, nPrint)

persistent nDone nTotal nStep nAttempted nCompleted
switch action
    case 'init'
        nDone = 0;
        nTotal = Value(1);
        nAttempted = Value(2);
        nCompleted = Value(3);
        nStep = nPrint;
    case 'tick'
        if isempty(nTotal)
            return;
        end
        nDone = nDone + 1;
        nAttempted = nAttempted + 1;
        nCompleted = nCompleted + double(Value);
        if mod(nDone, nStep) == 0 || nAttempted == nTotal
            fprintf('......Calculating......%d/%d attempted; %d/%d completed.\n', ...
                nAttempted, nTotal, nCompleted, nTotal);
        end
    case 'reset'
        nDone = [];
        nTotal = [];
        nStep = [];
        nAttempted = [];
        nCompleted = [];
end

end

