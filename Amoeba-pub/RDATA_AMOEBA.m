function Amoeba = RDATA_AMOEBA(RonkinMinDataPath, varargin)

% Solve contour intersections at each saved Ronkin minimum.
% The contourc/polyxpoly, optional fsolve and curve-resampling rules are unchanged.
%
% VARARGINS
%     DataID, DataTag, ResetData: output name and explicit restart, as before.
%     UseFSolve: false by default; refine contour intersections using fsolve.
%     NContourMesh: 301; KSeedTol: 1e-3; KRootTol: 1e-4.
%     ZeroThreshold: 1e-3; NResample: 25 (per degenerate contour).
%     NumSegments: 1; split input positions into this many consecutive segments.
%     MaxSegments: Inf; at most this many nonempty work segments per call.
% Attempted skips every saved calculation, including errors, on resume.
% Completed includes normal no-candidate/no-root results; errors remain unusable.
% Amoeba keeps kx/ky/mux/muy/E and adds EIndex, PointKind and PointResidual.
% PointKind is approximate, refined or curve_resample, not a root certificate.
% FSolveRaw{i} retains every attempted seed, including rejected/failed solves.
% KFinal is unwrapped; PeriodicShift and ShiftNorm use the periodic BZ.

%%%%% parse inputs
f = inputParser;
f.KeepUnmatched = true;
addRequired(f, 'RonkinMinDataPath', @(x) validateattributes(x,{'char', 'string'},{'nonempty'}));
addParameter(f, 'DataID', 0, @(x) validateattributes(x,{'numeric'},{'scalar', 'real', 'positive'}));
addParameter(f, 'DataTag', '', @(x) validateattributes(x,{'char', 'string'},{'nonempty'}));
addParameter(f, 'ResetData', false, @(x) validateattributes(x,{'logical'},{'scalar'}));
addParameter(f, 'UseFSolve', false, @(x) validateattributes(x,{'logical'},{'scalar'}));
addParameter(f, 'NContourMesh', 301, @(x) validateattributes(x,{'numeric'},{'scalar', 'integer', 'finite', '>=', 2}));
addParameter(f, 'KSeedTol', 1e-3, @(x) validateattributes(x,{'numeric'},{'scalar', 'real', 'finite', 'positive'}));
addParameter(f, 'KRootTol', 1e-4, @(x) validateattributes(x,{'numeric'},{'scalar', 'real', 'finite', 'positive'}));
addParameter(f, 'ZeroThreshold', 1e-3, @(x) validateattributes(x,{'numeric'},{'scalar', 'real', 'finite', 'positive'}));
addParameter(f, 'NResample', 25, @(x) validateattributes(x,{'numeric'},{'scalar', 'integer', 'finite', 'positive'}));
addParameter(f, 'NumSegments', 1, @(x) validateattributes(x,{'numeric'},{'scalar', 'integer', 'finite', 'positive'}));
addParameter(f, 'MaxSegments', Inf, @(x) validateattributes(x,{'numeric'},{'scalar', 'real', 'positive'}));
parse(f, RonkinMinDataPath, varargin{:});
if isnan(f.Results.MaxSegments) || (isfinite(f.Results.MaxSegments) && fix(f.Results.MaxSegments) ~= f.Results.MaxSegments)
    error('RDATA_AMOEBA:MaxSegments', 'MaxSegments must be a positive integer or Inf.');
end

%%%%% source identity and eligibility
source = load(RonkinMinDataPath);
Required = {'CoupleMat', 'CoupleCenter', 'ESeed', 'mux', 'muy'};
if ~all(isfield(source, Required))
    error('RDATA_AMOEBA:SourceSchema', 'Ronkin data is missing required fields.');
end
obj = MonoArray(source.CoupleMat, source.CoupleCenter);
validateattributes(source.ESeed, {'numeric'}, {'vector', 'nonempty'});
ESeed = source.ESeed(:);
NE = numel(ESeed);
validateattributes(source.mux, {'numeric'}, {'vector', 'real', 'numel', NE});
validateattributes(source.muy, {'numeric'}, {'vector', 'real', 'numel', NE});
mux = source.mux(:);
muy = source.muy(:);
if isfield(source, 'EIndex')
    validateattributes(source.EIndex, {'numeric'}, {'vector', 'real', 'integer', 'finite', 'positive', 'numel', NE});
    EIndex = source.EIndex(:);
    EIndexSource = 'source';
else
    EIndex = (1:NE).';
    EIndexSource = 'legacy_sequential';
end
if numel(unique(EIndex)) ~= NE
    error('RDATA_AMOEBA:SourceEIndex', 'Source EIndex must contain unique positive integers.');
end
FiniteSource = isfinite(ESeed) & isfinite(mux) & isfinite(muy);
if isfield(source, 'Completed')
    validateattributes(source.Completed, {'numeric', 'logical'}, {'vector', 'numel', NE});
    if any(~ismember(source.Completed(:), [0 1]))
        error('RDATA_AMOEBA:SourceCompleted', 'Source Completed must be logical or contain only 0 and 1.');
    end
    SourceCompleted = logical(source.Completed(:));
    CompletedSource = 'source';
else
    SourceCompleted = FiniteSource;
    CompletedSource = 'legacy_finite_mu_energy';
end
Eligible = SourceCompleted & FiniteSource;
SourceConfig = [];
if isfield(source, 'Config')
    SourceConfig = source.Config;
end

%%%%% scientific configuration (execution segmentation may change on resume)
Config = struct('SchemaVersion', 3, 'CoupleMat', obj.CoupleMat,...
    'CoupleCenter', obj.CoupleCenter, 'ESeed', ESeed, 'EIndex', EIndex,...
    'mux', mux, 'muy', muy, 'SourceRonkinConfig', SourceConfig,...
    'UseFSolve', f.Results.UseFSolve, 'NContourMesh', f.Results.NContourMesh,...
    'KSeedTol', f.Results.KSeedTol, 'KRootTol', f.Results.KRootTol,...
    'ZeroThreshold', f.Results.ZeroThreshold, 'NResample', f.Results.NResample,...
    'FSolveFunctionTolerance', 1e-12, 'FSolveStepTolerance', 1e-12,...
    'FSolveResidualTolerance', 1e-8);
FolderPath = './Revision2_Data';
DataTag = strtrim(string(f.Results.DataTag));
if strlength(DataTag) > 0
    DataPath = fullfile(FolderPath, sprintf('Amoeba_%s.mat', DataTag));
else
    DataPath = fullfile(FolderPath, sprintf('Amoeba_%d.mat', f.Results.DataID));
end

D0 = EmptyDiagnostic;
data = struct();
data.CoupleMat = obj.CoupleMat;
data.CoupleCenter = obj.CoupleCenter;
data.ESeed = ESeed;
data.EIndex = EIndex;
data.mux = mux;
data.muy = muy;
data.Attempted = false(NE, 1);
data.Completed = false(NE, 1);
data.Diagnostics = repmat(D0, NE, 1);
data.ResCell = repmat({zeros(0, 5)}, NE, 1);
data.PointKindCell = repmat({strings(0, 1)}, NE, 1);
data.PointResidualCell = repmat({zeros(0, 1)}, NE, 1);
data.FSolveRaw = repmat({repmat(EmptyFSolveRecord, 0, 1)}, NE, 1);
data.Config = Config;
data.RunInfo = struct('NumSegments', f.Results.NumSegments,...
    'MaxSegments', f.Results.MaxSegments, 'MATLABVersion', version,...
    'WorkerCount', 0, 'StartedAt', char(datetime('now')),...
    'UpdatedAt', char(datetime('now')), 'SourceRonkinPath', char(RonkinMinDataPath),...
    'EIndexSource', EIndexSource, 'CompletedSource', CompletedSource,...
    'Segments', struct('Segment', {}, 'EIndex', {}, 'Elapsed', {}, 'WorkerCount', {}, 'AttemptedCount', {}, 'CompletedCount', {}));
if isfile(DataPath) && ~f.Results.ResetData
    previous = load(DataPath);
    Required = {'Amoeba', 'Config', 'Attempted', 'Completed', 'Diagnostics', 'ResCell', 'PointKindCell',...
        'PointResidualCell', 'FSolveRaw', 'ESeed', 'EIndex', 'mux', 'muy', 'RunInfo'};
    if ~all(isfield(previous, Required)) || ~isstruct(previous.Config) || ~isscalar(previous.Config) ||...
            ~isstruct(previous.RunInfo) || ~isscalar(previous.RunInfo) ||...
            ~all(isfield(previous.RunInfo, {'Segments', 'StartedAt'}))
        error('RDATA_AMOEBA:ResumeSchema', 'Existing output has no compatible resume schema. Use ResetData=true to replace it explicitly.');
    end
    if ~isfield(previous.Config, 'mux') || ~isfield(previous.Config, 'muy') ||...
            ~isequaln(rmfield(previous.Config, {'mux', 'muy'}), rmfield(Config, {'mux', 'muy'}))
        error('RDATA_AMOEBA:ResumeConfig', 'Saved numerical configuration or energy identity differs from the request.');
    end
    if ~isequaln(previous.ESeed(:), ESeed) || ~isequaln(previous.EIndex(:), EIndex) ||...
            ~islogical(previous.Attempted) || ~isequal(size(previous.Attempted), [NE, 1]) ||...
            ~islogical(previous.Completed) || numel(previous.Completed) ~= NE ||...
            any(previous.Completed(:) & ~previous.Attempted) ||...
            numel(previous.mux) ~= NE || numel(previous.muy) ~= NE ||...
            ~iscell(previous.ResCell) || numel(previous.ResCell) ~= NE ||...
            ~iscell(previous.PointKindCell) || numel(previous.PointKindCell) ~= NE ||...
            ~iscell(previous.PointResidualCell) || numel(previous.PointResidualCell) ~= NE ||...
            ~iscell(previous.FSolveRaw) || numel(previous.FSolveRaw) ~= NE ||...
            ~all(cellfun(@(r) isstruct(r) && isequal(fieldnames(r), fieldnames(EmptyFSolveRecord)), previous.FSolveRaw)) ||...
            ~isstruct(previous.Diagnostics) || numel(previous.Diagnostics) ~= NE ||...
            ~isequal(fieldnames(previous.Diagnostics), fieldnames(D0))
        error('RDATA_AMOEBA:ResumeIdentity', 'Saved per-energy arrays do not match the configured identities.');
    end
    Done = previous.Attempted;
    if any(Done & ~Eligible) || ~isequaln(previous.mux(Done), mux(Done)) ||...
            ~isequaln(previous.muy(Done), muy(Done))
        error('RDATA_AMOEBA:ResumeMu', 'An attempted energy has changed source mu or is no longer eligible.');
    end
    data.Attempted = Done;
    data.Completed = previous.Completed(:);
    data.Diagnostics = previous.Diagnostics(:);
    data.ResCell = previous.ResCell(:);
    data.PointKindCell = previous.PointKindCell(:);
    data.PointResidualCell = previous.PointResidualCell(:);
    data.FSolveRaw = previous.FSolveRaw(:);
    data.RunInfo.Segments = previous.RunInfo.Segments;
    data.RunInfo.StartedAt = previous.RunInfo.StartedAt;
    for p = find(Done).'
        P = data.ResCell{p};
        if size(P, 2) ~= 5 || any(~isfinite(P(:))) ||...
                ~isequaln(P(:, 3:5), repmat([mux(p), muy(p), ESeed(p)], size(P, 1), 1)) ||...
                numel(data.PointKindCell{p}) ~= size(P, 1) || numel(data.PointResidualCell{p}) ~= size(P, 1) ||...
                any(~isfinite(data.PointResidualCell{p})) || data.Diagnostics(p).EIndex ~= EIndex(p)
            error('RDATA_AMOEBA:ResumePoints', 'Saved points do not match their exact source energy and mu.');
        end
    end
    fprintf('Resume %s: %d/%d attempted; %d/%d completed.\n',...
        DataPath, nnz(data.Attempted), NE, nnz(data.Completed), NE);
    if all(Done)
        fprintf('All energies have already been attempted; no calculation repeated.\n');
        Amoeba = previous.Amoeba;
        return;
    end
end
for p = 1:NE
    if ~data.Attempted(p)
        % Only unattempted energies may follow updates to the upstream minimum.
        SourceMuChanged = exist('previous', 'var') && ...
            (~isequaln(previous.mux(p), mux(p)) || ~isequaln(previous.muy(p), muy(p)));
        if ~Eligible(p) || SourceMuChanged
            data.ResCell{p} = zeros(0, 5);
            data.PointKindCell{p} = strings(0, 1);
            data.PointResidualCell{p} = zeros(0, 1);
            data.FSolveRaw{p} = repmat(EmptyFSolveRecord, 0, 1);
        end
        if ~Eligible(p) || SourceMuChanged || startsWith(data.Diagnostics(p).Status, 'upstream_')
            data.Diagnostics(p) = D0;
        end
        data.Diagnostics(p).EIndex = EIndex(p);
        if ~SourceCompleted(p)
            data.Diagnostics(p).Status = 'upstream_incomplete';
        elseif ~FiniteSource(p)
            data.Diagnostics(p).Status = 'upstream_nonfinite';
            data.Diagnostics(p).ErrorID = 'RDATA_AMOEBA:SourceNonfinite';
            data.Diagnostics(p).ErrorMessage = 'Source energy or mu is nonfinite.';
        end
    end
end
data.SourceCompleted = SourceCompleted;

%%%%% process missing energies; only the client assembles and saves results
if ~isfolder(FolderPath)
    mkdir(FolderPath);
end
opts = [];
if Config.UseFSolve
    opts = optimoptions('fsolve', 'Display', 'off',...
        'FunctionTolerance', Config.FSolveFunctionTolerance, 'StepTolerance', Config.FSolveStepTolerance);
end
Edges = floor((0:f.Results.NumSegments) .* NE ./ f.Results.NumSegments);
nSaved = 0;
for segment = 1:f.Results.NumSegments
    if nSaved >= f.Results.MaxSegments
        break;
    end
    Range = (Edges(segment)+1:Edges(segment+1)).';
    Work = Range(~data.Attempted(Range) & Eligible(Range));
    if isempty(Work)
        continue;
    end
    tSegment = tic;
    Points = cell(numel(Work), 1);
    Kinds = cell(numel(Work), 1);
    Residuals = cell(numel(Work), 1);
    Raw = cell(numel(Work), 1);
    Diagnostics = repmat(D0, numel(Work), 1);
    Completed = false(numel(Work), 1);
    ESegment = ESeed(Work);
    IndexSegment = EIndex(Work);
    MuxSegment = mux(Work);
    MuySegment = muy(Work);
    parfor j = 1:numel(Work)
        [Points{j}, Kinds{j}, Residuals{j}, Diagnostics(j), Completed(j), Raw{j}] =...
            SolveEnergy(ESegment(j), IndexSegment(j), MuxSegment(j), MuySegment(j), Config, opts);
    end
    data.ResCell(Work) = Points;
    data.PointKindCell(Work) = Kinds;
    data.PointResidualCell(Work) = Residuals;
    data.FSolveRaw(Work) = Raw;
    data.Diagnostics(Work) = Diagnostics;
    data.Attempted(Work) = true;
    data.Completed(Work) = Completed;
    pool = gcp('nocreate');
    WorkerCount = 0;
    if ~isempty(pool)
        WorkerCount = pool.NumWorkers;
    end
    record = struct('Segment', segment, 'EIndex', EIndex(Work),...
        'Elapsed', toc(tSegment), 'WorkerCount', WorkerCount,...
        'AttemptedCount', numel(Work), 'CompletedCount', nnz(Completed));
    data.RunInfo.Segments(end+1) = record;
    data.RunInfo.WorkerCount = WorkerCount;
    data.RunInfo.UpdatedAt = char(datetime('now'));
    data.Amoeba = AssembleAmoeba(data);
    SaveCheckpoint(DataPath, data);
    nSaved = nSaved + 1;
    fprintf('Amoeba segment %d/%d saved; %d/%d attempted; %d/%d completed.\n',...
        segment, f.Results.NumSegments, nnz(data.Attempted), NE, nnz(data.Completed), NE);
    for j = find(~Completed).'
        warning('RDATA_AMOEBA:EnergyFailed', 'EIndex %d: %s (%s)',...
            EIndex(Work(j)), Diagnostics(j).ErrorMessage, Diagnostics(j).ErrorID);
    end
end
if nSaved == 0
    data.Amoeba = AssembleAmoeba(data);
    data.RunInfo.UpdatedAt = char(datetime('now'));
    SourceChanged = false;
    if exist('previous', 'var')
        SourceChanged = ~isfield(previous, 'SourceCompleted') ||...
            ~isequaln(previous.SourceCompleted, data.SourceCompleted) ||...
            ~isequaln(previous.mux, data.mux) || ~isequaln(previous.muy, data.muy) ||...
            ~isequaln(previous.Diagnostics, data.Diagnostics);
    end
    if ~isfile(DataPath) || f.Results.ResetData || SourceChanged
        SaveCheckpoint(DataPath, data);
    end
end
fprintf('Amoeba paused/finished: %d/%d attempted; %d/%d completed; %d unattempted.\n',...
    nnz(data.Attempted), NE, nnz(data.Completed), NE, nnz(~data.Attempted));
Amoeba = data.Amoeba;

end


function D = EmptyDiagnostic
D = struct('EIndex', NaN, 'nSeed', NaN, 'nPoint', 0, 'nResampled', 0,...
    'nDiscrete', NaN, 'nFSolveSuccess', NaN, 'nFSolveFail', NaN,...
    'FSolveShift', [],...
    'ResidualMax', NaN, 'ResidualMedian', NaN, 'IsRealZero', false,...
    'IsImagZero', false, 'NoCandidate', false, 'Elapsed', 0,...
    'Status', 'pending', 'ErrorID', '', 'ErrorMessage', '');
end


function Record = EmptyFSolveRecord
Record = struct('KSeed', [NaN NaN], 'KFinal', [NaN NaN],...
    'ResidualSeed', NaN, 'ResidualFinal', NaN, 'ExitFlag', NaN,...
    'Iterations', NaN, 'FuncCount', NaN, 'PeriodicShift', [NaN NaN],...
    'ShiftNorm', NaN, 'Accepted', false, 'ErrorID', '', 'ErrorMessage', '');
end


function [Points, Kind, Residual, D, Completed, FSolveRaw] = SolveEnergy(E0, EIndex, mux, muy, Config, opts)
Points = zeros(0, 5);
Kind = strings(0, 1);
Residual = zeros(0, 1);
FSolveRaw = repmat(EmptyFSolveRecord, 0, 1);
D = EmptyDiagnostic;
D.EIndex = EIndex;
Completed = false;
tEnergy = tic;
try
    % Reconstruct on the worker instead of serializing the MonoArray object.
    obj = MonoArray(Config.CoupleMat, Config.CoupleCenter);
    [KSeed, isRealZero, isImagZero, D.nSeed, D.nResampled] = GetKSeed(obj, mux, muy, E0,...
        Config.NContourMesh, Config.KSeedTol, Config.ZeroThreshold, Config.NResample);
    D.IsRealZero = isRealZero;
    D.IsImagZero = isImagZero;
    D.NoCandidate = isempty(KSeed);
    if any(~isfinite(KSeed(:)))
        error('RDATA_AMOEBA:NonfiniteSeed', 'Contour seeds contain nonfinite values.');
    end
    if isRealZero && isImagZero
        % Preserve the original all-BZ case: no representative points are added.
        KRoot = zeros(0, 2);
        Label = "curve_resample";
        D.Status = 'degenerate_all_bz';
    elseif isRealZero || isImagZero
        KRoot = KSeed;
        Label = "curve_resample";
        D.Status = 'curve_resample';
    elseif Config.UseFSolve
        fun = @(k) [
            real(obj.HamPbc(exp(mux + 1i.*k(1)), exp(muy + 1i.*k(2))) - E0);
            imag(obj.HamPbc(exp(mux + 1i.*k(1)), exp(muy + 1i.*k(2))) - E0)
        ];
        KRoot = zeros(size(KSeed, 1), 2);
        nRoot = 0;
        D.nFSolveSuccess = 0;
        D.nFSolveFail = 0;
        FSolveRaw = repmat(EmptyFSolveRecord, size(KSeed, 1), 1);
        for ii = 1:size(KSeed, 1)
            Record = EmptyFSolveRecord;
            Record.KSeed = KSeed(ii, :);
            try
                Record.ResidualSeed = norm(fun(KSeed(ii, :).'));
                [kroot, fval, exitflag, output] = fsolve(fun, KSeed(ii, :).', opts);
                Record.KFinal = kroot(:).';
                Record.ResidualFinal = norm(fval);
                Record.ExitFlag = exitflag;
                Record.Iterations = output.iterations;
                Record.FuncCount = output.funcCount;
                Shift = Record.KFinal - Record.KSeed;
                Record.PeriodicShift = atan2(sin(Shift), cos(Shift));
                Record.ShiftNorm = norm(Record.PeriodicShift);
                if any(~isfinite(kroot(:))) || any(~isfinite(fval(:)))
                    error('RDATA_AMOEBA:NonfiniteFSolve', 'fsolve returned nonfinite values.');
                end
                Record.Accepted = exitflag > 0 && norm(fval) < Config.FSolveResidualTolerance;
            catch err
                Record.ErrorID = err.identifier;
                Record.ErrorMessage = err.message;
            end
            FSolveRaw(ii) = Record;
            if Record.Accepted
                % Paired seed-to-root distance on the periodic BZ, before deduplication.
                D.FSolveShift(end+1, 1) = Record.ShiftNorm;
                kroot = mod(Record.KFinal + pi, 2*pi) - pi;
                nRoot = nRoot + 1;
                KRoot(nRoot, :) = kroot(:).';
                D.nFSolveSuccess = D.nFSolveSuccess + 1;
            else
                D.nFSolveFail = D.nFSolveFail + 1;
            end
        end
        KRoot(nRoot+1:end, :) = [];
        if ~isempty(KRoot)
            KRoot = uniquetol(KRoot, Config.KRootTol, 'ByRows', true);
        end
        Label = "refined";
        D.Status = 'refined';
        D.nDiscrete = size(KRoot, 1);
    else
        KRoot = KSeed;
        Label = "approximate";
        D.Status = 'approximate';
        D.nDiscrete = size(KRoot, 1);
    end
    if ~isempty(KRoot)
        Residual = abs(obj.HamPbc(exp(mux + 1i.*KRoot(:, 1)), exp(muy + 1i.*KRoot(:, 2))) - E0);
        Residual = Residual(:);
        if any(~isfinite(Residual))
            error('RDATA_AMOEBA:NonfiniteResidual', 'Returned points have nonfinite residuals.');
        end
        Points = [KRoot, repmat([mux, muy, E0], size(KRoot, 1), 1)];
        Kind = repmat(Label, size(KRoot, 1), 1);
        D.ResidualMax = max(Residual);
        D.ResidualMedian = median(Residual);
    elseif ~(isRealZero && isImagZero)
        if D.NoCandidate
            D.Status = 'no_candidate';
        else
            D.Status = 'no_refined_root';
        end
    end
    D.nPoint = size(Points, 1);
    D.NoCandidate = D.nPoint == 0;
    Completed = true;
    % A thrown/nonfinite solve is unusable, but its saved attempt is not repeated.
    ErrorIndex = find(~cellfun(@isempty, {FSolveRaw.ErrorID}) | ...
        ~cellfun(@isempty, {FSolveRaw.ErrorMessage}), 1);
    if ~isempty(ErrorIndex)
        Completed = false;
        D.Status = 'error';
        D.ErrorID = FSolveRaw(ErrorIndex).ErrorID;
        D.ErrorMessage = FSolveRaw(ErrorIndex).ErrorMessage;
    end
catch err
    Points = zeros(0, 5);
    Kind = strings(0, 1);
    Residual = zeros(0, 1);
    D.nPoint = 0;
    D.Status = 'error';
    D.ErrorID = err.identifier;
    D.ErrorMessage = err.message;
end
D.Elapsed = toc(tEnergy);
end


function Amoeba = AssembleAmoeba(data)
Points = vertcat(data.ResCell{:});
IndexCell = cell(numel(data.EIndex), 1);
for p = 1:numel(data.EIndex)
    IndexCell{p} = repmat(data.EIndex(p), size(data.ResCell{p}, 1), 1);
end
Amoeba = struct('kx', Points(:, 1), 'ky', Points(:, 2), 'mux', Points(:, 3),...
    'muy', Points(:, 4), 'E', Points(:, 5), 'EIndex', vertcat(IndexCell{:}),...
    'PointKind', vertcat(data.PointKindCell{:}), 'PointResidual', vertcat(data.PointResidualCell{:}));
end


function SaveCheckpoint(DataPath, data)
TempPath = [tempname(fileparts(DataPath)), '.mat'];
Cleanup = onCleanup(@() DeleteTemporary(TempPath));
save(TempPath, '-struct', 'data', '-v7.3');
[ok, message] = movefile(TempPath, DataPath, 'f');
if ~ok
    error('RDATA_AMOEBA:SaveFailed', 'Cannot replace checkpoint: %s', message);
end
end


function DeleteTemporary(TempPath)
if isfile(TempPath)
    delete(TempPath);
end
end

function [KSeed, isRealZero, isImagZero, nSeed, nResampled] = GetKSeed(obj, mux, muy, E0, NContourMesh, KSeedTol, ZeroThreshold, NResample)
% Given an obj and the corresponding mux, muy, E
% Find the kx, ky satisfying the characteristic function obj.HamPbc(exp(mux+ikx), exp(muy+iky)) = E
% This will be done by finding the intersection of contour plot 
%     of real and imag part of the characteristic function at level 0
% Note that in some cases the real or imag part of the characteristic function will be zero

    nSeed = 0;
    nResampled = 0;
    kLst = pi.* linspace(-1, 1, NContourMesh);
    [kx, ky] = meshgrid(kLst);
    Z = obj.HamPbc(exp(mux + 1i.*kx), exp(muy + 1i.*ky)) - E0;
    
    if any(~isfinite(Z(:)))
        error('RDATA_AMOEBA:NonfiniteContour', 'Contour Hamiltonian values are nonfinite.');
    end
    maxRe = max(abs(real(Z(:))));
    maxIm = max(abs(imag(Z(:))));
    isRealZero = maxRe < ZeroThreshold;
    isImagZero = maxIm < ZeroThreshold;
    
    if isRealZero && isImagZero
        KSeed = zeros(0, 2);
        return;
    end
    
    if ~isRealZero && ~isImagZero
        cRe = contourc(kLst, kLst, real(Z), [0 0]);
        cIm = contourc(kLst, kLst, imag(Z), [0 0]);
        cReLst = SplitContourC(cRe);
        cImLst = SplitContourC(cIm);

        if numel(cReLst) == 0 || numel(cImLst) == 0
            KSeed = zeros(0, 2);
            return;
        end

        seedCell = repmat({zeros(0, 2)}, numel(cReLst), numel(cImLst));
        for ii = 1 : numel(cReLst)
            tmpcRe = cReLst{ii};
            for jj = 1 : numel(cImLst)
                tmpcIm = cImLst{jj};
                [tmpkx, tmpky] = polyxpoly(tmpcRe(:, 1), tmpcRe(:, 2), tmpcIm(:, 1), tmpcIm(:, 2));
                if ~isempty(tmpkx)
                    seedCell{ii, jj} = [tmpkx, tmpky];
                end
            end
        end
        KSeed = vertcat(seedCell{:});
        nSeed = size(KSeed, 1);
    
    else
        % one part is zero, directly sampling on the contour
        if isRealZero
            ZZ = imag(Z);
        else
            ZZ = real(Z);
        end
    
        c = SplitContourC( contourc(kLst, kLst, ZZ, [0 0]));

        if isempty(c)
            KSeed = zeros(0, 2);
            return;
        end

        seeds = repmat({zeros(0, 2)}, numel(c), 1);
        for ii = 1 : numel(c)
            seeds{ii} = ResampleByArcLength(c{ii}, NResample);
        end
        KSeed = vertcat(seeds{:});
        nResampled = size(KSeed, 1);
            
    end
    
    KSeed = mod(KSeed + pi, 2*pi) - pi;
    KSeed = uniquetol(KSeed, KSeedTol, 'ByRows', true);

end





%% helper func 3
function pts_new = ResampleByArcLength(pts, Nnew)
    x = pts(:, 1);
    y = pts(:, 2);

    ds = sqrt(diff(x).^2 + diff(y).^2);
    s = [0; cumsum(ds)];
    [s_unique, ia] = unique(s, 'stable');
    x = x(ia);
    y = y(ia);

    if s_unique(end) == 0
        pts_new = pts(1,:);
        return;
    end

    s_new = linspace(0, s_unique(end), Nnew).';
    x_new = interp1(s_unique, x, s_new, 'linear');
    y_new = interp1(s_unique, y, s_new, 'linear');

    pts_new = [x_new, y_new];
end







%% helper func 4
function contours = SplitContourC(C)
% split the contour results into separated lines
    k = 1;
    nSeg = 0;
    while k < size(C, 2)
        npt = C(2, k);
        nSeg = nSeg + 1;
        k = k + npt + 1;
    end
    
    contours = cell(nSeg, 1);
    k = 1;
    id = 0;
    while k < size(C, 2)
        id = id + 1;
    
        npt = C(2, k);
        x = C(1, k+1:k+npt).';
        y = C(2, k+1:k+npt).';
    
        contours{id} = [x, y];
        k = k + npt + 1;
    end
end
