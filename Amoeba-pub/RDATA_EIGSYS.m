
function [E, S] = RDATA_EIGSYS(obj, Ny, Nx, varargin)

% Get eigensystems obtained by directly solving the eigen-system of the OBC system.
% Save the data in a folder named EigSys_Data_(id), and lable it by Ny,x
%
% INPUTS
%     obj: a MonoArray2D object
%     Ny,x: number of sites along y and x direction for a 2D rectangular shape.
%
% VARARGINS
%     fDrop: function handle. Determine the sites that need to be dropped.
%          fDrop should accept x and y as inputs, and return boolen values. Those sites whose
%               positions (x, y) satisfying fDrop will be neglected.
%     DataID: a positive integer. Density data will be saved in dataset 'data' + DataID.
%     DataTag: a string/char. If DataTag is given, use this tag rather than DataID.
%     ResetData: false (default) | true
%          Whether to reset the dataset when it is existed.


%%%%% parse inputs
f = inputParser;
addRequired(f, 'obj', @(x) validateattributes(x,{'MonoArray'},{'scalar'}));
addRequired(f, 'Ny', @(x) validateattributes(x,{'numeric'},{'scalar', 'real', 'positive'}));
addRequired(f, 'Nx', @(x) validateattributes(x,{'numeric'},{'scalar', 'real', 'positive'}));
addParameter(f, 'fDrop', [], @(x) validatefDrop(x));
addParameter(f, 'DataID', 0, @(x) validateattributes(x,{'numeric'},{'scalar', 'real', 'positive'}));
addParameter(f, 'DataTag', '', @(x) validateattributes(x,{'char', 'string'},{'nonempty'}));
addParameter(f, 'ResetData', false, @(x) validateattributes(x,{'logical'},{'scalar'}));
parse(f, obj, Ny, Nx, varargin{:});


%%%%% file path and key
FolderPath = './Revision2_Data';
if ~isfolder(FolderPath)
    mkdir(FolderPath);
    fprintf("Folder %s not found. New data folder created.\n", FolderPath);
end

DataTag = strtrim(string(f.Results.DataTag));
if strlength(DataTag) > 0
    DataPath = fullfile(FolderPath, sprintf('EigSys_%s.mat', DataTag));
else
    DataPath = fullfile(FolderPath, sprintf('EigSys_%d.mat', f.Results.DataID));
end


%%%%% check whether current Ny, Nx data already exists
need_calc = false;
if isfile(DataPath)
    if ~f.Results.ResetData
        fprintf("Data already existed at %s (ResetData = false). Directly load it.\n", DataPath);
        data = load(DataPath, 'EigVal', 'EigVec');
        E = data.EigVal;
        S = data.EigVec;
        fprintf("Successfully load data from %s.\n", DataPath);

    else
        need_calc = true;
        fprintf("Data %s not existed, but ResetData = true. Recalculating.\n", DataPath);
    end
else
    need_calc = true;
    fprintf("Data not found at %s. Calculating.\n", DataPath);
end


%%%%% claculation
if need_calc
    fprintf("Start calculating, this will take some time...\n")
    [HObc, pos] = obj.HamObc(Ny, Nx, 'fDrop', f.Results.fDrop);
    [S, E] = eig(HObc, 'vector');

    data = struct();
    data.CoupleMat = obj.CoupleMat;
    data.CoupleCenter = obj.CoupleCenter;
    data.EigVal = E;
    data.EigVec = S;
    data.Xpos = pos.Xpos;
    data.Ypos = pos.Ypos;
    
    save(DataPath, "-struct", "data", "-v7.3");
    fprintf("Successfully save data to %s.\n", DataPath);
end


end







%% validate func
function validatefDrop(f)
if ~isempty(f)
    validateattributes(f, {'function_handle'}, {'scalar'});

    if nargin(f) ~= 2
        error('"fDrop" should accept two inputs x and y.');
    end

    try
        test_out = f([1, 2], [3, 4]);
        if ~islogical(test_out)
            error('"fDrop" should return logical arrays.');
        end
    catch
        error('Error in "fDrop". Please cheack your input function.');
    end
end

end