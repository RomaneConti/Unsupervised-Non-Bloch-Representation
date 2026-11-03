
function [HObc, varargout] = HamObc(obj, Ny, Nx, varargin)

% The OBC Hamiltonian for a 2D (square-shaped) single-bands model
% Possibly drop some sites base on the Ny*Nx square shape.
%
% INPUTS 
%     obj: A MonoCArray object, see the MonoArray class for its properties.
%     Ny,x: number of unit cells (atoms) along y and x in the obc array.
%
% VARARGINS
%     fDrop: function handle. Determine the sites that need to be dropped.
%          fDrop should accept x and y as inputs, and return boolen values. Those sites whose
%               positions (x, y) satisfying fDrop will be neglected.
%
% OUTPUTS
%     HObc: OBC Hamiltonian with possible dropped sites.
%     pos (varargout): a struct, with fields 'Xpos' and 'Ypos'.
%          Representing the position of each site


%%%%% parse inputs
f = inputParser;
addRequired(f, 'obj', @(x) validateattributes(x,{'MonoArray'},{'nonempty'}));
addRequired(f, 'Ny', @(x) validateattributes(x,{'numeric'},{'scalar', 'real', 'positive'}));
addRequired(f, 'Nx', @(x) validateattributes(x,{'numeric'},{'scalar', 'real', 'positive'}));
addParameter(f, 'fDrop', [], @(x) validatefDrop(x));
parse(f, obj, Ny, Nx, varargin{:});

CoupleMat = obj.CoupleMat;
CoupleCenter = obj.CoupleCenter;
if Ny < size(CoupleMat, 1)
    warning(['In MonoArray>HamObc.', newline,...
        'The size along y direction cannot cover the maximum hopping range. Automatically set Ny=%d.'], size(CoupleMat, 1));
    Ny = size(CoupleMat, 1);
end

if Nx < size(CoupleMat, 2)
    warning(['In MonoArray>HamObc.', newline,...
        'The size along x direction cannot cover the maximum hopping range. Automatically set Nx=%d.'], size(CoupleMat, 2));
    Nx = size(CoupleMat, 2);
end


%%%%% get hopping range towards left, right, up and down
nonZeroID = find(CoupleMat);
nonZeroID = nonZeroID(:);
JLst = CoupleMat(nonZeroID);
JLst = JLst(:);
[row, col] = ind2sub(size(CoupleMat), nonZeroID);
CoupleDirLst = [row, col] - CoupleCenter;

%%% Note for CoupleDir [Rx, Ry], there will be a term J*exp(ikx*Rx)*exp(iky*Ry) in HPbc.
%%% This term corresponds to c^dagger(x-Rx, y-Ry) c(x, y) in HObc
NR = CoupleCenter(2) - 1;
NL = size(CoupleMat, 2) - CoupleCenter(2);
ND = CoupleCenter(1) - 1;
NU = size(CoupleMat, 1) - CoupleCenter(1);


%%%%% HObc
%%% First, construct Hamiltonian for geometry (Ny+NU+ND) * (Nx+NL+NR), to prevent boundary check
%%% The target system corresponds to y index from NU+1 to NU+Ny, x index from NL+1 to NL+Nx
Nrow = Ny + NU + ND;
Ncol = Nx + NL + NR;
Ham = zeros( Nrow * Ncol);

for idJ = 1 : length(JLst)
    J = JLst(idJ);
    Rx = CoupleDirLst(idJ, 2);
    Ry = CoupleDirLst(idJ, 1);
    for idy = NU+1 : NU+Ny
        for idx = NL+1 : NL+Nx
            Ham( (idx-Rx-1)*Nrow + idy-Ry, (idx-1)*Nrow + idy) = J;
        end
    end
end


%%%%% extract square HObc form the Nrow*Ncol system
ObcID = transpose( NU + (1:Ny)) + Nrow .* (NL-1 + (1 : Nx));
ObcID = ObcID(:);
HObc = Ham(ObcID, ObcID);


%%%%% drop
[Xpos, Ypos] = meshgrid(0: Nx-1, Ny-1 :-1 :0);
if ~isempty(f.Results.fDrop)
    idDrop = f.Results.fDrop(Xpos, Ypos);
    Xpos(idDrop) = [];
    Ypos(idDrop) = [];
    HObc(idDrop, :) = [];
    HObc(:, idDrop) = [];
end


%%%%% varargout
switch nargout
    case {0, 1}
        varargout = {};
    case 2
        varargout = {struct('Xpos', Xpos(:), 'Ypos', Ypos(:))};
    otherwise
        error('Too many outputs!');
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
