
classdef MonoArray

% class of the monatomic 2D array
%
% A MonoArray object has the following properties
%
%     CoupleMat: a 2d matrix, represening the coupling coeffcients of the 2D 
%
%     Center: a vector, given a position in the CoupleMat.
%          All the elements in the CoupleMat start from this position.
%          E.g., CoupleMat = [0 3 0; 1 0 2; 0 4 0] and Center = [2, 2] represent 
%               the hopping towards left/right/up/down is 1/2/3/4.
%
% The MonoChain object supports the following methods
%
%     HPbc/HObc: Hamiltonian under peridic/open boundary condition


    properties
        CoupleMat double = [0 3 0; 1 0 2; 0 4 0];
        CoupleCenter double = [2 2];
    end

    properties (Access = private)
        needValidation logical = true;
    end


    methods

        function obj = MonoArray(CoupleMat0, CoupleCenter0)
%             disp('construct')
            if nargin == 0
                obj.CoupleMat = [0 3 0; 1 0 2; 0 4 0];
                obj.CoupleCenter = [2 2];
            else
                obj.needValidation = false;
                tempobj = obj;
                tempobj.CoupleMat = CoupleMat0;
                tempobj.CoupleCenter = CoupleCenter0;
                tempobj.needValidation = true;
                tempobj.validateproperties;
                obj = tempobj;
            end
        end


        %%%%% set validity functions
        function obj = set.CoupleMat(obj, newMat)
%             disp('setMat')
            validateattributes(newMat, {'numeric'}, {'nonempty', '2d'});
            if all( newMat(:)==0 )
                error(['In MonoArray/set.CoupleMat', newline, 'Property "CoupleMat" cannot be all-zero matrix.'])
            end
            tempobj = obj;
            tempobj.CoupleMat = newMat;
            tempobj.validateproperties;
            obj = tempobj;
        end

        function obj = set.CoupleCenter(obj, newCenter)
%             disp('setCenter')
            validateattributes(newCenter, {'numeric'}, {'nonempty', 'vector', 'real', 'positive', 'numel', 2});
            tempobj = obj;
            tempobj.CoupleCenter = newCenter(:).';
            tempobj.validateproperties;
            obj = tempobj;
        end

        function validateproperties(obj)
            if obj.needValidation && any( size(obj.CoupleMat) - obj.CoupleCenter(:).' < 0)
                error(['In MonoArray/validateproperties', newline,...
                    'Property "CoupleCenter"  must be within the dimensions of "CoupleMat".']);
            end
        end


        %%%%% methods
        
        ShowConfig(obj)

        HPbc = HamPbc(obj, betax, betay)

        [HObc, varargout] = HamObc(obj, Ny, Nx, varargin)

        R = Ronkin(obj, mux, muy, E0, varargin)

        [EObc, SObc] = EigHighPrec(obj, NumCell, varargin)

        PltEig(obj, varargin)

    end


    methods (Static)

        

        
    end


end
