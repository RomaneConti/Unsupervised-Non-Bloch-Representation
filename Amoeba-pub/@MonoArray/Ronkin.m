
function R = Ronkin(obj, mux, muy, E0, varargin)

% The Ronkin function for given parameters
% R = sum log( abs( f(exp(μ+ik)) )) dkx*dky / 4pi^2, where we define β=exp(μ+ik)
%
% INPUTS
%     mux,y: log absolute value of betax,y
%     E0: ref energy
%     t1, t2, gamma: hopping parameters
%
% VARARGINS
%     NIntRes: resolution of the numrical integral
%          default is -1, which represents that do not use numerical integral


%%%%% parse inputs
f = inputParser;
addRequired(f, 'obj', @(x) validateattributes(x,{'MonoArray'},{'nonempty'}));
addRequired(f, 'mux', @(x) validateattributes(x,{'numeric'},{'scalar', 'real'}));
addRequired(f, 'muy', @(x) validateattributes(x,{'numeric'},{'scalar', 'real'}));
addRequired(f, 'E0', @(x) validateattributes(x,{'numeric'},{'scalar'}));
addParameter(f, 'NIntRes', -1, @(x) validateattributes(x,{'numeric'},{'scalar'}));
parse(f, obj, mux, muy, E0, varargin{:});


%%%%% Ronkin func
if f.Results.NIntRes < 0
    func = @(kx, ky) arrayfun(@(kx, ky)...
        log( abs( obj.HamPbc( exp(mux+1i.*kx), exp(muy+1i.*ky)) - E0)), kx, ky);
    R = integral2(func, -pi, pi, -pi, pi) ./ 4 ./ pi^2;
else
    % use numerical integral
    Nk = round(f.Results.NIntRes);
    kLst = -pi + ((0:Nk-1) + 1/2) .* (2*pi/Nk);
    [kx, ky] = meshgrid(kLst);
    tmp = obj.HamPbc( exp(mux+1i.*kx), exp(muy+1i.*ky)) - E0;
    R = sum(log(abs(tmp)), 'all') ./ Nk^2;
end


