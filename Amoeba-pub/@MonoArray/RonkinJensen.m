function R = RonkinJensen(obj, mux, muy, E0, varargin)

% The Ronkin function with the kx integral evaluated by Jensen's formula
% R = mean log(abs(H(exp(mux+ikx), exp(muy+iky)) - E0)) over kx, ky.
%
% INPUTS
%     mux,y: log absolute value of betax,y
%     E0: ref energy
%
% VARARGINS
%     NIntRes: positive integer resolution of the ky midpoint integral
%          default is 1024; there is no numerical kx integral.
%
% For each ky, write H-E0 = betax^nmin * P(betax). If a is the leading
% coefficient of P and z are its roots, the kx mean is
%     log(abs(a)) + nmin*mux + sum(max(mux, log(abs(z)))).
% Negative nmin accounts for the pole at zero. No log smoothing is used.


%%%%% parse inputs
f = inputParser;
addRequired(f, 'obj', @(x) validateattributes(x,{'MonoArray'},{'scalar'}));
addRequired(f, 'mux', @(x) validateattributes(x,{'numeric'},{'scalar', 'real', 'finite'}));
addRequired(f, 'muy', @(x) validateattributes(x,{'numeric'},{'scalar', 'real', 'finite'}));
addRequired(f, 'E0', @(x) validateattributes(x,{'numeric'},{'scalar', 'finite'}));
addParameter(f, 'NIntRes', 1024, @(x) validateattributes(x,{'numeric'},...
    {'scalar', 'real', 'finite', 'integer', 'positive'}));
parse(f, obj, mux, muy, E0, varargin{:});
validateattributes(obj.CoupleCenter, {'numeric'}, {'integer'});
validateattributes(obj.CoupleMat, {'numeric'}, {'finite'});


%%%%% Laurent coefficients, in ascending powers of betax
% nx = col - CoupleCenter(2); ny = CoupleCenter(1) - row, as in HamPbc.
Nk = f.Results.NIntRes;
kLst = -pi + ((0:Nk-1) + 1/2) .* (2*pi/Nk);
ny = obj.CoupleCenter(1) - (1:size(obj.CoupleMat, 1));
yPart = exp(ny(:) .* (muy + 1i.*kLst));
% Use transpose, not conjugate transpose: the couplings may be complex.
Coeff = transpose(obj.CoupleMat) * yPart;
Coeff(obj.CoupleCenter(2), :) = Coeff(obj.CoupleCenter(2), :) - E0;


%%%%% Jensen integral
Rx = zeros(1, Nk);
for p = 1:Nk
    a = Coeff(:, p);
    first = find(a ~= 0, 1, 'first');
    if isempty(first)
        % This entire kx slice is zero. A sampled log singularity gives -Inf,
        % as in Ronkin; an isolated ky slice need not make the true integral diverge.
        Rx(p) = -Inf;
        continue;
    end
    last = find(a ~= 0, 1, 'last');
    nmin = first - obj.CoupleCenter(2);
    % Remove exact zero coefficients only, including ky-dependent degree loss.
    a = flipud(a(first:last));
    if first == last
        Rx(p) = log(abs(a(1))) + nmin*mux;
    else
        z = roots(a);
        Rx(p) = log(abs(a(1))) + nmin*mux + sum(max(mux, log(abs(z))));
    end
end
R = mean(Rx);


end
