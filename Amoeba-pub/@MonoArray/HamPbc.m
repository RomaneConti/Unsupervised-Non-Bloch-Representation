
function HPbc = HamPbc(obj, betax, betay)

% The bulk Hamiltonian for a 2D single-bands model
%
% INPUTS
%     obj: A MonoArray object, see the MonoArray class for its properties
%     betax,y: the generalized wave vector, beta = exp(ik) with k the Bloch wave vector


%%%%% parse inputs
f = inputParser;
addRequired(f, 'obj', @(x) validateattributes(x,{'MonoArray'},{'nonempty'}));
addRequired(f, 'betax', @(x) validateattributes(x,{'numeric'},{'nonzero'}));
addRequired(f, 'betay', @(x) validateattributes(x,{'numeric'},{'nonzero'}));
parse(f, obj, betax, betay);

if any( size(betax) - size(betay) ~= 0)
    error(['In MonoArray>HamPbc.', newline, 'Inputs "betax" and "betay" should have the same shape.']);
end


%%%%% Ham
nonZeroID = find(obj.CoupleMat);
nonZeroID = nonZeroID(:);
J = obj.CoupleMat(nonZeroID);
J = J(:);
[row, col] = ind2sub(size(obj.CoupleMat), nonZeroID);
CoupleDir = [row, col] - obj.CoupleCenter;

% Note that, in the x direction, elements at left have smaller indices
% However, in the y direction, upper elements have smaller elements, which need to be fixed.
% eg, given CoupleMat = [0 3 0; 1 0 2; 0 4 0] and Center = [2, 2], element 3 should correspond
%     to CoupleDir (0, 1) rather than (0, -1)
CoupleDir = CoupleDir .* [-1 1];

% HPbc = 0;
% for p = 1 : length(J)
%     HPbc = HPbc + J(p) .* (betax.^CoupleDir(p, 2)) .* (betay.^CoupleDir(p, 1));
% end

xPart = transpose(betax(:)) .^ CoupleDir(:, 2);
yPart = transpose(betay(:)) .^ CoupleDir(:, 1);
HPbc = reshape(sum(J(:) .* (xPart .* yPart)), size(betax));



end