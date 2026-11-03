%%%%% Load formal Round50 results
DataFold = fullfile(fileparts(mfilename('fullpath')), 'Revision2_Data');
R1 = load(fullfile(DataFold, 'RonkinMin_Model1_Round50.mat'), 'Diagnostics');
R2 = load(fullfile(DataFold, 'RonkinMin_Model2_Round50.mat'), 'Diagnostics');
R3 = load(fullfile(DataFold, 'RonkinMin_Model3_Round50.mat'), 'Diagnostics');
A1 = load(fullfile(DataFold, 'Amoeba_Model1_Round50.mat'));
A2 = load(fullfile(DataFold, 'Amoeba_Model2_Round50.mat'));
A3 = load(fullfile(DataFold, 'Amoeba_Model3_Round50.mat'));

%%%%% Figure, axes, layout: each row is Model 1, 2, 3
h = figure(1146); clf(h);
set(h, 'Position', [300 60 1040 480], 'Color', 'w');
ax0 = axes(h, 'Position', [0 0 1 1], 'Tag', 'BackGround'); hold(ax0, 'on');
initInvisibleAxes(ax0);

ax1a = axes(h, 'Position', [0.04 0.74 0.15 0.20], 'Tag', 'A');
ax1b = axes(h, 'Position', [0.25 0.74 0.15 0.20], 'Tag', 'A');
ax1c = axes(h, 'Position', [0.46 0.74 0.145 0.20], 'Tag', 'A');
ax1d = axes(h, 'Position', [0.65 0.74 0.145 0.20], 'Tag', 'A');
ax1e = axes(h, 'Position', [0.85 0.74 0.145 0.20], 'Tag', 'A');

ax2a = axes(h, 'Position', [0.04 0.42 0.15 0.20], 'Tag', 'A');
ax2b = axes(h, 'Position', [0.25 0.42 0.15 0.20], 'Tag', 'A');
ax2c = axes(h, 'Position', [0.46 0.42 0.145 0.20], 'Tag', 'A');
ax2d = axes(h, 'Position', [0.65 0.42 0.145 0.20], 'Tag', 'A');
ax2e = axes(h, 'Position', [0.85 0.42 0.145 0.20], 'Tag', 'A');

ax3a = axes(h, 'Position', [0.04 0.10 0.15 0.20], 'Tag', 'A');
ax3b = axes(h, 'Position', [0.25 0.10 0.15 0.20], 'Tag', 'A');
ax3c = axes(h, 'Position', [0.46 0.10 0.145 0.20], 'Tag', 'A');
ax3d = axes(h, 'Position', [0.65 0.10 0.145 0.20], 'Tag', 'A');
ax3e = axes(h, 'Position', [0.85 0.10 0.145 0.20], 'Tag', 'A');

%%%%% Plot
AmoebaArgs = {'UseFSolve', true, 'PointTol', 1e-6, 'CurveTol', Inf};
HistArgs = {'Norm', true, 'BinEdges', -16:0.5:-3};
fsolveArgs = {'PointTol', 1e-6};

[~, P1] = PltAmoeba(ax1a, A1, 'Comp', 'x', AmoebaArgs{:});
PltAmoeba(ax1b, A1, 'Comp', 'y', AmoebaArgs{:});
PltEigResidual(ax1c, P1, HistArgs{:});
PltRonkinResidual(ax1d, R1.Diagnostics, HistArgs{:});
Pltfsolve(ax1e, A1.FSolveRaw, fsolveArgs{:});

[~, P2] = PltAmoeba(ax2a, A2, 'Comp', 'x', AmoebaArgs{:});
PltAmoeba(ax2b, A2, 'Comp', 'y', AmoebaArgs{:});
PltEigResidual(ax2c, P2, HistArgs{:});
PltRonkinResidual(ax2d, R2.Diagnostics, HistArgs{:});
Pltfsolve(ax2e, A2.FSolveRaw, fsolveArgs{:});

[~, P3] = PltAmoeba(ax3a, A3, 'Comp', 'x', AmoebaArgs{:});
PltAmoeba(ax3b, A3, 'Comp', 'y', AmoebaArgs{:});
PltEigResidual(ax3c, P3, HistArgs{:});
PltRonkinResidual(ax3d, R3.Diagnostics, HistArgs{:});
Pltfsolve(ax3e, A3.FSolveRaw, fsolveArgs{:});

%%%%% retouch
set(ax1c, 'YLim', [0 0.6], 'YTick', [0 0.2 0.4 0.6]);
set(ax2c, 'YLim', [0 0.25], 'YTick', [0 0.1 0.2]);
set(ax3c, 'YLim', [0 0.25], 'YTick', [0 0.1 0.2]);
arrayfun(@(ax) set(ax, 'YLim', [0 0.55], 'YTick', [0 0.2 0.4]), [ax1d; ax2d; ax3d]);
arrayfun(@(ax) view(ax, [24.6 40.7]), [ax1a; ax1b; ax2a; ax2b; ax3a; ax3b]);

legend(ax1d, ax1d.Children([2 1]), {'$\Delta R_{\mathrm{start}}$', '$\Delta R_{\mathrm{int}}$'},...
    'Interpreter', 'Latex', 'FontSize', 11, 'box', 'off', 'Position', [0.65 0.85 0.10 0.08]);
legend(ax1e, ax1e.Children([1 2]), {'Rejected', 'Accepted'},...
    'Interpreter', 'tex', 'FontSize', 10, 'box', 'off', 'Position', [0.85 0.85 0.10 0.08]);


% %%%%% Save plots only
axes_3d = [ax1a; ax1b; ax2a; ax2b; ax3a; ax3b; ax2e; ax3e];
axImg = Rasterize3DObjects(h, axes_3d, 600);

uistack(ax0, 'top');
uistack(axImg, 'bottom');

all_axes = [ax1a, ax1b, ax1c, ax1d, ax1e, ax2a, ax2b, ax2c, ax2d, ax2e, ax3a, ax3b, ax3c, ax3d, ax3e];
for id = 1 : length(all_axes)
    ax = all_axes(id);
    abc = {'a', 'b', 'c'};
    x = ax.Position(1);
    y = ax.Position(2) + ax.Position(4);
    T = sprintf('(%s%d)', abc{ceil(id/5)}, mod(id-1, 5)+1);
    text(ax0, 100*x-4, 100*y+3, T, 'FontName', 'Arial', 'FontSize', 10);
end

text(ax0, 92.5, 81, 'Not applicable', 'FontName', 'Arial', 'FontSize', 10, 'HorizontalAlignment', 'center');
text(ax0, 92.5, 77, '(continuous contours)', 'FontName', 'Arial', 'FontSize', 10, 'HorizontalAlignment', 'center');







function initInvisibleAxes(ax)
set(ax, 'XColor', 'None', 'YColor', 'None', 'Color', 'None');
set(ax, 'XLim', [0 100], 'YLim', [0 100], 'XTick', [], 'YTick', []);
ax.Interactions = [];
ax.Toolbar.Visible = 'off';
end



function axImg = Rasterize3DObjects(h, axes_3d, dpi)
% use this function to export a vector plot into pdf
% first print all 3D axes as rgb data
% delete all 3D objects and use a axes to show them, keep all 2d objects

tmph = figure(19198); clf(tmph);
set(tmph, 'Position', h.Position, 'InvertHardcopy','off');
tmp_axes_3d = copyobj(axes_3d, tmph);
Invisible3Dlabels(tmp_axes_3d);

set(tmph, 'Color', 'k');
rgb_black = im2double(print(tmph, '-RGBImage', sprintf('-r%d', dpi)));
set(tmph, 'Color', 'w');
rgb_white = im2double(print(tmph, '-RGBImage', sprintf('-r%d', dpi)));

alpha = 1 - mean(rgb_white - rgb_black, 3);
alpha = max(min(alpha, 1),0);
rgb = rgb_black ./ max(alpha, 1e-6);
rgb = max(min(rgb, 1), 0);

close(tmph);

Invisible3Dobjs(axes_3d);
axImg = axes(h, 'Position', [0 0 1 1]);
image(axImg, 'XData', [0 1], 'YData', [0 1], 'CData', rgb, 'AlphaData', alpha);
set(axImg, 'XLim', [0 1], 'YLim', [0 1]);
set(axImg, 'YDir', 'Reverse', 'XTick', [], 'YTick', []);
set(axImg, 'XColor', 'None', 'YColor', 'None', 'Color', 'None');

end


function Invisible3Dobjs(axes)
    for ii = 1 : numel(axes)
        ax = axes(ii);
        ZLim = ax.ZLim;
        ZTick = ax.ZTick;
        ZExponent = ax.ZAxis.Exponent;

        box(ax, 'off');
        grid(ax, 'off');
        delete(ax.Children);
%         set(ax.Children, 'visible', 'off');
        
        pause(1e-6);
        ax.XRuler.Axle.Visible = 'off';
        ax.YRuler.Axle.Visible = 'off';
        try
            ax.ZRuler.Axle.Visible = 'off';
        catch
            '2D ax does not have Z';
        end
        ax.Color = 'none';

        set(ax, 'ZLim', ZLim, 'ZTick', ZTick);
        ax.ZAxis.Exponent = ZExponent;
    end
end


function Invisible3Dlabels(axes)
    for ax = axes
        set(ax, 'XTickLabel', [], 'YTickLabel', [], 'ZTickLabel', [], 'TickLength', [0 0]);
        txt_objs = findall(ax, 'Type', 'Text');
        delete(txt_objs);
    end
end






function [h, P] = PltAmoeba(ax, data, varargin)
f = inputParser; f.KeepUnmatched = true;
addParameter(f, 'Comp', 'x', @(x) any(strcmpi(x, {'x','y'})));
addParameter(f, 'UseFSolve', true, @(x) islogical(x) && isscalar(x));
addParameter(f, 'PointTol', 1e-8, @(x) validateattributes(x, {'numeric'}, {'scalar','real','nonnegative','nonnan'}));
addParameter(f, 'CurveTol', Inf, @(x) validateattributes(x, {'numeric'}, {'scalar','real','nonnegative','nonnan'}));
parse(f, varargin{:});
scatterArgs = reshape([fieldnames(f.Unmatched), struct2cell(f.Unmatched)].', 1, []);

Points = cell(numel(data.FSolveRaw), 1);
kind = "seed";
if f.Results.UseFSolve, kind = "refined"; end
for i = 1:numel(Points)
    raw = data.FSolveRaw{i};
    if isempty(raw), continue; end
    if f.Results.UseFSolve
        K = vertcat(raw.KFinal); r = [raw.ResidualFinal].';
        valid = [raw.ExitFlag].' > 0;
    else
        K = vertcat(raw.KSeed); r = [raw.ResidualSeed].';
        valid = true(size(r));
    end
    valid = valid & all(isfinite(K), 2) & isfinite(r) & r >= 0 & r < f.Results.PointTol;
    K = mod(K(valid,:) + pi, 2*pi) - pi; r = r(valid);
    % Per-energy periodic angular tolerance; keep the lowest-residual representative.
    [r, order] = sort(r); K = K(order,:); keep = false(size(r));
    for j = 1:numel(r)
        dk = abs(mod(K(keep,:) - K(j,:) + pi, 2*pi) - pi);
        keep(j) = ~any(all(dk <= data.Config.KRootTol, 2));
    end
    Points{i} = [K(keep,:), repmat([data.mux(i), data.muy(i), data.ESeed(i), data.EIndex(i)], nnz(keep), 1), r(keep)];
end
points = vertcat(Points{:});
if isempty(points), points = zeros(0,7); end
nDiscrete = size(points,1);
A = data.Amoeba;
curve = strcmp(A.PointKind, 'curve_resample') & isfinite(A.PointResidual) & A.PointResidual < f.Results.CurveTol;
points = [points; A.kx(curve), A.ky(curve), A.mux(curve), A.muy(curve), A.E(curve), A.EIndex(curve), A.PointResidual(curve)];
P = struct('kx', real(points(:,1)), 'ky', real(points(:,2)), 'mux', real(points(:,3)), 'muy', real(points(:,4)), ...
    'E', points(:,5), 'EIndex', real(points(:,6)), 'PointResidual', real(points(:,7)), ...
    'PointKind', [repmat(kind,nDiscrete,1); repmat("curve_resample",nnz(curve),1)]);
comp = lower(char(f.Results.Comp));
h = scatter3(ax, P.kx/pi, P.ky/pi, P.(['mu' comp]), 1, [0 47 196]/255, 'filled', ...
    'MarkerFaceAlpha', 0.55, scatterArgs{:});
xlabel(ax, '$k_x/\pi$', 'Interpreter', 'Latex', 'FontSize', 10);
ylabel(ax, '$k_y/\pi$', 'Interpreter', 'Latex', 'FontSize', 10);
zlabel(ax, ['$\mu_' comp '$'], 'Interpreter', 'Latex', 'FontSize', 11);
box(ax, 'on'); grid(ax, 'on');
set(ax, 'XLim', [-1 1], 'YLim', [-1 1], 'ZLim', [-0.5 0.5], ...
    'XTick', [-1 0 1], 'YTick', [-1 0 1], 'ZTick', [-0.5 0 0.5]);
view(ax, [30 39]);
end


function h = PltEigResidual(ax, data, varargin)
f = inputParser; f.KeepUnmatched = true;
addParameter(f, 'Norm', true, @(x) islogical(x) && isscalar(x));
parse(f, varargin{:});
histArgs = reshape([fieldnames(f.Unmatched), struct2cell(f.Unmatched)].', 1, []);
norm = 'count'; 
Ty = '$N$';
if f.Results.Norm
    norm = 'probability';
    Ty = '$P$';
end
r = data.PointResidual;
r = log10(max(r(isfinite(r) & r >= 0), eps));
h = histogram(ax, r, 'Normalization', norm, 'FaceColor', [0 47 196]/255, 'EdgeColor', 'none', histArgs{:});
box(ax, 'on');
xlabel(ax, '$\log_{10}|H-E|$', 'Interpreter', 'Latex', 'FontSize', 10);
ylabel(ax, Ty, 'Interpreter', 'Latex', 'FontSize', 11);
end


function h = PltRonkinResidual(ax, data, varargin)
f = inputParser; f.KeepUnmatched = true;
addParameter(f, 'Norm', true, @(x) islogical(x) && isscalar(x));
parse(f, varargin{:});
histArgs = reshape([fieldnames(f.Unmatched), struct2cell(f.Unmatched)].', 1, []);
norm = 'count'; 
Ty = '$N$';
if f.Results.Norm
    norm = 'probability';
    Ty = '$P$';
end
gap = [data.CandidateGap].'; difference = NaN(numel(data),1);
for i = 1:numel(data)
    j = data(i).BestStart;
    if isfinite(j) && j >= 1 && j == fix(j) && j <= numel(data(i).CheckDifference)
        difference(i) = abs(data(i).CheckDifference(j));
    end
end
gap = log10(max(gap(isfinite(gap) & gap >= 0), eps));
difference = log10(max(difference(isfinite(difference)), eps));
hold(ax, 'on');
% Resolve native bin options once on both series, then use identical edges.
h(1) = histogram(ax, [gap; difference], 'Normalization', norm, 'DisplayStyle', 'stairs', ...
    'EdgeColor', [0 47 196]/255, 'LineWidth', 1, histArgs{:});
edges = h(1).BinEdges; h(1).BinEdges = edges; h(1).Data = gap;
h(2) = histogram(ax, difference, 'Normalization', norm, 'DisplayStyle', 'stairs', ...
    'EdgeColor', [0.76 0.34 0.14], 'LineWidth', 1, histArgs{:});
h(2).BinEdges = edges;
box(ax, 'on');
xlabel(ax, '$\log_{10}|\Delta R|$', 'Interpreter', 'Latex', 'FontSize', 10);
ylabel(ax, Ty, 'Interpreter', 'Latex', 'FontSize', 11);
end


function h = Pltfsolve(ax, data, varargin)
f = inputParser; f.KeepUnmatched = true;
addParameter(f, 'PointTol', 1e-8, @(x) validateattributes(x, {'numeric'}, {'scalar','real','nonnegative','nonnan'}));
parse(f, varargin{:});
scatterArgs = reshape([fieldnames(f.Unmatched), struct2cell(f.Unmatched)].', 1, []);
hold(ax, 'on');
set(ax, 'XScale', 'log', 'YScale', 'log', ...
    'XLim', [1e-8 1e-3], 'YLim', [1e-15 1e-3], 'XTick', 10.^(-8:2:-4), 'YTick', 10.^(-15:4:-3));
xlabel(ax, '$\delta_s$', 'Interpreter', 'Latex', 'FontSize', 11); 
ylabel(ax, '$\delta_f$', 'Interpreter', 'Latex', 'FontSize', 11); 
box(ax, 'on');
raw = vertcat(data{:}); h = gobjects(0);
if isempty(raw)
    h(1) = scatter(ax, nan, nan, 5, [0 47 196]/255, ...
        'filled', 'MarkerFaceAlpha', 0.35, scatterArgs{:});
    h(2) = scatter(ax, nan, nan, 8, [0.76 0.34 0.14], ...
        'filled', 'MarkerEdgeAlpha', 0.35, scatterArgs{:});
    return;
end

seed = [raw.ResidualSeed].'; final = [raw.ResidualFinal].';
valid = isfinite(seed) & isfinite(final) & seed > 0 & final > 0;
accepted = [raw.ExitFlag].' > 0 & final < f.Results.PointTol;
plot(ax, [1e-8 1e-3], [1e-8 1e-3], '--', 'Color', [0.6 0.6 0.6]);

h(1) = scatter(ax, seed(valid & accepted), final(valid & accepted), 5, [0 47 196]/255, ...
    'filled', 'MarkerFaceAlpha', 0.35, scatterArgs{:});
h(2) = scatter(ax, seed(valid & ~accepted), final(valid & ~accepted), 8, [0.76 0.34 0.14], ...
    'filled', 'MarkerEdgeAlpha', 0.35, scatterArgs{:});
end
