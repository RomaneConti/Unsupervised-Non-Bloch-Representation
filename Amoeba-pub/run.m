
RootFolder = fileparts(mfilename('fullpath'));
cd(RootFolder);
addpath(RootFolder);

CoupleMat = {[0 3 0; 1 0 2; 0 4 0],...
    [0 1i 0 0; 2 0 1 1.25; 0 1i 0 0],...
    [0 2 0 0; 1 0 1.2 -0.25; 0 1 0 0; 0 1.25 0 0]};

CoupleCenter = {[2 2], [2 2], [2 2]};

obj1 = MonoArray(CoupleMat{1}, CoupleCenter{1});
obj2 = MonoArray(CoupleMat{2}, CoupleCenter{2});
obj3 = MonoArray(CoupleMat{3}, CoupleCenter{3});

objs = {obj1, obj2, obj3};
Tags = {'Model1_Round50', 'Model2_Round50', 'Model3_Round50'};
NyNxPair = {[100, 100], [100, 100], [100, 100]};
DropFun = {@(x, y) (x-50).^2 + (y-50).^2 >= 50.^2,...
    @(x, y) (x-50).^2 + (y-50).^2 >= 50.^2,...
    @(x, y) (x-50).^2 + (y-50).^2 >= 50.^2
};


for id = 1 : length(Tags)
    NyNx = NyNxPair{id};
    obj = objs{id};
    [~, pos] = obj.HamObc(NyNx(1), NyNx(2), 'fDrop', DropFun{id});
    figure; 
    scatter(pos.Xpos, pos.Ypos, 2); 
    title(Tags{id});
end


for id = 1 : length(Tags)
    Tag = Tags{id};
    obj = objs{id};
    NyNx = NyNxPair{id};
    RDATA_EIGSYS(obj, NyNx(1), NyNx(2), 'DataTag', Tag, 'fDrop', DropFun{id});
end

diary('./Revision2_Data/Round50_run.log');

t0 = tic;
for m = 1:3
    tag = sprintf('Model%d_Round50', m);
    fprintf('\n========== %s ==========\n', tag);
    d = load(['./Revision2_Data/EigSys_' tag '.mat'], 'EigVal', 'CoupleMat', 'CoupleCenter');
    obj = MonoArray(d.CoupleMat, d.CoupleCenter);

    RDATA_RONKIN_MIN(obj, d.EigVal(:), 'DataTag', tag, 'ResetData', false,...
        'UseJensen', true, 'NIntRes', 1024, 'NCheckRes', 2048,...
        'NumStarts', 3, 'tol', 1e-5, 'NumSegments', 40);

    source = ['./Revision2_Data/RonkinMin_' tag '.mat'];
    RDATA_AMOEBA(source, 'DataTag', tag, 'ResetData', false,...
        'UseFSolve', true, 'NContourMesh', 501, 'NumSegments', 40);
end

fprintf('\nAll finished. Total time: %.2f hours\n', toc(t0)/3600);
diary off;
