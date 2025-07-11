clear all
figure
%% 
 %% %% plot the paretofront without constraint handling
for i=1:10
    ParetoFront_all{i} = load(['Z:\Creative_GAN\MONBO_Automized\Dataset\GAN1\', sprintf('dset_%d.mat', i-1)]);
end
ParetoFront_all_ = [];
for i=1:10
    ParetoFront_all_ = [ParetoFront_all_;ParetoFront_all{i}.dset.Y];

end
    plot(ParetoFront_all_(:,1),  ParetoFront_all_(:,1)./ParetoFront_all_(:,2), 'g.','MarkerSize',20)
    hold on
%% plot the paretofront with constraint handling
for i=1:10
    ParetoFront_all{i} = load(['Z:\Creative_GAN\MONBO_Automized\Dataset\GAN2\', sprintf('dset_%d.mat', i-1)]);
end


ParetoFront_all_ = [];
for i=1:10
    ParetoFront_all_ = [ParetoFront_all_;ParetoFront_all{i}.dset.Y];
end
    plot(ParetoFront_all_(:,1),  ParetoFront_all_(:,1)./ParetoFront_all_(:,2), 'b.','MarkerSize',20)
    hold on



 %% %% plot the paretofront without constraint handling
for i=1:10
    ParetoFront_all{i} = load(['Z:\Creative_GAN\MONBO_Automized\Dataset\GAN3\', sprintf('dset_%d.mat', i-1)]);
end
ParetoFront_all_ = [];
for i=1:10
    ParetoFront_all_ = [ParetoFront_all_;ParetoFront_all{i}.dset.Y];
end
    plot(ParetoFront_all_(:,1),  ParetoFront_all_(:,1)./ParetoFront_all_(:,2), 'y.','MarkerSize',20)
    hold on
%%

for i=1:10
    ParetoFront_all{i} = load(['Z:\Creative_GAN\MONBO_Automized\Dataset\GAN4\', sprintf('dset_%d.mat', i-1)]);
end
ParetoFront_all_ = [];
for i=1:10
    ParetoFront_all_ = [ParetoFront_all_;ParetoFront_all{i}.dset.Y];
end

    plot(ParetoFront_all_(:,1),  ParetoFront_all_(:,1)./ParetoFront_all_(:,2), 'm.','MarkerSize',20)
        hold on
        %% GAN 5

for i=1:10
    ParetoFront_all{i} = load(['Z:\Creative_GAN\MONBO_Automized\Dataset\GAN5\', sprintf('dset_%d.mat', i-1)]);
end
ParetoFront_all_ = [];
for i=1:10
    ParetoFront_all_ = [ParetoFront_all_;ParetoFront_all{i}.dset.Y];
end

    plot(ParetoFront_all_(:,1),  ParetoFront_all_(:,1)./ParetoFront_all_(:,2), 'c.','MarkerSize',20)
        hold on

 %% %% original data

load('Z:\Creative_GAN\MO-PaDGAN\airfoil\dataset_openfoam\cl_cd_train.mat');

    plot(cl_cd(:,1),  cl_cd(:,1)./cl_cd(:,2), 'k.','MarkerSize',20)




% legend({'GAN1','GAN2',  'GAN3','GAN4', 'Original'})