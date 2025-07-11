% % % plot the paretofront with constraint handling
% % for i=1:2
% %     ParetoFront_all{i} = load(['Z:\Creative_GAN\MONBO_Automized\Dataset\', sprintf('dset_%d.mat', i-1)]);
% % end
% % figure 
% % ParetoFront_all_ = [];
% % for i=1:2
% %     ParetoFront_all_ = [ParetoFront_all_;ParetoFront_all{i}.dset.Y];
% % 
% % end
% %     plot(ParetoFront_all_(:,1),  ParetoFront_all_(:,1)./ParetoFront_all_(:,2), '.','MarkerSize',20)
% %     hold on
% % legend(legend_all)
% % title('With constraint handler')
% % xlim([-0.1 0.9])
% % ylim([-5 60])

 %% %% plot the paretofront constraint handling
for i=1:6
    ParetoFront_all{i} = load(['Z:\Creative_GAN\MONBO_Automized\Dataset\', sprintf('dset_%d.mat', i-1)]);
end
figure 
legend_all=[];
for i=1:6
    plot(ParetoFront_all{i}.dset.Y(:,1),  ParetoFront_all{i}.dset.Y(:,1)./ParetoFront_all{i}.dset.Y(:,2), '.')
    hold on
    legend_all{i} = sprintf('iter%d', i);
end
legend(legend_all)
title('Without constraint handler')
axis([0 1 0 100]); % Set the axes limits