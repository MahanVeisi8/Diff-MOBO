% %  %% %% plot the paretofront without constraint handling
% clear
% figure; % Create a new figure
% hold on; % Allows multiple plots to be combined into one
% 
% 
for i=1:6
    ParetoFront_all{i} = load(['Z:\Creative_GAN\MONBO_Automized\Dataset\test1\', sprintf('dset_%d.mat', i-1)]);
end
% ParetoFront_all_ = [];
% for i=1:6
%     ParetoFront_all_ = [ParetoFront_all_;ParetoFront_all{i}.dset.Y];
% 
% end
%     plot(ParetoFront_all_(:,1),  ParetoFront_all_(:,1)./ParetoFront_all_(:,2), 'g.','MarkerSize',20)
% 
% axis([0 500 0 120]); % Set the axes limits
% 
% 
% % Preallocate x and y
% x = [];
% y = [];
% 
% % Loop until the figure is closed
% for i=1:10
%     [xi, yi, button] = ginput(1); % Get the point that was clicked
%     if button == 1  % Only accept left-clicks, ignore other buttons
%         x = [x; xi];
%         y = [y; yi];
%         plot(x, y, 'o'); % Plot the current point
%     end
% end

% Your data
x = ParetoFront_all_(:,1);
y = ParetoFront_all_(:,1)./ParetoFront_all_(:,2);

% Create a scatter plot
scatterPlot = scatter(x, y, 'filled');
axis([0 500 0 120]); % Set the axes limits
% Select a point
[xi, yi] = ginput(1);

% Calculate the Euclidean distance between the clicked point and all other points
distances = sqrt((scatterPlot.XData - xi).^2 + (scatterPlot.YData - yi).^2);

% Find the point with the smallest distance to the clicked location
[~, index] = min(distances);

% The coordinates of the closest point in the plot
closestX = scatterPlot.XData(index);
closestY = scatterPlot.YData(index);

% Highlight the selected point in the plot
hold on;
scatter(closestX, closestY, 'r', 'filled');

% Print out the coordinates of the selected point
disp(['The selected point is at X=', num2str(closestX), ' Y=', num2str(closestY)]);
