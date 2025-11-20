% Given sample points
points = [
    -2, 0;
     0, 0.9;
     2, 2.0;
     3, 6.5;
     4, 2.9;
     5, 8.8;
     6, 3.95;
     8, 5.03;
    10, 5.97;
    12, 7.1;
    13, 1.2;
    14, 8.2;
    16, 8.5;
    18, 10.1
];

% Separate points into X and Y coordinates
X = points(:, 1);
Y = points(:, 2);

% Parameters for RANSAC
maxIter = 1000; % Number of iterations
threshold = 1;  % Distance threshold to consider a point as inlier
minInliers = 0; % Keep track of the best model's inlier count
bestLine = [0, 0]; % Best line parameters [slope, intercept]

% RANSAC algorithm
for i = 1:maxIter
    % Randomly select 2 points to define a line
    idx = randperm(length(X), 2);
    p1 = points(idx(1), :);
    p2 = points(idx(2), :);
    
    % Ensure the points are not identical
    if p1 == p2
        continue;
    end
    
    % Define the line: y = mx + b
    slope = (p2(2) - p1(2)) / (p2(1) - p1(1));
    intercept = p1(2) - slope * p1(1);
    
    % Calculate the distances of all points to the line
    distances = abs(Y - (slope * X + intercept)) ./ sqrt(slope^2 + 1);
    
    % Determine inliers based on the threshold
    inliers = find(distances < threshold);
    
    % Check if this model has more inliers than the previous best
    if length(inliers) > minInliers
        minInliers = length(inliers);
        bestLine = [slope, intercept];
    end
end

% Display the results
disp(['Best line: y = ' num2str(bestLine(1)) 'x + ' num2str(bestLine(2))]);

% Plot the points and the best fit line
figure;
hold on;
scatter(X, Y, 'bo', 'DisplayName', 'Data Points'); % Plot the data points
plot(X, bestLine(1) * X + bestLine(2), 'r-', 'LineWidth', 2, 'DisplayName', 'RANSAC Line'); % Plot the RANSAC best fit line
legend('Location', 'best');
xlabel('X');
ylabel('Y');
title('RANSAC Line Fitting');
grid on;
hold off;
