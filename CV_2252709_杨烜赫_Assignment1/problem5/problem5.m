% Read the two input images
I1 = imread('sse1.bmp');
I2 = imread('sse2.bmp');

% Convert images to grayscale (SIFT works on grayscale images)
grayI1 = rgb2gray(I1);
grayI2 = rgb2gray(I2);

% Detect keypoints using DoG-based interest point detector (SURF as an approximation)
points1 = detectSURFFeatures(grayI1); % Similar to DoG (You can also use detectSIFTFeatures)
points2 = detectSURFFeatures(grayI2); 

% Select the strongest 100 points for each image
strongestPoints1 = points1.selectStrongest(100);
strongestPoints2 = points2.selectStrongest(100);

% Display corner detection results for Image 1 with hollow circles
figure;
imshow(I1);
hold on;
% Plot the strongest points manually using their locations
plot(strongestPoints1.Location(:,1), strongestPoints1.Location(:,2), 'ro', 'MarkerSize', 5, 'LineWidth', 1); 
title('Corner Detection for Image 1 with Hollow Circles');

% Display corner detection results for Image 2 with hollow circles
figure;
imshow(I2);
hold on;
% Plot the strongest points manually using their locations
plot(strongestPoints2.Location(:,1), strongestPoints2.Location(:,2), 'ro', 'MarkerSize', 5, 'LineWidth', 1); 
title('Corner Detection for Image 2 with Hollow Circles');

% Extract descriptors at the detected feature points
[features1, validPoints1] = extractFeatures(grayI1, strongestPoints1);
[features2, validPoints2] = extractFeatures(grayI2, strongestPoints2);

% Match features between the two images
indexPairs = matchFeatures(features1, features2);

% Retrieve matched points
matchedPoints1 = validPoints1(indexPairs(:, 1));
matchedPoints2 = validPoints2(indexPairs(:, 2));

% Display matching points between the two images in a larger figure
figure;
showMatchedFeatures(I1, I2, matchedPoints1, matchedPoints2, 'montage');
title('Matching Result between Image 1 and Image 2');

% Estimate the geometric transformation between matched points
[tform, inlierPoints2, inlierPoints1] = estimateGeometricTransform(matchedPoints2, matchedPoints1, 'projective');

% Warp image 2 to align with image 1
outputView = imref2d(size(I1));
I2Warped = imwarp(I2, tform, 'OutputView', outputView);

% Create a panorama by blending the two images
panorama = max(I1, I2Warped);

% Display the panorama result in a larger figure
figure;
imshow(panorama);
title('Panorama Stitched Using SIFT and SURF Feature Matching');
