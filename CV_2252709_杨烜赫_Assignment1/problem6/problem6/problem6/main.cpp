#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

int main() {
    // Directly specify the image paths
    string imagePath1 = "D:\\CV assignment1\\problem6\\sse1.bmp";
    string imagePath2 = "D:\\CV assignment1\\problem6\\sse2.bmp";

    // Load the images
    Mat img1 = imread(imagePath1, IMREAD_GRAYSCALE);
    Mat img2 = imread(imagePath2, IMREAD_GRAYSCALE);

    // Check if the images are loaded successfully
    if (img1.empty() || img2.empty()) {
        cout << "Could not open or find the images!" << endl;
        return -1;
    }

    // Initialize the ORB detector
    Ptr<ORB> orb = ORB::create();

    // Detect keypoints and compute descriptors
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    orb->detectAndCompute(img1, Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, Mat(), keypoints2, descriptors2);

    // Match the descriptors using BFMatcher (Brute Force Matcher)
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Sort matches based on their distance (best matches first)
    sort(matches.begin(), matches.end());

    // Draw the matches
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);

    // Display the result
    namedWindow("ORB Feature Matching", WINDOW_NORMAL);
    imshow("ORB Feature Matching", img_matches);
    waitKey(0);

    // Save the output image
    imwrite("orb_feature_matching_result.jpg", img_matches);

    return 0;
}
