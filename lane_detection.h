//
// Created by kosuke on 11/23/17.
//

#ifndef LANE_DET_DEMO_LANE_DETECTION_H
#define LANE_DET_DEMO_LANE_DETECTION_H

#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

// helper function
string type2str(int type);
void visualizeHist(Mat hist);

// for image processing
Mat transformImg(Mat img, bool inverse = false);

Mat maskHSV(Mat birdView);

Mat maskHLS(Mat birdView);

Mat sobelThresh(Mat birdView, bool isXsobel, int numChannel, int lowThresh, int highThresh);

Mat combineSobel(Mat lSobelX, Mat lSobelY, Mat sSobelX, Mat sSobelY );

Mat nonMaxSuppression(const Mat src,  const bool remove_plateaus);

Mat makeHist(Mat combined, Point A, Point B);

int findPeak(Mat hist, vector<int> maxCandidates, int& preMax);

vector<Point> suggestPeaks(Mat hist);

void divideCandidates(vector<Point> candidates, vector<int>& left, vector<int>& right, int cols);

void masking(Mat combined, Mat& leftLane, Mat& rightLane, int leftMaxInd, int rightMaxInd);

Mat getLane(Mat img);

#endif //LANE_DET_DEMO_LANE_DETECTION_H
