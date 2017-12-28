//
// Created by kosuke on 11/21/17.
//


#ifndef LANE_DET_DEMO_POLYFIT_H
#define LANE_DET_DEMO_POLYFIT_H

#include <vector>
#include <opencv2/core.hpp>

//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

void makePonts(Mat img, vector<int>& x, vector<int>& y);

void makePolyFit(vector<int> x, vector<int> y, vector<double>& a);


#endif //LANE_DET_DEMO_POLYFIT_H



