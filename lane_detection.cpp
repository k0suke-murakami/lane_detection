//
// Created by kosuke on 11/23/17.
//

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <typeinfo>
#include <algorithm>
#include <vector>
#include "polyfit.h"
#include "lane_detection.h"
#include <cmath>
#include <cassert>
#include <chrono>
#include <random>


using namespace cv;
using namespace std;

int preMaxL = 0;
int preMaxR = 0;
vector<double> preCoefL(3, 0), preCoefR(3, 0);

//Mat leftLaneMask(720, 1280, CV_8UC1, Scalar(0));
//Mat rightLaneMask(720, 1280, CV_8UC1, Scalar(0));
//bool isMaskInitialized = false;
//int sum_time = 0;
//preMaxL = preMaxR = 0;
bool isFirst = true;
int minPreL, minPreR;
vector<double> coefPreL, coefPreR;

Mat transformImg(Mat img, bool inverse){
    float width  = img.size().width;
    float height = img.size().height;

    // set src and dst
    const Point2f src[] = {Point2f((width / 2) - 55    , height / 2 + 100),
                           Point2f((width / 6) - 10    , height),
                           Point2f((width * 5 / 6) + 60, height),
                           Point2f((width / 2) + 55    , height / 2 + 100)};
    const Point2f dst[] = {Point2f((width / 4)         , 0),
                           Point2f((width / 4)         , height),
                           Point2f((width * 3 / 4)     , height),
                           Point2f((width * 3 / 4)     , 0)};

    //inverse = false: default
    Mat M;
    if(!inverse)     M = getPerspectiveTransform(src, dst);
    else if(inverse) M = getPerspectiveTransform(dst, src);

    Mat dstImg;
    warpPerspective(img, dstImg, M, img.size(), INTER_LINEAR);

    return dstImg;
}

Mat maskHSV(Mat birdView){
    Mat birdHSV, yellowMask, whiteMask, combinedMask;
    cvtColor(birdView, birdHSV, CV_BGR2HSV);
    int sensitivity = 30;
    inRange(birdHSV, Scalar(20, 100, 100), Scalar(40, 255, 255), yellowMask);
    inRange(birdHSV, Scalar(0, 0, 255-sensitivity), Scalar(255, sensitivity, 255), whiteMask);
    bitwise_or(yellowMask, whiteMask, combinedMask);
    return combinedMask;
}

Mat maskHLS(Mat birdView){
    Mat birdHLS, yellowMask, whiteMask, combinedMask;
    cvtColor(birdView, birdHLS, CV_BGR2HLS);
    int sensitivity = 30;
    inRange(birdHLS, Scalar(15, 100, 30), Scalar(30, 255, 255), yellowMask);
    inRange(birdHLS, Scalar(0, 200, 0), Scalar(255, 255, 255), whiteMask);
    bitwise_or(yellowMask, whiteMask, combinedMask);
    return combinedMask;
}

Mat sobelThresh(Mat birdView, bool isXsobel, int numChannel, int lowThresh, int highThresh){
    double min, max;
    vector<Mat> channels(3);
    Mat birdHLS, sobel, absSobel, normSobel, intSobel, binarySobel;
    cvtColor(birdView, birdHLS, CV_BGR2HLS);
    split(birdHLS, channels);
    if(isXsobel){
        Sobel(channels[numChannel], sobel, CV_64F, 1, 0);
    }
    else{
        Sobel(channels[numChannel], sobel, CV_64F, 0, 1);
    }
    // abs(sobel)
    absdiff(sobel, Scalar::all(0), absSobel);
    minMaxLoc(absSobel, &min, &max);
    // normalize
    normSobel = 255*absSobel/max;
    // convert from CV_64F(double) to CV_32F(int)
    normSobel.convertTo(intSobel, CV_32S);
    minMaxLoc(intSobel, &min, &max);
    inRange(intSobel, Scalar(lowThresh), Scalar(highThresh), binarySobel);
    minMaxLoc(binarySobel, &min, &max);
    return binarySobel;
}

Mat combineSobel(Mat lSobelX, Mat lSobelY, Mat sSobelX, Mat sSobelY ){
    Mat lSobel, sSobel, sobel;
    bitwise_or(lSobelX, lSobelY, lSobel);
    bitwise_or(sSobelX, sSobelY, sSobel);
    bitwise_or(lSobel, sSobel, sobel);
    return sobel;
}

string type2str(int type) {
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

Mat nonMaxSuppression(const Mat src,  const bool remove_plateaus) {
    // find pixels that are equal to the local neighborhood not maximum (including 'plateaus')
    Mat mask;
    dilate(src, mask, cv::Mat());
    compare(src, mask, mask, cv::CMP_GE);
    // optionally filter out pixels that are equal to the local minimum ('plateaus')
    if (remove_plateaus) {
        Mat non_plateau_mask;
        erode(src, non_plateau_mask, cv::Mat());
        compare(src, non_plateau_mask, non_plateau_mask, cv::CMP_GT);
        bitwise_and(mask, non_plateau_mask, mask);
    }
    return mask;
}

Mat makeHist(Mat combined, Point A, Point B){
    Mat combinedSplit = combined(Rect(A, B));
    Mat colSum(1, combined.cols, CV_32SC1);
    reduce(combinedSplit, colSum, 0, CV_REDUCE_SUM, CV_32SC1);
    normalize(colSum, colSum, 0, 256, NORM_MINMAX, -1, Mat() );

    blur(colSum, colSum, Size(25,15));
    colSum.convertTo(colSum, CV_32F);
    return colSum;
}

int findPeak(Mat hist, vector<int> maxCandidates, int& preMax){
    if(maxCandidates.empty()) return preMax;
    vector<int> localMax, maxInds;
//    for(int i=1; i < maxCandidates.size()-1; i++)
    for(int i=0; i < maxCandidates.size(); i++)
    {
//        float pVal = hist.at<float>(maxCandidates[i-1]);
        float val  = hist.at<float>(maxCandidates[i]);
//        float nVal = hist.at<float>(maxCandidates[i+1]);
        // filter peaks
//        if(val > 255*0.2 && val >= pVal && val >= nVal){
        if(val > 255*0.2){
            localMax.push_back(int(val));
            maxInds.push_back(maxCandidates[i]);
        }
    }
    if(maxInds.empty()) return preMax;

    vector<int>::iterator resultMax;
    resultMax = max_element(localMax.begin(), localMax.end());
//    cout <<distance(localMax.begin(), resultMax) << endl;
    int ind = distance(localMax.begin(), resultMax);
    // update pre-max
    preMax = maxInds[ind];
    return maxInds[ind];
}

//for debug
void visualizeHist(Mat hist){
    hist.convertTo(hist, CV_8UC1);
    // visualize hist
    int hist_h = 400;
    Mat histImage( hist_h, hist.cols, CV_8UC3, Scalar( 0,0,0) );
    Mat colSumSmoothed(1, hist.cols, CV_32S);
    blur(hist, hist, Size(19,19));
    for( int i = 1; i < hist.cols; i++ ) {
        line(histImage, Point((i - 1), hist_h - cvRound(hist.at<unsigned char>(i - 1))),
             Point( (i), hist_h - cvRound(hist.at<unsigned char>(i))),
             Scalar(255, 0, 0), 2, 8, 0);
    }


    cv::imshow("image", histImage);
    cv::waitKey();
}

vector<Point> suggestPeaks(Mat hist){
    Mat peaks = nonMaxSuppression(hist,  true);
    vector<Point> maxCandidates;   // output, locations of non-zero pixels
    findNonZero(peaks, maxCandidates);
    return maxCandidates;
}

void divideCandidates(vector<Point> candidates, vector<int>& left, vector<int>& right, int cols){
    for (int i = 0; i < candidates.size(); i++){
        if(candidates[i].x < cols/2){
            left.push_back(candidates[i].x);
        }
        else{
            right.push_back(candidates[i].x);
        }
    }
}


void masking(Mat combined, Mat& leftLane, Mat& rightLane, int leftMaxInd, int rightMaxInd){
    int leftY, rightY, leftMaxI, rightMaxI;
    Mat histI;
    Mat maskRoi( combined.size(), CV_8UC1, Scalar(0));
    vector<Point> peakI;
    for (int i = 0; i < 8; i++){
        leftY = combined.rows*(8-i)/8;
        rightY = combined.rows*(7-i)/8;
        histI = makeHist(combined, Point(0            , leftY),
                         Point(combined.cols, rightY));

        peakI = suggestPeaks(histI);
        vector<int> lPeak, rPeak;
        divideCandidates(peakI, lPeak, rPeak, combined.cols);

//        visualizeHist(histI);
//        cout << histI << endl;

        leftMaxI  = findPeak(histI, lPeak, leftMaxInd);
        rightMaxI = findPeak(histI, rPeak, rightMaxInd);

        if (abs(leftMaxI-leftMaxInd) < 60){
            Rect roiI(leftMaxI-60, leftY-90, 120, 90);
            rectangle(leftLane, roiI, Scalar(255), -1);
            leftMaxInd = leftMaxI;
        }
        else{ // drawing roi with pre-max
            Rect roiI(leftMaxInd-60, leftY-90, 120, 90);
            rectangle(leftLane, roiI, Scalar(255), -1);
        }

        if (abs(rightMaxI-rightMaxInd) < 60){
            Rect roiI(rightMaxI-60, leftY-90, 120, 90);
            rectangle(rightLane, roiI, Scalar(255), -1);
            rightMaxInd = rightMaxI;
        }
        else{ // drawing roi with pre-max
            Rect roiI(rightMaxInd-60, leftY-90, 120, 90);
            rectangle(rightLane, roiI, Scalar(255), -1);
        }
    }
}

void drawPreCoef(Mat img){
    if(isFirst) return;
    for(int y = minPreL; y < img.rows; y++){
        circle(img, Point(int(coefPreL[0]+coefPreL[1]*y+coefPreL[2]*y*y),y) , 10, Scalar(0, 255, 0), 1, LINE_8, 0);
    }

    for(int y = minPreR; y < img.rows; y++) {
        circle(img, Point(int(coefPreR[0] + coefPreR[1] * y + coefPreR[2] * y * y), y), 10, Scalar(0, 255, 0), 1, LINE_8, 0);
    }
}

void drawLanes(Mat birdView, Mat combined, int distance,
               vector<double> coefL, vector<double>coefR, int minL, int minR){
    random_device rnd;
    mt19937 mt(rnd());
    if(minL < minR) uniform_int_distribution<> randN(minR, combined.rows);
    else; uniform_int_distribution<> randN(minL, combined.rows);
    for (int i = 0; i < 8; i++){
        int yR = randN(mt);
        int iDist = abs(int(coefR[0]+coefR[1]*yR+coefR[2]*yR*yR)-int(coefL[0]+coefL[1]*yR+coefL[2]*yR*yR));
        if(abs(distance-iDist) > 100){
            drawPreCoef(birdView);
            return;
        }
    }
    // TODO: more efficient drawing
    for(int y = minL; y < combined.rows; y++){
        circle(birdView, Point(int(coefL[0]+coefL[1]*y+coefL[2]*y*y),y) , 10, Scalar(0, 255, 0), 1, LINE_8, 0);
    }

    for(int y = minR; y < combined.rows; y++) {
        circle(birdView, Point(int(coefR[0] + coefR[1] * y + coefR[2] * y * y), y), 10, Scalar(0, 255, 0), 1, LINE_8, 0);
    }

    minPreL = minL; minPreR = minR;
    coefPreL = coefL; coefPreR = coefR;
    isFirst = false;

}





// check data type
//    string ty =  type2str( combinedMask.type() );
//    printf("Matrix: %s %dx%d \n", ty.c_str(), combined.cols, combined.rows );

Mat getLane(Mat img) {
    auto start = std::chrono::system_clock::now();

//    Mat img = cv::imread("test6.jpg");

    if(img.empty())
    {
        std::cerr << "Failed to open image file." << std::endl;
//        return -1;
    }
    // TODO: calibration and resolve distortion
    Mat birdView = transformImg(img);
    Mat HSVm = maskHSV(birdView);
    Mat HLSm = maskHLS(birdView);
    Mat combined;
    bitwise_or(HSVm, HLSm, combined);

//    combined = combinedM;

    // sobelThresh(Mat img, bool isXsobel, int channel(from HLS space), int low thresh, int high thresh)
//    Mat lSobelX = sobelThresh(birdView, true , 0, 15, 100);
//    Mat lSobelY = sobelThresh(birdView, false, 0, 15, 100);

//    Mat lSobelX = sobelThresh(birdView, true , 1, 50, 225);
//    Mat lSobelY = sobelThresh(birdView, false, 1, 50, 225);
//    Mat sSobelX = sobelThresh(birdView, true , 2, 50, 255);
//    Mat sSobelY = sobelThresh(birdView, false, 2, 50, 255);
//    Mat sobel   = combineSobel(lSobelX, lSobelY, sSobelX, sSobelY);

    // filtered image
//    Mat combined = combinedMask;
//    bitwise_and(sobel, combinedMask, combined);



    Mat hist = makeHist(combined, Point(0,combined.rows), Point(combined.cols, combined.rows/2));


    vector<Point> maxCandidates = suggestPeaks(hist);
    vector<int> lCandidates, rCandidates;
    divideCandidates(maxCandidates, lCandidates, rCandidates, combined.cols);


//    visualizeHist(hist);

    // TODO: put pre-max when tracking instead of 0
    int leftMaxInd  = findPeak(hist, lCandidates, preMaxL);
    int rightMaxInd = findPeak(hist, rCandidates, preMaxR);
    // error if value is pre-max(in this case 0)
    if (leftMaxInd == 0 || rightMaxInd == 0) {
        cout << "ERROR: could not find lane" << endl;
        return img;
    }


//    assert(leftMaxInd  != 0);
//    assert(rightMaxInd != 0);

//    cout << hist.at<float>(leftMaxInd) << endl;
//    cout << hist.at<float>(rightMaxInd) << endl;

    // reusing mask does not make this fast
    Mat leftLaneMask(combined.size(), CV_8UC1, Scalar(0)), rightLaneMask( combined.size(), CV_8UC1, Scalar(0));
    masking(combined, leftLaneMask, rightLaneMask, leftMaxInd, rightMaxInd);


    Mat resultR, resultL;
    combined.copyTo(resultL,leftLaneMask);
    combined.copyTo(resultR,rightLaneMask);
    vector<int> xPl, yPl, xPr, yPr;
    vector<double> coefL(3, 0), coefR(3, 0);

    makePonts(resultL,xPl,yPl);
    makePonts(resultR,xPr,yPr);

    makePolyFit(yPl, xPl, coefL);
    makePolyFit(yPr, xPr, coefR);

    int minL, minR;
    if(yPl.empty()) minL = combined.rows;
    else minL = *min_element(begin(yPl), end(yPl));

    if(yPr.empty()) minR = combined.rows;
    else minR = *min_element(begin(yPr), end(yPr));

    // drawing appropriate lanes
    int distance = rightMaxInd - leftMaxInd;
    drawLanes(birdView, combined, distance, coefL, coefR, minL, minR);

//    // TODO: more efficient drawing
//    for(int y = minL; y < combined.rows; y++){
//        circle(birdView, Point(int(coefL[0]+coefL[1]*y+coefL[2]*y*y),y) , 10, Scalar(0, 255, 0), 1, LINE_8, 0);
//    }
//
//    for(int y = minR; y < combined.rows; y++) {
//        circle(birdView, Point(int(coefR[0] + coefR[1] * y + coefR[2] * y * y), y), 10, Scalar(0, 255, 0), 1, LINE_8, 0);
//    }

    Mat laneImg = transformImg(birdView, true);
    Mat result;
    addWeighted(img, 0.75, laneImg, 0.50, 0, result);

//    cout << "L: "<< coefL[0] << " " << coefL[1] << " " << coefL[2] << " " << "R: " << coefR[0] << " " << coefR[1] << " " <<  coefR[2]<<endl;
//    cout << "L: "<< (coefL[0] - preCoefL[0])*(coefL[0] - preCoefL[0]) << " " << (coefL[1] - preCoefL[1])*(coefL[1] - preCoefL[1])
//         << " " << (coefL[2] - preCoefL[2])*(coefL[2] - preCoefL[2]) << " " << "R: " << (coefR[0] - preCoefR[0])*(coefR[0] - preCoefR[0])
//         << " " << (coefR[1] - preCoefR[1])*(coefR[1] - preCoefR[1]) << " " <<  (coefR[2] - preCoefR[2])*(coefR[2] - preCoefR[2])<<endl;
//    preCoefL = coefL; preCoefR = coefR;
//    double lCurb = (1+pow(pow((2*coefL[2]*double(combined.rows) + coefL[1]),2),1.5))/fabs(2*coefL[2]);
//    double rCurb = (1+pow(pow((2*coefR[2]*double(combined.rows) + coefR[1]),2),1.5))/fabs(2*coefR[2]);
//    cout << "L curvature: " <<lCurb <<" " << "R curvature: "<< rCurb << endl;
    auto end = std::chrono::system_clock::now();

    // Showing process time per frame
    auto diff = end - start;
    cout<< endl << "elapsed time = "
        << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()
        << " msec."
        << std::endl;


//    cv::imshow("image", HSVm);
//    cv::waitKey();
//    cv::imshow("image", HLSm);
//    cv::waitKey();
//    cv::imshow("image", birdView);
//    cv::waitKey();
//    cv::imshow("image", combined);
//    cv::waitKey();

//    cv::imshow("image", leftLane);
//    cv::waitKey();
//    cv::imshow("image", rightLane);
//    cv::waitKey();


    return result;
}
