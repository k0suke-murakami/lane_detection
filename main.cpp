#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "lane_detection.h"

using namespace cv;
using namespace std;

int main(int argc, const char* argv[])
{
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    VideoCapture cap("../challenge_video.mp4");

    // Check if camera opened successfully
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }
    // Default resolution of the frame is obtained.The default resolution is system dependent.
    int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

    // Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
    VideoWriter video("out.mp4",CV_FOURCC('M','J','P','G'),30, Size(frame_width,frame_height));

    while(1){
        Mat frame;
        // Capture frame-by-frame
        cap >> frame;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        Mat result = getLane(frame);

        // Write the frame into the file 'outcpp.avi'
        video.write(result);
        // Display the resulting frame
        imshow( "Frame", result );

        // Press  ESC on keyboard to exit
        char c=(char)waitKey(1);
        if(c==27)
            break;
        else if(c==32)
            continue;
    }

    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    destroyAllWindows();
    return 0;
}