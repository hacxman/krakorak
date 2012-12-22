#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;

int main(void) {
  namedWindow("a", CV_WINDOW_AUTOSIZE);
  namedWindow("b", CV_WINDOW_AUTOSIZE);
  VideoCapture cap(0);
  Mat img;
//  vector<Mat> avg(10);
  Mat avg;
  cap >> avg;
  for (;;) {
    cap >> img;
    if (img.empty()) {exit(1);}
    imshow("a", img);

    accumulate(img, avg);
    imshow("b", avg);

    int c = cvWaitKey(1);
    if (c == 27) break;
  }
}

