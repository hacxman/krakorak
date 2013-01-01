#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

#include <list>

#include "UEyeOpenCVException.hpp"
#include "UEyeOpenCV.hpp"

using namespace std;
using namespace cv;

const int FLT_SIZE = 10;

Scalar clrs[] = {Scalar(255,  0,  0),
                 Scalar(  0,  255,  0),
                 Scalar(  0,  0,  255),
                 Scalar(255,  0,255),
                 Scalar(255,255,  0),
                 Scalar(  0,255,255)};

int main(void) {
  FileStorage config("cfg.yml", FileStorage::READ);
  String driver;
  config["driver"] >> driver;

  namedWindow("a", CV_WINDOW_AUTOSIZE);
  namedWindow("b", CV_WINDOW_AUTOSIZE);
  VideoCapture *capNative;
  UeyeOpencvCam *capUeye;
  if (driver == "native") {
    capNative = new VideoCapture(0);
  } else if (driver == "file") {
    String fname;
    config["filename"] >> fname;
    capNative = new VideoCapture(fname); //0);
  } else if (driver == "ueye") {
    capUeye = new UeyeOpencvCam(640, 480);
    cout << capUeye->getHIDS() << endl;
  } else {
    cerr << "invalid driver: '" << driver << "'" << endl;
    cerr << "posible are: native, file, ueye" << endl;
    exit(2);
  }

  Mat img;
  list<Mat> avgs;
  list<Mat>::iterator it;
  Mat _img;
  for (int i = 0; i < FLT_SIZE; i++) {
    if (driver == "native" || driver == "file") {
      *capNative >> _img;
    } else if (driver == "ueye") {
      _img = capUeye->getFrame();
    }

    Rect myROI(10, 10, _img.cols-40, _img.rows-40);
    cv::Mat croppedImage = _img(myROI);
    _img = croppedImage;


    cvtColor(_img, img, CV_32F);
    _img *= 1./255;
    cvtColor(_img, img, CV_RGB2GRAY);
    avgs.push_back(img);
  }
  cout << "avgs.len = " << avgs.size() << endl;
  for (int frame_id = 0;;frame_id++) {
    if (driver == "native" || driver == "file") {
      *capNative >> img;
      cerr << frame_id << endl;
    } else if (driver == "ueye") {
      img = capUeye->getFrame();
    }

    Rect myROI(10, 10, img.cols-40, img.rows-40);
    cv::Mat croppedImage = img(myROI);
    img = croppedImage;

    if (img.empty()) {break;}
    imshow("a", img);

    img.convertTo(_img, CV_32F);
    _img *= 1./255;
    cvtColor(_img, img, CV_RGB2GRAY);

    avgs.pop_back();
    avgs.push_front(img.clone());

    it = avgs.begin();
    Mat avg = (*it).clone();
    avg /= (float)FLT_SIZE;
    for (it++; it != avgs.end(); it++) {
//      accumulateWeighted(*it, avg, 1/20.0);
      accumulate((*it) / FLT_SIZE, avg);
    }
    Mat o = img.clone();
    o = img - avg;
    Scalar m = mean(img);
    cout << m[1] << endl;
    vector<vector<Point>> cont;
    threshold(img - avg, o, 0.07, 1, THRESH_BINARY);
    //o *= 255;
    //o.convertTo(o, CV_8U);
    //adaptiveThreshold(o, o, 25, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 65, 2.4);
//    imshow("b", o);
    o.convertTo(o, CV_8U);
    findContours(o, cont, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
//    adaptiveThreshold(img - avg, o, 1, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 4, 0.1);
    Mat a; //= img.clone();
//    img.convertTo(a, CV_GRAY2RGB);
    a = Mat::zeros(img.rows, img.cols, CV_8UC3);
//    printf("%d\n", a.channels());
    for (int i = 0; i < cont.size(); i++) {
      drawContours(a, cont, i, clrs[i%6], CV_FILLED);
    };
//    drawContours(a, cont, -1, Scalar(255.0,1.0,0.0));
    imshow("b", a);

    int c = cvWaitKey(1);
    if (c == 27) break;
  }
}

