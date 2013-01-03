#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/flann/flann.hpp>  // OpenCV window I/O

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
  Mat oldCenters;
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
    threshold(img - avg, o, 0.1, 1, THRESH_BINARY);
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
    Mat centers;
    //vector<Point2f> cs;
    for (int i = 0; i < cont.size(); i++) {
      cout << "Cont " << i << endl;
      Moments mmts = moments(cont[i], true);
      Point2f enCirc; float radius;
      minEnclosingCircle(cont[i], enCirc, radius);
      if (mmts.m00 == 0) continue;
      double _x = mmts.m10/mmts.m00;
      double _y = mmts.m01/mmts.m00;
      cout << "    x,y = " << _x << " " << _y << " " << radius << endl;
      drawContours(a, cont, i, clrs[i%6], CV_FILLED);
      line(a, Point(_x-10, _y), Point(_x+10, _y), Scalar(255, 255, 255, 255));
      line(a, Point(_x, _y-10), Point(_x, _y+10), Scalar(255, 255, 255, 255));
      centers.push_back(Mat(Matx21f(_x, _y)));
      //cs.push_back(Point(_x, _y));
//      cout << centers.cols << " "
//        << centers.rows << endl;

    };
    //centers = Mat(cs);
    if (!oldCenters.empty()) {
      //cout << "KOKOOOOOOOOOOOOOOOOOOOOOOOOOT "
      //  << oldCenters.cols << " "
      //  << oldCenters.rows << endl;
      cv::flann::KDTreeIndexParams parms(16);
//      oldCenters.convertTo(oldCenters, CV_32F);

      cv::flann::Index index(oldCenters, parms);

//      vector<int> indices;
//      vector<float> dists;
      Mat indices = Mat(centers.rows, 2, CV_32SC1);
      Mat dists = Mat(centers.rows, 2, CV_32FC1);

      centers.convertTo(centers, CV_32F);
//      vector<Point2f> query; query.push_back(Point(oldCenters.col(0)));
      index.knnSearch(centers, indices, dists, 2, cv::flann::SearchParams());
      for (int row = 0; row < indices.rows; row++) {
        for (int col = 0; col < indices.cols; col++) {
//          int idx = indices[row]; //indices.at<int>(row, col);
//          float dst = dists[row]; //dists.at<int>(row, col);
          int idx = indices.at<int>(row, col);
          float dst = dists.at<float>(row, col);
          if (dst < 20.0) {
            cout << "KUUUUUUUUUUUUUUUUUUUUUURWAAAAAAAAAAA" << endl;
            cout << dst << " " << idx << endl;
            Point p1(oldCenters.at<float>(idx, 0),oldCenters.at<float>(idx, 1));
            Point p2(centers.at<float>(row, 0),centers.at<float>(row, 1));
            cout << "p1 " << p1.x << " " << p1.y << endl;
            line(a, p1+Point(-10, -10), p1+Point(10, 10), Scalar(255, 255, 0, 255));
            line(a, p1+Point(10, -10), p1+Point(-10, 10), Scalar(255, 255, 0, 255));
            //line(a, p2+Point(-10, -10), p2+Point(10, 10), Scalar(255, 0, 255, 128));
            //line(a, p2+Point(10, -10), p2+Point(-10, 10), Scalar(255, 0, 255, 128));
      //      line(a, p1, p2, Scalar(255, 255, 255, 0));
          }

        }
      }

    }
//    } else { //cout << "             PICAAAAAAAAAA"
//        << centers.cols << " "
//        << centers.rows << endl; };


    oldCenters = centers.clone();
//    drawContours(a, cont, -1, Scalar(255.0,1.0,0.0));
    imshow("b", a);

    int c = cvWaitKey(50);
    if (c == 27) break;
  }
}
