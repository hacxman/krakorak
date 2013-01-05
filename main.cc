#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion
#include <chrono>

#include <algorithm>

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/flann/flann.hpp>  // OpenCV window I/O
#include <opencv2/gpu/gpu.hpp>

#include <list>
#include <unistd.h>

#define GUI

//#include "UEyeOpenCVException.hpp"
//#include "UEyeOpenCV.hpp"

using namespace std;
using namespace std::chrono;
using namespace cv;

int FLT_SIZE = 10;

Scalar clrs[] = {Scalar(255,  0,  0),
                 Scalar(  0,  255,  0),
                 Scalar(  0,  0,  255),
                 Scalar(255,  0,255),
                 Scalar(255,255,  0),
                 Scalar(  0,255,255)};


void init_cuda() {
  cout << "CUDA enabled devices: " << gpu::getCudaEnabledDeviceCount() << endl;
}

// indices must be of indices.size()
inline void nearestN(vector<Point2f> &centers, vector<Point2f> &oldCenters, vector<int> &indices) {
      int c_size = centers.size();
      int oc_size = oldCenters.size();
//      vector<int> indices(c_size);
      { // find nearest neighbors, in O(n^2)
        vector<float> mins(c_size);
#pragma omp parallel for
        for (int i = 0; i < c_size; i++) {
          for (int j = 0; j < oc_size; j++) {
            Point p1 = centers[i];
            Point p2 = oldCenters[j];
            float len = norm(p1 - p2);
            if (mins[i] == 0 || mins[i] > len) {
              mins[i] = len;
              indices[i] = j;
            }
          }
        }
      }
}


int th_trackbar = 20;
int mw_trackbar = 10;

void init_gui() {
  namedWindow("a", CV_WINDOW_AUTOSIZE);
  namedWindow("b", CV_WINDOW_AUTOSIZE);
  createTrackbar("th", "a", &th_trackbar, 100);
  createTrackbar("mw", "a", &mw_trackbar, 100);
}

struct Capt {
  VideoCapture *capNative;
//  UeyeOpencvCam *capUeye;

  string driver;

  Capt(const FileStorage &config) {
//    String driver;
    config["driver"] >> driver;

    if (driver == "native") {
      capNative = new VideoCapture(0);
    } else if (driver == "file") {
     String fname;
     config["filename"] >> fname;
     capNative = new VideoCapture(fname); //0);
  //  } else if (driver == "ueye") {
  //    capUeye = new UeyeOpencvCam(640, 480);
  //    cout << capUeye->getHIDS() << endl;
    } else {
      cerr << "invalid driver: '" << driver << "'" << endl;
  //    cerr << "posible are: native, file, ueye" << endl;
      exit(2);
    }
  }

  void operator>>(Mat &m) {
    if (driver == "native" || driver == "file") {
      *capNative >> m;
//    } else if (driver == "ueye") {
//      m = capUeye->getFrame();
    }

  }

};

int main(void) {

  init_cuda();

  FileStorage config("cfg.yml", FileStorage::READ);
  Capt cap(config);

  init_gui();

  Mat img;
  list<Mat> avgs;
  list<Mat>::iterator it;
  vector<Point2f> oldCenters;
  Mat _img;
  for (int i = 0; i < FLT_SIZE; i++) {
    cap >> _img;

    //Rect myROI(0, 0, _img.cols, _img.rows);
    Rect myROI(10, 10, _img.cols-40, _img.rows-40);
    cv::Mat croppedImage = _img(myROI);
    _img = croppedImage;


    cvtColor(_img, img, CV_32F);
    _img *= 1./255;
    cvtColor(_img, img, CV_RGB2GRAY);
    avgs.push_back(img);
  }
  cout << "avgs.len = " << avgs.size() << endl;

  high_resolution_clock _CLOCK;
  high_resolution_clock::time_point _LAST(_CLOCK.now());
  for (int frame_id = 0;;frame_id++) {
    cap >> img;
    FLT_SIZE = mw_trackbar;

    Rect myROI(10, 10, img.cols-40, img.rows-40);
    //Rect myROI(0, 0, _img.cols, _img.rows);
    cv::Mat croppedImage = img(myROI);
    img = croppedImage;

    Mat origImg = img.clone();
    {
      high_resolution_clock::time_point now = _CLOCK.now();
      duration<float> dur = duration_cast<duration<float>>(now - _LAST);
      _LAST = now;
      char lala[32]; sprintf(lala, "%d %f %f", frame_id, dur.count(), 1/dur.count());
      cerr << "\r" << lala;
      putText(origImg, lala, Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0, 255));
    }

    if (img.empty()) {break;}
#ifdef GUI
    //imshow("a", img);
#endif

    img.convertTo(_img, CV_32F);
    _img *= 1./255;
    cvtColor(_img, img, CV_RGB2GRAY);

    while (avgs.size() > mw_trackbar) {
      avgs.pop_back();
    }
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
    cout << m[0] << endl;
    vector<vector<Point>> cont;
    threshold(img - avg, o, m[0]/(th_trackbar/10.0), 1, THRESH_BINARY);

    o.convertTo(o, CV_8U);
    findContours(o, cont, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    Mat a = Mat::zeros(img.rows, img.cols, CV_8UC3);
    vector<Point2f> centers;
    vector<float> radii;
    for (int i = 0; i < cont.size(); i++) {
      cout << "Cont " << i << endl;
      Moments mmts = moments(cont[i], true);
      Point2f enCirc; float radius;
      minEnclosingCircle(cont[i], enCirc, radius);
    //  circle(origImg, enCirc, radius, Scalar(255,255,255,255));
      if (mmts.m00 == 0) continue;
      double _x = mmts.m10/mmts.m00;
      double _y = mmts.m01/mmts.m00;
      cout << "    x,y = " << _x << " " << _y << " " << radius << endl;
      drawContours(a, cont, i, clrs[i%6], CV_FILLED);
    //  drawContours(origImg, cont, i, clrs[i%6]);
      line(a, Point(_x-5, _y), Point(_x+5, _y), Scalar(255, 255, 255, 35));
      line(a, Point(_x, _y-5), Point(_x, _y+5), Scalar(255, 255, 255, 35));
      centers.push_back(Point(_x, _y));
      radii.push_back(radius);
    };
    if (!oldCenters.empty()) {
      int c_size = centers.size();
      int oc_size = oldCenters.size();
      vector<int> indices(c_size);
      nearestN(centers, oldCenters, indices);
#pragma omp parallel for
      for (int i = 0; i < c_size; i++) {
          int idx = indices[i];
          cout << "idx: " << idx << endl;
          Point p2 = centers[i];
          Point p1 = oldCenters[idx];
          float dst = norm(p2 - p1);
          {
              char lala[32]; sprintf(lala, "r: %.2f, d: %.2f", radii[i], dst);
              putText(origImg, lala, p2+Point(10,10), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 255, 0, 255));
          }
          if (dst < radii[i]*10.0 && dst > 1.0) {
//          if (dst < 20.0) {
            cout << "KUUUUUUUUUUUUUUUUUUUUUURWAAAAAAAAAAA" << endl;
            cout << "dst: " << dst << " idx: " << idx << endl;
            cout << "     p1 " << p1.x << " " << p1.y << endl;
            cout << "     p2 " << p2.x << " " << p2.y << endl;
            line(a, p1+Point(-10, -10), p1+Point(10, 10), Scalar(255, 255, 0, 55));
            line(a, p1+Point(10, -10), p1+Point(-10, 10), Scalar(255, 255, 0, 55));
            line(a, p2+Point(-10, -10), p2+Point(10, 10), Scalar(255, 0, 255, 128));
            line(a, p2+Point(10, -10), p2+Point(-10, 10), Scalar(255, 0, 255, 128));
            line(origImg, p1, p2, Scalar(255, 255, 255, 0));
            line(origImg, p2, p2 + (p2 - p1), Scalar(0, 255, 0, 0));
          }

//        }
      }

    }
//    } else { //cout << "             PICAAAAAAAAAA"
//        << centers.cols << " "
//        << centers.rows << endl; };
    {
      char lala[32]; sprintf(lala, "th %f %f cont %d", m[0], m[0]/(th_trackbar/10.0), cont.size());
      putText(origImg, lala, Point(10, 40), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 255, 0, 255));
    }

    cerr << " " << th_trackbar << " " << cont.size() << "     "; cerr.flush();

    oldCenters = centers; //centers.clone();
//    drawContours(a, cont, -1, Scalar(255.0,1.0,0.0));
#ifdef GUI
    imshow("a", origImg);

    imshow("b", a);

    int c = waitKey(1);//frame_id > 150 && frame_id < 250 ? 0: 2);
//    cout << "#####  " << (0xff & c) << "  #####" << endl;
    if (c == 27) break;
    if ((0xff & c) == ' ') usleep(20000);
#endif
  }
}
