//******************************************************************************
//
//                 Low Cost Vision
//
//******************************************************************************
// Project:        ueyeOpencv
// File:           UEyeOpenCV.hpp
// Description:    Wrapper class of UEye camera to support OpenCV Mat using the UEye SDK
// Author:         Wouter Langerak
// Notes:          For more functionalities use the SDK of UEye, the purpose of this project is to make it compatible with OpenCV Mat.
//
// License:        GNU GPL v3
//
// This file is part of ueyeOpencv.
//
// ueyeOpencv is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// ueyeOpencv is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ueyeOpencv.  If not, see <http://www.gnu.org/licenses/>.
//******************************************************************************

#pragma once
#include <ueye.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <UEyeOpenCVException.hpp>
/**
 * Wrapper class of UEye camera to support OpenCV Mat using the UEye SDK
 */
class UeyeOpencvCam {
public:
/**
 * Constructor creates camera interface of UEye cam
 * @param wdth : width of the image taken by the camera
 * @param heigh : height of the image taken by the camera
 */
        UeyeOpencvCam(int wdth, int heigh);
        /**
         * Returns the camera id
         * @return camera id
         */
        HIDS getHIDS();
        /**
         * Deconstructor
         */
        ~UeyeOpencvCam();
        /**
         * Returns an OpenCV Mat of the current view of the camera.
         * @return OpenCV Mat
         */
        cv::Mat getFrame();
        /**
         * Paste an OpenCV Mat of the current view of the camera in the @param mat.
         */
        void getFrame(cv::Mat& mat);
        /**
         * Toggle auto white balance.
         * @param set
         */
        void setAutoWhiteBalance(bool set=true);
        /**
         * Toggle auto gain.
         * @param set
         */
        void setAutoGain(bool set=true);
private:
        HIDS hCam;
        cv::Mat mattie;
        int width;
        int height;
};
