//******************************************************************************
//
//                 Low Cost Vision
//
//******************************************************************************
// Project:        ueyeOpencv
// File:           UEyeOpenCV.cpp
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

#include <UEyeOpenCV.hpp>
#include <iostream>
#include <ueye.h>

UeyeOpencvCam::UeyeOpencvCam(int wdth, int heigh) {
	width = wdth;
	height = heigh;
	using std::cout;
	using std::endl;
	mattie = cv::Mat(height, width, CV_8UC3);
	hCam = 0;
	char* ppcImgMem;
	int pid;
	INT nAOISupported = 0;
	double on = 1;
	double empty;
	int retInt = IS_SUCCESS;
	retInt = is_InitCamera(&hCam, 0);
	if (retInt != IS_SUCCESS) {
		throw UeyeOpenCVException(hCam, retInt);
	}
	retInt = is_SetColorMode(hCam, IS_CM_BGR8_PACKED);
	if (retInt != IS_SUCCESS) {
		throw UeyeOpenCVException(hCam, retInt);
	}
	retInt = is_ImageFormat(hCam, IMGFRMT_CMD_GET_ARBITRARY_AOI_SUPPORTED, (void*) &nAOISupported, sizeof(nAOISupported));
	if (retInt != IS_SUCCESS) {
		throw UeyeOpenCVException(hCam, retInt);
	}
	retInt = is_AllocImageMem(hCam, width, height, 24, &ppcImgMem, &pid);
	if (retInt != IS_SUCCESS) {
		throw UeyeOpenCVException(hCam, retInt);
	}
	retInt = is_SetImageMem(hCam, ppcImgMem, pid);
	if (retInt != IS_SUCCESS) {
		throw UeyeOpenCVException(hCam, retInt);
	}
	//set auto settings
	retInt = is_SetAutoParameter(hCam, IS_SET_ENABLE_AUTO_WHITEBALANCE, &on, &empty);
	if (retInt != IS_SUCCESS) {
		throw UeyeOpenCVException(hCam, retInt);
	}
	retInt = is_SetAutoParameter(hCam, IS_SET_ENABLE_AUTO_GAIN, &on, &empty);
	if (retInt != IS_SUCCESS) {
		throw UeyeOpenCVException(hCam, retInt);
	}

  retInt = is_SetBinning(hCam, IS_BINNING_4X_VERTICAL | IS_BINNING_4X_HORIZONTAL);
	if (retInt != IS_SUCCESS) {
		throw UeyeOpenCVException(hCam, retInt);
	}


	retInt = is_CaptureVideo(hCam, IS_WAIT);
	if (retInt != IS_SUCCESS) {
		throw UeyeOpenCVException(hCam, retInt);
	}
}

UeyeOpencvCam::~UeyeOpencvCam() {
	int retInt = is_ExitCamera(hCam);
	if (retInt != IS_SUCCESS) {
		throw UeyeOpenCVException(hCam, retInt);
	}
}

cv::Mat UeyeOpencvCam::getFrame() {
	getFrame(mattie);
	return mattie;
}

void UeyeOpencvCam::getFrame(cv::Mat& mat) {
	VOID* pMem;
	int retInt = is_GetImageMem(hCam, &pMem);
	if (retInt != IS_SUCCESS) {
		throw UeyeOpenCVException(hCam, retInt);
	}
//	if (mat.cols == width && mat.rows == height && mat.depth() == 3) {
		memcpy(mat.ptr(), pMem, width * height * 3);
//	} else {
//		throw UeyeOpenCVException(hCam, -1337);
//	}
}

HIDS UeyeOpencvCam::getHIDS() {
	return hCam;
}

void UeyeOpencvCam::setAutoWhiteBalance(bool set) {
	double empty;
	double on = set ? 1 : 0;
	int retInt = is_SetAutoParameter(hCam, IS_SET_ENABLE_AUTO_WHITEBALANCE, &on, &empty);
	if (retInt != IS_SUCCESS) {
		throw UeyeOpenCVException(hCam, retInt);
	}
}

void UeyeOpencvCam::setAutoGain(bool set) {
	double empty;
	double on = set ? 1 : 0;
	int retInt = is_SetAutoParameter(hCam, IS_SET_ENABLE_AUTO_GAIN, &on, &empty);
	if (retInt != IS_SUCCESS) {
		throw UeyeOpenCVException(hCam, retInt);
	}
}
