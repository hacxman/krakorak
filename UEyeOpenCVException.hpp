//******************************************************************************
//
//                 Low Cost Vision
//
//******************************************************************************
// Project:        ueyeOpencv
// File:           UEyeOpenCVException.hpp
// Description:    Exception for UeyeOpenCV
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
#include <sstream>
#include <exception>
#include <iostream>

/**
 * Exception for UeyeOpenCV
 */
class UeyeOpenCVException : public std::exception {
private:
        HIDS cam;
        int exceptionId;
public:
        /**
         * Constructor
         * @param cam id of the camera (HIDS)
         * @param err
         */
        UeyeOpenCVException(HIDS cam, int err) {
                exceptionId = err;
        }
        /**
         * returns the error message.
         * @return
         */
        const char * what() const throw () {
                std::stringstream ss;
                ss << "UeyeOpenCVException on camera " << cam <<", with exit code:\t" << exceptionId;

                return ss.str().c_str();
        }
        /**
         * Returns the camera id(HIDS) of the camera on which the are occurred.
         * @return the camera id(HIDS) of the camera on which the are occurred.
         */
        HIDS getCam()
        {
                return cam;
        }
        /**
                * Returns the exception id of the camera on which the are occurred.
                * @return the exception id of the camera on which the are occurred.
                */
        int getExceptionId()
        {
                return exceptionId;
        }
};
