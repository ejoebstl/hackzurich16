#ifndef EYE_CENTER_H
#define EYE_CENTER_H

#include "opencv2/imgproc/imgproc.hpp"

cv::Rect findEyeCenter(cv::Mat face, cv::Mat face_color, cv::Rect eye, std::string debugWindow);

#endif
