#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>
#include <chrono>

#include "constants.h"
#include "common.hpp"
#include "findEyeCenter.h"
#include "findEyeCorner.h"
#include "classify.hpp"


/** Constants **/



/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
cv::String face_cascade_name = "res/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
std::string main_window_name = "Capture - Face detection";
std::string face_window_name = "Capture - Face";
cv::RNG rng(12345);
cv::Mat debugImage;
cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);

void ClassifyInit() {
  // Load the cascades
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n"); return; };

  if(kEnableDebug) {
      cv::namedWindow("left_pupil",CV_WINDOW_NORMAL);
      cv::moveWindow("left_pupil", 400, 10);
      cv::namedWindow("right_pupil",CV_WINDOW_NORMAL);
      cv::moveWindow("right_pupil", 500, 10);
      cv::namedWindow(main_window_name,CV_WINDOW_NORMAL);
      cv::moveWindow(main_window_name, 400, 100);
      cv::namedWindow(face_window_name,CV_WINDOW_NORMAL);
      cv::moveWindow(face_window_name, 10, 100);
      cv::namedWindow("Right Eye",CV_WINDOW_NORMAL);
      cv::moveWindow("Right Eye", 10, 400);
      cv::namedWindow("Left Eye",CV_WINDOW_NORMAL);
      cv::moveWindow("Left Eye", 10, 500);
      cv::namedWindow("aa",CV_WINDOW_NORMAL);
      cv::moveWindow("aa", 10, 600);
      cv::namedWindow("aaa",CV_WINDOW_NORMAL);
      cv::moveWindow("aaa", 10, 650);
      
  }

  createCornerKernels();
  ellipse(skinCrCbHist, cv::Point(113, 155.6), cv::Size(23.4, 15.2),
          43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);
}

void ClassifyDeinit() {
  releaseCornerKernels();
}

cv::Point unscalePoint(cv::Point p, cv::Rect faceSize, cv::Size origSize) {
  float ratiox = (((float)faceSize.width)/origSize.width);
  float ratioy = (((float)faceSize.height)/origSize.height);
  int x = round(p.x / ratiox) + faceSize.x;
  int y = round(p.y / ratioy) + faceSize.y;
  return cv::Point(x,y);
}


MeasureInfo pxThresh(cv::Mat img, int cx, int cy, int r1, int r2, int low, int high) {
   int match = 0;

   MeasureInfo info;

   for(int x = cx - r2; x < cx + r2; x++) {
       for(int y = cy - r2; y < cy + r2; y++) {
            if(x >= 0 && y >= 0 && x < img.cols && y < img.rows) {
                int dist = ((x - cx) * (x - cx) + (y - cy) * (y - cy));
                if(dist <= r2 * r2 && dist >= r1 * r1) {
                    uchar intensity = img.at<uchar>(y, x);
                    info.sum += intensity;
                    if(intensity >= low && intensity <= high) {
                        info.matches++;
                    } else {
                        info.rejects++;
                    }
                    info.count++;
                }
            }
       }
   }
   info.avg = (double)info.sum / (double)info.count;
   info.score = info.matches * 4 - info.rejects;

   return info; 
}

void matchRadius(cv::Mat &in, MeasureInfo &bestSmall, MeasureInfo &bestLarge, int &score, int x, int y, int thresh) {
    for(int r1 = 1; r1 < in.rows / 5; r1++) {
        for(int r2 = r1; r2 < in.rows / 2.1; r2++) {
           MeasureInfo large = pxThresh(in, x, y, r1, r2, thresh, 255);
           MeasureInfo small = pxThresh(in, x, y, 0, r1, 0, thresh);

           if(score < large.score + small.score * 2) {
               bestLarge = large;
               bestSmall = small;
               bestLarge.x = x;
               bestLarge.y = y;
               bestLarge.r1 = r1;
               bestLarge.r2 = r2;
               bestSmall.x = x;
               bestSmall.y = y;
               bestSmall.r1 = r1;
               bestSmall.r2 = r2;
               score = large.score + small.score * 2;
           } 
        }
    }
}

std::deque<cv::Point> eyeCenterThesis;

cv::Vec4b matchCircleBf(cv::Mat intensity, MeasureInfo &bestSmall, MeasureInfo &bestLarge, cv::Point globalOffset) {

    if(intensity.rows > 50)
        return cv::Vec4b(0, 0, 0, 0);

    assert(intensity.type() == CV_8UC1);

    cv::Mat in; 
    const int dt = 2;
    pyrDown(intensity, in);

    const int thresh = 80;

    int score = 0;
    int mrgx = 5;
    int mrgy = 0;


    for(int x = mrgx; x < in.cols - mrgx; x++) {
        for(int y = mrgy; y < in.rows - mrgy; y++) {
            matchRadius(in, bestSmall, bestLarge, score, x, y, thresh);
        }
    }

    //eyeCenterThesis.push_back(cv::Point(bestLarge.x, bestLarge.y) + globalOffset);

    //while(eyeCenterThesis.size() > 3) {
    //    eyeCenterThesis.pop_front();
    //}

    //cv::Point center = Median(eyeCenterThesis) - globalOffset;

    //if(center.x < in.cols && center.y < in.rows && center.y >= 0 && center.x >= 0) {
    //    score = 0;
    //    matchRadius(in, bestSmall, bestLarge, score, center.x, center.y, thresh);
    //} else {
    //    std::cout << "Discard Thesis " << center << std::endl;
    //}

    return cv::Vec4b(bestLarge.x * dt, bestLarge.y * dt, bestLarge.r2 * dt, bestLarge.r1 * dt); 
}

std::deque<cv::Rect> left_pupils;
std::deque<cv::Rect> right_pupils;

void matchLines(cv::Mat _eye, std::string window_name, cv::Point globalOffset, MeasureInfo &bestSmall, MeasureInfo &bestLarge) {
    if(_eye.rows == 0)
        return; 

    cv::Mat eye = _eye.clone();
    cv::Mat hsv;
    cv::cvtColor(_eye, hsv, cv::COLOR_RGB2HSV);
    std::vector<cv::Mat> channels; 
    std::vector<cv::Mat> channels2; 
    cv::Mat res = cv::Mat::zeros(eye.size(), CV_8UC1);
    cv::split(eye, channels);
    cv::split(hsv, channels2);
    channels.insert(channels.end(), channels2.begin(), channels2.end());
    channels.push_back(res);

    for(int i = 0; i < eye.rows; i++) {
        for(int j = 0; j < eye.cols; j++) {
            //res.at<uchar>(i, j) = //255 - channels[2].at<uchar>(i, j);
            //    ((float)channels2[1].at<uchar>(i, j) / 255)
            //    * (1 - (float)channels2[2].at<uchar>(i, j) / 255) * 255; 
            //if(channels[0].at<uchar>(i, j) < channels[1].at<uchar>(i, j)) {
            //    res.at<uchar>(i, j) = 0;
            //} else {
            //    res.at<uchar>(i, j) = 255;
           // }
            res.at<uchar>(i, j) = 
                std::min(255, std::max(0, 
                            (int)channels[1].at<uchar>(i, j) - 
                            (int)channels[0].at<uchar>(i, j) + 128));
            if(channels2[2].at<uchar>(i, j) > 100) {
                res.at<uchar>(i, j) = 0;
            }
        }
    }
    equalizeHist(res, res);

    std::vector<cv::Vec4b> circles;

    cv::Vec4b circle = matchCircleBf(res, bestLarge, bestSmall, globalOffset);

    bestLarge.x += globalOffset.x;
    bestLarge.y += globalOffset.y;

    circles.push_back(circle);

    for( size_t i = 0; i < circles.size(); i++ )
    {
        cv::Point center(circles[i][0], circles[i][1]);
        int radius = circles[i][2];
        cv::circle(res, center, 1, 255, -1, 8, 0);
        cv::circle(res, center, circles[i][3], 255, 1, 8, 0);
        cv::circle(res, center, radius, 255, 1, 8, 0);
        //std::cout << "found pupil " + window_name + ": " << ms << ", " << circle << " in " << _eye.size() << std::endl;
        
        cv::circle(_eye, center, 1, cv::Scalar(0, 255, 0), -1, 8, 0);
        cv::circle(_eye, center, radius, cv::Scalar(255, 0, 0), 1, 8, 0);
        cv::circle(_eye, center, circles[i][3], cv::Scalar(0, 0, 255), 1, 8, 0);
    }

    if(circle[2] > 0) {
        static int debugc = 0;
        
        if(kEnableDebug) {
            int x = std::max(0, circle[0] - circle[2]);
            int y = std::max(0, circle[1] - circle[2]);

            cv::Mat cutout = 
                    _eye(cv::Rect(x, y, 
                                std::min(_eye.cols - x, circle[2] * 2), 
                                std::min(_eye.rows - y, circle[2] * 2)));

            debugc++;
            std::vector<int> hist(255);

            for(int i = 0; i < cutout.cols; i++) {
                for(int j = 0; j < cutout.rows; j++) {
                        hist[channels[0].at<uchar>(i, j)]++;
                }
            }

            cv::Mat histviz = cv::Mat::zeros(200, 255, CV_8UC1);

            for(int i = 0; i < 254; i++) {
                cv::line(histviz, cv::Point(200, i), cv::Point(200 - hist[i], i), 255, 1);
            }

            cv::imshow("hist" + window_name, histviz);

            cv::imwrite("dbg/" + window_name + std::to_string(debugc) + ".bmp", cutout);
            debugc++;
        }
    }

    if(kEnableDebug) {
        cv::Mat mix(channels.size() * eye.rows, eye.cols, CV_8UC1);

        for(int i = 0; i < channels.size(); i++) {
            channels[i].copyTo(mix(cv::Rect(0, i * eye.rows, eye.cols, eye.rows)));
        }
        cv::imshow(window_name, mix);
    }
}


void findEyes(cv::Mat frame_gray, cv::Mat frame_color, cv::Rect face, MeasureInfo &leftSmall, MeasureInfo &leftLarge, MeasureInfo &rightSmall, MeasureInfo &rightLarge) {
  cv::Mat faceROI = frame_gray(face);
  cv::Mat colorFaceROI = frame_color(face);
  cv::Mat debugFace = faceROI;

  if (kSmoothFaceImage) {
    double sigma = kSmoothFaceFactor * face.width;
    GaussianBlur( faceROI, faceROI, cv::Size( 0, 0 ), sigma);
  }
  //-- Find eye regions and draw them
  int eye_region_width = face.width * (kEyePercentWidth/100.0);
  int eye_region_height = face.width * (kEyePercentHeight/100.0);
  int eye_region_top = face.height * (kEyePercentTop/100.0);
  cv::Rect leftEyeRegion(face.width*(kEyePercentSide/100.0),
                         eye_region_top,eye_region_width,eye_region_height);
  cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0),
                          eye_region_top,eye_region_width,eye_region_height);

  //-- Find Eye Centers
  cv::Rect leftPupil = findEyeCenter(faceROI, colorFaceROI, leftEyeRegion,"Left Eye");
  cv::Rect rightPupil = findEyeCenter(faceROI, colorFaceROI, rightEyeRegion,"Right Eye");

  if(leftPupil.width != 0)
      left_pupils.push_back(leftPupil + face.tl());

  if(rightPupil.width != 0)
      right_pupils.push_back(rightPupil + face.tl());

  if(left_pupils.size() == 0 || right_pupils.size() == 0)
      return;

  const int median_count = 10;

  if(left_pupils.size() > median_count)
      left_pupils.pop_front();
  
  if(right_pupils.size() > median_count)
      right_pupils.pop_front();

  leftPupil = Median(left_pupils) - face.tl();
  rightPupil = Median(right_pupils) - face.tl();

  // get corner regions
  cv::Rect leftRightCornerRegion(leftEyeRegion);
  leftRightCornerRegion.width -= leftPupil.x;
  leftRightCornerRegion.x += leftPupil.x;
  leftRightCornerRegion.height /= 2;
  leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
  cv::Rect leftLeftCornerRegion(leftEyeRegion);
  leftLeftCornerRegion.width = leftPupil.x;
  leftLeftCornerRegion.height /= 2;
  leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
  cv::Rect rightLeftCornerRegion(rightEyeRegion);
  rightLeftCornerRegion.width = rightPupil.x;
  rightLeftCornerRegion.height /= 2;
  rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
  cv::Rect rightRightCornerRegion(rightEyeRegion);
  rightRightCornerRegion.width -= rightPupil.x;
  rightRightCornerRegion.x += rightPupil.x;
  rightRightCornerRegion.height /= 2;
  rightRightCornerRegion.y += rightRightCornerRegion.height / 2;
  rectangle(debugFace,leftRightCornerRegion,200);
  rectangle(debugFace,leftLeftCornerRegion,200);
  rectangle(debugFace,rightLeftCornerRegion,200);
  rectangle(debugFace,rightRightCornerRegion,200);
  // change eye centers to face coordinates
  rightPupil.x += rightEyeRegion.x;
  rightPupil.y += rightEyeRegion.y;
  leftPupil.x += leftEyeRegion.x;
  leftPupil.y += leftEyeRegion.y;
  // draw eye centers
  rectangle(debugFace, rightPupil, 255, 1);
  rectangle(debugFace, leftPupil, 255, 1);

  const float widthExt = 5;
  const float heightExt = 0;

  leftPupil = cv::Rect(leftPupil.x - widthExt, leftPupil.y - heightExt, leftPupil.width + widthExt * 2, leftPupil.height + heightExt * 2);
  rightPupil = cv::Rect(rightPupil.x - widthExt, rightPupil.y - heightExt, rightPupil.width + widthExt * 2, rightPupil.height + heightExt * 2);
  
  leftPupil += face.tl();
  rightPupil += face.tl();

  //rectangle(frame_color, 
  //        leftPupil,
  //        cv::Scalar(0, 255, 0), 1);

  if(leftPupil.width > 0 && leftPupil.height > 0) 
      matchLines(frame_color(leftPupil), "left_pupil", leftPupil.tl(), leftSmall, leftLarge);

  //rectangle(frame_color, 
  //        rightPupil,
  //        cv::Scalar(0, 255, 0), 1);
  
  if(rightPupil.width > 0 && rightPupil.height > 0) 
      matchLines(frame_color(rightPupil), "right_pupil", rightPupil.tl(), rightSmall, rightLarge);
  
  //-- Find Eye Corners
  /*
  if (kEnableEyeCorner) {
    cv::Point2f leftRightCorner = findEyeCorner(faceROI(leftRightCornerRegion), true, false);
    leftRightCorner.x += leftRightCornerRegion.x;
    leftRightCorner.y += leftRightCornerRegion.y;
    cv::Point2f leftLeftCorner = findEyeCorner(faceROI(leftLeftCornerRegion), true, true);
    leftLeftCorner.x += leftLeftCornerRegion.x;
    leftLeftCorner.y += leftLeftCornerRegion.y;
    cv::Point2f rightLeftCorner = findEyeCorner(faceROI(rightLeftCornerRegion), false, true);
    rightLeftCorner.x += rightLeftCornerRegion.x;
    rightLeftCorner.y += rightLeftCornerRegion.y;
    cv::Point2f rightRightCorner = findEyeCorner(faceROI(rightRightCornerRegion), false, false);
    rightRightCorner.x += rightRightCornerRegion.x;
    rightRightCorner.y += rightRightCornerRegion.y;
    circle(faceROI, leftRightCorner, 3, 200);
    circle(faceROI, leftLeftCorner, 3, 200);
    circle(faceROI, rightLeftCorner, 3, 200);
    circle(faceROI, rightRightCorner, 3, 200);
  }
*/
  if(kEnableDebug)
      imshow(face_window_name, faceROI);
//  cv::Rect roi( cv::Point( 0, 0 ), faceROI.size());
//  cv::Mat destinationROI = debugImage( roi );
//  faceROI.copyTo( destinationROI );
}


cv::Mat findSkin (cv::Mat &frame) {
  cv::Mat input;
  cv::Mat output = cv::Mat(frame.rows,frame.cols, CV_8U);

  cvtColor(frame, input, CV_BGR2YCrCb);

  for (int y = 0; y < input.rows; ++y) {
    const cv::Vec3b *Mr = input.ptr<cv::Vec3b>(y);
//    uchar *Or = output.ptr<uchar>(y);
    cv::Vec3b *Or = frame.ptr<cv::Vec3b>(y);
    for (int x = 0; x < input.cols; ++x) {
      cv::Vec3b ycrcb = Mr[x];
//      Or[x] = (skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) > 0) ? 255 : 0;
      if(skinCrCbHist.at<uchar>(ycrcb[1], ycrcb[2]) == 0) {
        Or[x] = cv::Vec3b(0,0,0);
      }
    }
  }
  return output;
}

/**
 * @function detectAndDisplay
 */
void detectAndDisplay(cv::Mat frame, MeasureInfo &leftSmall, MeasureInfo &leftLarge, MeasureInfo &rightSmall, MeasureInfo &rightLarge) {
  std::vector<cv::Rect> faces;
  //cv::Mat frame_gray;

  std::vector<cv::Mat> rgbChannels(3);
  cv::split(frame, rgbChannels);
  cv::Mat frame_gray = rgbChannels[2];

  //cvtColor( frame, frame_gray, CV_BGR2GRAY );
  //equalizeHist( frame_gray, frame_gray );
  //cv::pow(frame_gray, CV_64F, frame_gray);
  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );
//  findSkin(debugImage);

  for( int i = 0; i < faces.size(); i++ )
  {
    rectangle(debugImage, faces[i], 1234);
  }
  //-- Show what you got
  if (faces.size() > 0) {
    findEyes(frame_gray, frame, faces[0], leftSmall, leftLarge, rightSmall, rightLarge);
    if(kEnableDebug)  
        imshow(main_window_name, frame);

  }
}
