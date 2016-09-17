#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>

#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"


/** Constants **/


/** Function Headers */
void detectAndDisplay( cv::Mat frame );

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
cv::String face_cascade_name = "res/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
std::string main_window_name = "Capture - Face detection";
std::string face_window_name = "Capture - Face";
cv::RNG rng(12345);
cv::Mat debugImage;
cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);

/**
 * @function main
 */
int main( int argc, const char** argv ) {
  cv::Mat frame;

  // Load the cascades
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n"); return -1; };

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

  createCornerKernels();
  ellipse(skinCrCbHist, cv::Point(113, 155.6), cv::Size(23.4, 15.2),
          43.0, 0.0, 360.0, cv::Scalar(255, 255, 255), -1);

  // I make an attempt at supporting both 2.x and 3.x OpenCV
#if CV_MAJOR_VERSION < 3
  CvCapture* capture = cvCaptureFromCAM( -1 );
  if( capture ) {
    while( true ) {
      frame = cvQueryFrame( capture );
#else
  cv::VideoCapture capture(-1);
  if( capture.isOpened() ) {
    while( true ) {
      capture.read(frame);
#endif
      // mirror it
      cv::flip(frame, frame, 1);
      //frame.copyTo(debugImage);

      // Apply the classifier to the frame
      if( !frame.empty() ) {
        detectAndDisplay( frame );
      }
      else {
        printf(" --(!) No captured frame -- Break!");
        break;
      }

      imshow(main_window_name, frame);

      int c = cv::waitKey(10);
      if( (char)c == 'c' ) { break; }
      if( (char)c == 'f' ) {
        imwrite("frame.png",frame);
      }

    }
  }

  releaseCornerKernels();

  return 0;
}

cv::Point unscalePoint(cv::Point p, cv::Rect faceSize, cv::Size origSize) {
  float ratiox = (((float)faceSize.width)/origSize.width);
  float ratioy = (((float)faceSize.height)/origSize.height);
  int x = round(p.x / ratiox) + faceSize.x;
  int y = round(p.y / ratioy) + faceSize.y;
  return cv::Point(x,y);
}

struct MeasureInfo {
    int x, y, r1, r2;
    int matches, rejects, score; 
    int count, sum;
    double avg; 

    MeasureInfo() :
        x(0), y(0), r1(0), r2(0), matches(0), rejects(0), score(0), 
        count(0), sum(0), avg(0) {
    }
};

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
   info.score = info.matches * 2 - info.rejects;

   return info; 
}

cv::Vec4b matchCircleBf(cv::Mat intensity, MeasureInfo &bestSmall, MeasureInfo &bestLarge) {

    if(intensity.rows > 50)
        return cv::Vec4b(0, 0, 0, 0);

    assert(intensity.type() == CV_8UC1);

    cv::Mat in;
    pyrDown(intensity, in);

    const int thresh = 180;

    int score = 0;
    int mrgx = 5;
    int mrgy = 5;


    for(int x = mrgx; x < in.cols - mrgx; x++) {
        for(int y = mrgy; y < in.rows - mrgy; y++) {
            for(int r1 = 1; r1 < in.rows / 4; r1++) {
                for(int r2 = r1; r2 < in.rows / 1.5; r2++) {
                   MeasureInfo large = pxThresh(in, x, y, r1, r2, thresh, 255);
                   MeasureInfo small = pxThresh(in, x, y, 0, r1, 0, thresh);

                   if(score < large.score) {
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
                       score = large.score;
                   } 
                }
            }
        }
    }

    return cv::Vec4b(bestLarge.x * 2, bestLarge.y * 2, bestLarge.r2 * 2, bestLarge.r1 * 2); 
}

void matchLines(cv::Mat _eye, std::string window_name) {
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
            //    ((float)channels[1].at<uchar>(i, j) / 255)
            //    * (1 - (float)channels[2].at<uchar>(i, j) / 255) * 255; 
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

    MeasureInfo bestSmall, bestLarge; 

    cv::Vec4b circle = matchCircleBf(res, bestLarge, bestSmall);

        using namespace std::chrono;
        long ms = duration_cast< milliseconds >(
            system_clock::now().time_since_epoch()
        ).count();

    std::cout << window_name << ", " <<
        ms << ", " <<
        bestLarge.x << ", " <<
        bestLarge.y << ", " <<
        bestLarge.r1 << ", " <<
        bestLarge.r2 << ", " <<
        bestLarge.matches << ", " <<
        bestLarge.rejects << ", " <<
        bestLarge.avg << ", " <<
        bestSmall.matches << ", " <<
        bestSmall.rejects << ", " <<
        bestSmall.avg << std::endl;
   
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

    cv::Mat mix(channels.size() * eye.rows, eye.cols, CV_8UC1);

    for(int i = 0; i < channels.size(); i++) {
        channels[i].copyTo(mix(cv::Rect(0, i * eye.rows, eye.cols, eye.rows)));
    }

    cv::imshow(window_name, mix);
}

void findEyes(cv::Mat frame_gray, cv::Mat frame_color, cv::Rect face) {
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
  const float heightExt = -5;

  leftPupil = cv::Rect(leftPupil.x - widthExt, leftPupil.y - heightExt, leftPupil.width + widthExt * 2, leftPupil.height + heightExt * 2);
  rightPupil = cv::Rect(rightPupil.x - widthExt, rightPupil.y - heightExt, rightPupil.width + widthExt * 2, rightPupil.height + heightExt * 2);
  
  leftPupil += face.tl();
  rightPupil += face.tl();

  //rectangle(frame_color, 
  //        leftPupil,
  //        cv::Scalar(0, 255, 0), 1);

  if(leftPupil.width > 0 && leftPupil.height > 0) 
      matchLines(frame_color(leftPupil), "left_pupil");

  //rectangle(frame_color, 
  //        rightPupil,
  //        cv::Scalar(0, 255, 0), 1);
  
  if(rightPupil.width > 0 && rightPupil.height > 0) 
      matchLines(frame_color(rightPupil), "right_pupil");
  
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
void detectAndDisplay( cv::Mat frame ) {
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
    findEyes(frame_gray, frame, faces[0]);
  }
}
