#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>
#include <chrono>

#include "classify.hpp"

void printMeasureInfo(MeasureInfo &bestSmall, MeasureInfo &bestLarge, std::string name);
/**
 * @function main
 */
int main( int argc, const char** argv ) {
  cv::Mat frame;

  ClassifyInit();

  MeasureInfo leftSmall, rightSmall, leftLarge, rightLarge;

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
        detectAndDisplay( frame, leftSmall, leftLarge, rightSmall, rightLarge );
        printMeasureInfo(leftSmall, leftLarge, "left");
        printMeasureInfo(rightSmall, rightLarge, "right");
      }
      else {
        printf(" --(!) No captured frame -- Break!");
        break;
      }

      int c = cv::waitKey(10);
      if( (char)c == 'c' ) { break; }
      if( (char)c == 'f' ) {
        imwrite("frame.png",frame);
      }

    }
  }

  ClassifyDeinit();

  return 0;
}

void printMeasureInfo(MeasureInfo &bestSmall, MeasureInfo &bestLarge, std::string name) {
    using namespace std::chrono;
    long ms = duration_cast< milliseconds >(
        system_clock::now().time_since_epoch()
    ).count();

    std::cout << name << ", " <<
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
}
