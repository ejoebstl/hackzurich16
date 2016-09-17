#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//#include <mgl2/mgl.h>

#include <iostream>
#include <queue>
#include <stdio.h>

#include "constants.h"
#include "helpers.h"
#include "common.hpp"

// Pre-declarations
cv::Mat floodKillEdges(cv::Mat &mat);

#pragma mark Visualization
/*
template<typename T> mglData *matToData(const cv::Mat &mat) {
  mglData *data = new mglData(mat.cols,mat.rows);
  for (int y = 0; y < mat.rows; ++y) {
    const T *Mr = mat.ptr<T>(y);
    for (int x = 0; x < mat.cols; ++x) {
      data->Put(((mreal)Mr[x]),x,y);
    }
  }
  return data;
}

void plotVecField(const cv::Mat &gradientX, const cv::Mat &gradientY, const cv::Mat &img) {
  mglData *xData = matToData<double>(gradientX);
  mglData *yData = matToData<double>(gradientY);
  mglData *imgData = matToData<float>(img);
  
  mglGraph gr(0,gradientX.cols * 20, gradientY.rows * 20);
  gr.Vect(*xData, *yData);
  gr.Mesh(*imgData);
  gr.WriteFrame("vecField.png");
  
  delete xData;
  delete yData;
  delete imgData;
}*/

#pragma mark Helpers

cv::Point unscalePoint(cv::Point p, cv::Rect origSize) {
  float ratio = (((float)kFastEyeWidth)/origSize.width);
  int x = round(p.x / ratio);
  int y = round(p.y / ratio);
  return cv::Point(x,y);
}

void scaleToFastSize(const cv::Mat &src,cv::Mat &dst) {
  cv::resize(src, dst, cv::Size(kFastEyeWidth,(((float)kFastEyeWidth)/src.cols) * src.rows));
}

cv::Mat computeMatXGradient(const cv::Mat &mat) {
  cv::Mat out(mat.rows,mat.cols,CV_64F);
  
  for (int y = 0; y < mat.rows; ++y) {
    const uchar *Mr = mat.ptr<uchar>(y);
    double *Or = out.ptr<double>(y);
    
    Or[0] = Mr[1] - Mr[0];
    for (int x = 1; x < mat.cols - 1; ++x) {
      Or[x] = (Mr[x+1] - Mr[x-1])/2.0;
    }
    Or[mat.cols-1] = Mr[mat.cols-1] - Mr[mat.cols-2];
  }
  
  return out;
}

#pragma mark Main Algorithm

float pxThresh2(cv::Mat img, int cx, int cy, int r1, int r2, int low, int high) {
   int ct = 0;
   int match = 0;
   for(int x = cx - r2; x < cx + r2; x++) {
       for(int y = cy - r2; y < cy + r2; y++) {
            if(x >= 0 && y >= 0 && x < img.cols && y < img.rows) {
                int dist = ((x - cx) * (x - cx) + (y - cy) * (y - cy));
                if(dist <= r2 * r2 && dist >= r1 * r1) {
                    uchar intensity = img.at<uchar>(y, x);
                    if(intensity >= low && intensity <= high) {
                        match += 2;
                    } else {
                        match--;
                    }
                    ct++;
                }
            }
       }
   }
   return match; 
}

cv::Vec3f matchCircleBf2(cv::Mat intensity, int thresh) {

    if(intensity.rows > 50)
        return cv::Vec3f(0, 0, 0);

    assert(intensity.type() == CV_8UC1);

    cv::Mat in;
    pyrDown(intensity, in);

    float max = 0;
    int mx = -1, my = -1, mr1 = 0, mr2 = 0;

    int marg = intensity.rows / 8;

    for(int r2 = 1; r2 < in.rows / 2.1; r2++) {
        for(int x = r2 + marg; x < in.cols - r2 - marg; x++) {
            for(int y = r2 + marg; y < in.rows - r2 - marg; y++) {
               float v = 
                   pxThresh2(in, x, y, 0, r2, thresh, 255);

               if(v > max) {
                    max = v;
                    mx = x;
                    my = y;
                    mr2 = r2;
               }
            }
        }
    }
   
    return cv::Vec3f(mx * 2, my * 2, mr2 * 2); 
}
void testPossibleCentersFormula(int x, int y, const cv::Mat &weight,double gx, double gy, cv::Mat &out) {
  // for all possible centers
  for (int cy = 0; cy < out.rows; ++cy) {
    double *Or = out.ptr<double>(cy);
    const unsigned char *Wr = weight.ptr<unsigned char>(cy);
    for (int cx = 0; cx < out.cols; ++cx) {
      if (x == cx && y == cy) {
        continue;
      }
      // create a vector from the possible center to the gradient origin
      double dx = x - cx;
      double dy = y - cy;
      // normalize d
      double magnitude = sqrt((dx * dx) + (dy * dy));
      dx = dx / magnitude;
      dy = dy / magnitude;
      double dotProduct = dx*gx + dy*gy;
      dotProduct = std::max(0.0,dotProduct);
      // square and multiply by the weight
      if (kEnableWeight) {
        Or[cx] += dotProduct * dotProduct * (Wr[cx]/kWeightDivisor);
      } else {
        Or[cx] += dotProduct * dotProduct;
      }
    }
  }
}

cv::Rect findEyeCenter(cv::Mat face, cv::Mat face_color, cv::Rect eye, std::string debugWindow) {
  cv::Mat eyeROIUnscaled = face(eye);
  cv::Mat eyeROI, eyeROIColorScaled;
  cv::Mat eyeROIColor = face_color(eye);
  scaleToFastSize(eyeROIUnscaled, eyeROI);
  scaleToFastSize(eyeROIColor, eyeROIColorScaled);
  // draw eye region
  rectangle(face,eye,1234);
  //-- Find the gradient
  cv::Mat gradientX = computeMatXGradient(eyeROI);
  cv::Mat gradientY = computeMatXGradient(eyeROI.t()).t();
  //-- Normalize and threshold the gradient
  // compute all the magnitudes
  cv::Mat mags = matrixMagnitude(gradientX, gradientY);
  //compute the threshold
  double gradientThresh = computeDynamicThreshold(mags, kGradientThreshold);
  //double gradientThresh = kGradientThreshold;
  //double gradientThresh = 0;
  //normalize
  for (int y = 0; y < eyeROI.rows; ++y) {
    double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
    const double *Mr = mags.ptr<double>(y);
    for (int x = 0; x < eyeROI.cols; ++x) {
      double gX = Xr[x], gY = Yr[x];
      double magnitude = Mr[x];
      if (magnitude > gradientThresh) {
        Xr[x] = gX/magnitude;
        Yr[x] = gY/magnitude;
      } else {
        Xr[x] = 0.0;
        Yr[x] = 0.0;
      }
    }
  }
  //imshow(debugWindow,gradientX);
  //-- Create a blurred and inverted image for weighting
  cv::Mat weight;
  GaussianBlur( eyeROI, weight, cv::Size( kWeightBlurSize, kWeightBlurSize ), 0, 0 );
  for (int y = 0; y < weight.rows; ++y) {
    unsigned char *row = weight.ptr<unsigned char>(y);
    for (int x = 0; x < weight.cols; ++x) {
      row[x] = (255 - row[x]);
    }
  }


  //imshow(debugWindow,eyeROIColor);
  //-- Run the algorithm!
  cv::Mat outSum = cv::Mat::zeros(eyeROI.rows,eyeROI.cols,CV_64F);
  // for each possible gradient location
  // Note: these loops are reversed from the way the paper does them
  // it evaluates every possible center for each gradient location instead of
  // every possible gradient location for every center.
  printf("Eye Size: %ix%i\n",outSum.cols,outSum.rows);
  for (int y = 0; y < weight.rows; ++y) {
    const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
    for (int x = 0; x < weight.cols; ++x) {
      double gX = Xr[x], gY = Yr[x];
      if (gX == 0.0 && gY == 0.0) {
        continue;
      }
      testPossibleCentersFormula(x, y, weight, gX, gY, outSum);
    }
  }

  // scale all the values down, basically averaging them
  double numGradients = (weight.rows*weight.cols);
  cv::Mat out;
  outSum.convertTo(out, CV_32F,1.0/numGradients);
  //imshow(debugWindow,out);
  //-- Find the maximum point
  cv::Point maxP;
  double maxVal;
  cv::minMaxLoc(out, NULL,&maxVal,NULL,&maxP);

  // Seperating stuff. 
  uchar center = weight.at<uchar>(maxP.y, maxP.x);
  const uchar t = 50;
  cv::Mat whsl;
  cv::cvtColor(eyeROIColorScaled, whsl, cv::COLOR_BGR2HSV);


  cv::Vec3f circle = matchCircleBf2(weight, center - t);
  std::cout << "weight circle: " << circle << std::endl;
  cv::circle(weight, cv::Point(circle[0], circle[1]), circle[2], 255, 1);

  if(kEnableDebug) {
    imshow(debugWindow,weight);
  }

  cv::Rect region(circle[0] - circle[2], circle[1] - circle[2], circle[2] * 2, circle[2] * 2); 

  //if(region.width <= 0 || region.height <= 0) {
  //  region = cv::Rect(maxP, cv::Size(1, 1));
  //}

  /*
  for(int i = 0; i < weight.rows; i++) {
      for(int j = 0; j < weight.cols; j++) {
          if(std::abs(weight.at<uchar>(i, j) - center) > t) {
          //cv::Vec3b color = whsl.at<cv::Vec3b>(i, j);
          //if(color[1] < 40) {
              weight.at<uchar>(i, j) = 0;
          } else {
              weight.at<uchar>(i, j) = 255;
          }
      }
  }


  cv::floodFill(weight, maxP, 1, &region);
  cv::rectangle(eyeROIColorScaled, region.tl(), region.br(), cv::Scalar(0, 0, 255), 1);
  */

  //-- Flood fill the edges
  if(kEnablePostProcess) {
    cv::Mat floodClone;
    //double floodThresh = computeDynamicThreshold(out, 1.5);
    double floodThresh = maxVal * kPostProcessThreshold;
    cv::threshold(out, floodClone, floodThresh, 0.0f, cv::THRESH_TOZERO);
    if(kPlotVectorField) {
      //plotVecField(gradientX, gradientY, floodClone);
      imwrite("eyeFrame.png",eyeROIUnscaled);
    }
    cv::Mat mask = floodKillEdges(floodClone);
    //imshow(debugWindow + " Mask",mask);
    //imshow(debugWindow,out);
    // redo max
    cv::minMaxLoc(out, NULL,&maxVal,NULL,&maxP,mask);
  }

  return cv::Rect(unscalePoint(region.tl(), eye), unscalePoint(region.br(), eye));
}

#pragma mark Postprocessing

bool floodShouldPushPoint(const cv::Point &np, const cv::Mat &mat) {
  return inMat(np, mat.rows, mat.cols);
}

// returns a mask
cv::Mat floodKillEdges(cv::Mat &mat) {
  rectangle(mat,cv::Rect(0,0,mat.cols,mat.rows),255);
  
  cv::Mat mask(mat.rows, mat.cols, CV_8U, 255);
  std::queue<cv::Point> toDo;
  toDo.push(cv::Point(0,0));
  while (!toDo.empty()) {
    cv::Point p = toDo.front();
    toDo.pop();
    if (mat.at<float>(p) == 0.0f) {
      continue;
    }
    // add in every direction
    cv::Point np(p.x + 1, p.y); // right
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x - 1; np.y = p.y; // left
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x; np.y = p.y + 1; // down
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x; np.y = p.y - 1; // up
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    // kill it
    mat.at<float>(p) = 0.0f;
    mask.at<uchar>(p) = 0;
  }
  return mask;
}
