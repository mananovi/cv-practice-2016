// ght.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"
#include "stdafx.h"
#include <iostream>
#include <string>
#include <conio.h> 
#include <math.h>
 
#include <opencv2\opencv.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
using namespace cv;
using namespace std;
#define TORAD 0.0174532
const int splitSet = 10;
double ugl(int nox, int noy, int stx, int sty)
{
   double x,y;
   x = nox-stx;
   y = noy-sty;
   if ((x==0.0) && (y == 0.0))
      return -1.0;
   if (x == 0.0)
      return ((y > 0.0) ? 90 : 270);
   double theta = atan(y/x);
   theta *= 360 / (2 * 3.1415926);
   if (x > 0.0)
      return ((y >= 0.0) ? theta : 360 + theta);
   else
      return (180 + theta);
}


void ratioTestMatching(DescriptorMatcher& descriptorMatcher, const Mat& descriptors1, const Mat& descriptors2,
                       vector<DMatch>& filteredMatches12, float ratio = 0.6f)
{
  const int knn = 2;
  filteredMatches12.clear();
  vector<vector<DMatch> > matches12;
  descriptorMatcher.knnMatch(descriptors1, descriptors2, matches12, 2);
  for (size_t m = 0; m < matches12.size(); m++)
  {
    if (matches12[m][0].distance / matches12[m][1].distance < ratio)
      filteredMatches12.push_back(matches12[m][0]);
  }
}

void crossCheckMatching(Ptr<DescriptorMatcher>& descriptorMatcher, const Mat& descriptors1, const Mat& descriptors2,
                        vector<DMatch>& filteredMatches12, int knn = 1)
{
  filteredMatches12.clear();
  vector<vector<DMatch> > matches12, matches21;
  descriptorMatcher->knnMatch(descriptors1, descriptors2, matches12, knn);
  descriptorMatcher->knnMatch(descriptors2, descriptors1, matches21, knn);
  for (size_t m = 0; m < matches12.size(); m++)
  {
    bool findCrossCheck = false;
    for (size_t fk = 0; fk < matches12[m].size(); fk++)
    {
      DMatch forward = matches12[m][fk];

      for (size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++)
      {
        DMatch backward = matches21[forward.trainIdx][bk];
        if (backward.trainIdx == forward.queryIdx)
        {
          filteredMatches12.push_back(forward);
          findCrossCheck = true;
          break;
        }
      }
      if (findCrossCheck)
        break;
    }
  }
}

void drawX(Point& p, Mat& unionImage)
{
  int x = p.x;
  int y = p.y;
  Point pt1(x - 5, y - 5);
  Point pt2(x + 5, y + 5);
  line(unionImage, pt1, pt2, Scalar(0, 0, 255), 2);
  pt1 = Point(x - 5, y + 5);
  pt2 = Point(x + 5, y - 5);
  line(unionImage, pt1, pt2, Scalar(0, 0, 255), 2);
}

void drawKpt(Mat& img, const KeyPoint& p, const Scalar& color, int flags, Point offset = Point(0, 0))
{
  Point center(cvRound(p.pt.x)+offset.x, cvRound(p.pt.y)+offset.y);

  if (flags & DrawMatchesFlags::DRAW_RICH_KEYPOINTS)
  {
    int radius = cvRound(p.size / 2); // KeyPoint::size is a diameter

    // draw the circles around keypoints with the keypoints size
    circle(img, center, radius, color, 1, CV_AA);

    // draw orientation of the keypoint, if it is applicable
    if (p.angle != -1)
    {
      float srcAngleRad = p.angle * (float)CV_PI / 180.f;
      Point orient(cvRound(cos(srcAngleRad) * radius), cvRound(sin(srcAngleRad) * radius));
      line(img, center, center + orient, color, 1, CV_AA);
    }
  }
  else
  {
    // draw center with R=3
    int radius = 3;
    circle(img, center, radius, color, 1, CV_AA);
  }
}

int main(int argc, char* argv[])
{
	
  if (argc < 3)
  {
    cout << "Format: ght <image.png> <image_in_scene.png>" << endl;
    return -1;
  }
  
  cout << "< Creating detector, descriptor extractor and descriptor matcher ..." << endl;
  Ptr<FeatureDetector> detector = FeatureDetector::create("ORB");
  Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("ORB");
  // Ptr<DescriptorMatcher> descriptorMatcher = new BruteForceMatcher<L1<float> > ();
  BFMatcher descriptorMatcher(NORM_HAMMING);

  cout << "< Reading the images..." << endl;
  //Mat image = imread( "box.png" ), scene = imread( "box_in_scene.png" );
  Mat image = imread(argv[1]), scene = imread(argv[2]);
  cout << ">" << endl;
  if (image.empty() || scene.empty())
  {
    cout << "Can not read images" << endl;
    return -1;
  }

  cout << endl << "< Extracting keypoints from first image..." << endl;
  vector<KeyPoint> keypoints1;
  detector->detect(image, keypoints1);
  cout << keypoints1.size() << " points" << endl << ">" << endl;

  cout << "< Computing descriptors for keypoints from first image..." << endl;
  Mat descriptors1;
  descriptorExtractor->compute(image, keypoints1, descriptors1);
  cout << ">" << endl;

  namedWindow("keypoints", 0);
  Mat drawImg;
  drawKeypoints(image, keypoints1, drawImg, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  imshow("keypoints", drawImg);
  waitKey(0);

  cout << endl << "< Extracting keypoints from second image..." << endl;
  vector<KeyPoint> keypoints2;
  detector->detect(scene, keypoints2);
  cout << keypoints2.size() << " points" << endl << ">" << endl;

  cout << "< Computing descriptors for keypoints from second image..." << endl;
  Mat descriptors2;
  descriptorExtractor->compute(scene, keypoints2, descriptors2);
  cout << ">" << endl;

  drawKeypoints(scene, keypoints2, drawImg, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  imshow("keypoints", drawImg);
  waitKey(0);

  vector<DMatch> matches;
  //crossCheckMatching(descriptorMatcher, descriptors1, descriptors2, matches, 1);
  ratioTestMatching(descriptorMatcher, descriptors1, descriptors2, matches, 0.8);
  namedWindow("matches", 0);
  drawMatches(image, keypoints1, scene, keypoints2, matches, drawImg, Scalar(0, 255, 0), Scalar(0, 0, 255), Mat(),
              DrawMatchesFlags::DEFAULT);
  imshow("matches", drawImg);
  waitKey(0);

  Point center(image.cols / 2, image.rows / 2);
  Mat hgt(scene.rows/splitSet, scene.cols/splitSet, CV_32S, Scalar::all(0));

  for (int matchInd = 0; matchInd <  matches.size(); matchInd++)
  {
	  //drawKpt( image, keypoints1[ matches[ matchInd ].queryIdx ] , Scalar(0, 0, 255), 1);
	  //drawKpt( scene, keypoints2[ matches[ matchInd ].trainIdx ] , Scalar(0, 0, 255), 1);
	  /*
	  double ang;
	  double rads =  sqrt( (center.x - keypoints1[ matchInd ].pt.x ) * (center.x
		  - keypoints1[ matchInd ].pt.x) + 
		   (center.y - keypoints1[ matchInd ].pt.y ) * (center.y - keypoints1[ matchInd ].pt.y) );
	  ang = ugl(  center.x, center.y, keypoints1[ matchInd ].pt.x, keypoints1[ matchInd ].pt.y
		  );
	  Point nCen( keypoints1[ matchInd ].pt.x + rads * cos( (float)CV_PI / 180.f*ang ),
		  keypoints1[ matchInd ].pt.y + rads * sin( (float)CV_PI / 180.f*ang ) );
	  Point keyP( keypoints1[ matchInd ].pt.x, keypoints1[ matchInd ].pt.y );
	  line( image, keyP,  nCen, Scalar(0, 0, 255), 1, CV_AA);
	  */
	  
	  double ang = ugl(  center.x, center.y, keypoints1[  matches[ matchInd ].queryIdx ].pt.x,
		  keypoints1[  matches[ matchInd ].queryIdx ].pt.y );
	  double radEt =  sqrt( (center.x - keypoints1[  matches[ matchInd ].queryIdx ].pt.x ) *
		  (center.x - keypoints1[  matches[ matchInd ].queryIdx ].pt.x) + 
		   (center.y - keypoints1[  matches[ matchInd ].queryIdx ].pt.y ) * 
		   (center.y - keypoints1[  matches[ matchInd ].queryIdx ].pt.y) );
	  double delta = keypoints2[ matches[ matchInd ].trainIdx ].angle -
		  keypoints1[ matches[ matchInd ].queryIdx ].angle;
	  double scAngle = ang + delta;
	  double Scale = keypoints2[ matches[ matchInd ].trainIdx ].size /
		  keypoints1[ matches[ matchInd ].queryIdx ].size;
	  double scRad = radEt * Scale;
	  Point nCen( keypoints2[ matches[ matchInd ].trainIdx ].pt.x
		  + scRad * cos( (float)CV_PI / 180.f * scAngle ),
		  keypoints2[ matches[ matchInd ].trainIdx ].pt.y +
		  scRad * sin( (float)CV_PI / 180.f * scAngle ) );
	  Point keyP( keypoints2[ matches[ matchInd ].trainIdx ].pt.x,
		  keypoints2[ matches[ matchInd ].trainIdx ].pt.y );
	  //line( scene, keyP,  nCen, Scalar(0, 0, 255), 1, CV_AA);
	  hgt.at< int >( nCen.y / splitSet,  nCen.x / splitSet ) += 1;

  }
 
  Point maxP;
  minMaxLoc(hgt, 0, 0, 0, &maxP, Mat());
  Point resultCenter(maxP.x*splitSet, maxP.y*splitSet);
  namedWindow("result", 0);
  drawX(resultCenter, scene);
  imshow("result", scene);
  waitKey(0);
  vector<Point2f> objP;
  vector<Point2f> sceneP;

  for ( int i = 0; i <  matches.size(); ++i )
  {
    
    objP.push_back( keypoints1[ matches[i].queryIdx ].pt );
    sceneP.push_back( keypoints2[ matches[i].trainIdx ].pt );
  }

  Mat H = findHomography( objP, sceneP, CV_RANSAC );
  Mat Out( scene.rows , scene.cols , CV_8UC1 );
  warpPerspective( image, Out, H, Size( Out.cols, Out.rows ) , INTER_NEAREST,
	  BORDER_REPLICATE);
  imshow("PerspectTrans", Out);
  waitKey(0);
  return 0;
}

