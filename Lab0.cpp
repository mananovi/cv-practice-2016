// Lab0.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"
#include <iostream>
#include <string>
#include <conio.h> 

#include <opencv2\opencv.hpp>



using namespace cv;
using namespace std;



int main(int argc, char* argv[])
{ 
   
    Mat i_img;
  
  
    if( argc > 1 )
    {
       i_img = imread(    string( argv[ 1 ] ) , 1 );  //1-й параметр - имя картинки
    }
    else
    {
        cout << "Image file name expected" << endl;
        getch();
        return 0;
    }
  
  
    Mat MonoChr1( i_img.rows, i_img.cols, CV_8UC1 );
    Mat MonoChr2( i_img.rows, i_img.cols, CV_8UC1 );
    Mat o_img1( i_img.rows, i_img.cols, CV_8UC1 );
    cvtColor( i_img, MonoChr1, CV_BGR2GRAY );
    cvtColor( i_img, MonoChr2, CV_BGR2GRAY );
    equalizeHist( MonoChr1, o_img1 );
    imshow( "Image", o_img1 );
	waitKey();


    RNG rng(12345);
    int maxCorners = 100;
    vector<Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;
    Mat copy;
    copy = i_img.clone();
    goodFeaturesToTrack( MonoChr2, corners, maxCorners, qualityLevel,
                       minDistance, Mat(), blockSize, useHarrisDetector, k );
  
    int r = 4;
    for( int i = 0; i < corners.size(); ++i )
    {
        circle( copy, corners[i], r,
            Scalar( rng.uniform( 0, 255 ),
            rng.uniform( 0, 255 ), rng.uniform( 0, 255 ) ), -1, 8, 0 );
    }
    imshow( "Image", copy );
    waitKey();
    
    Mat o_img2( i_img.rows, i_img.cols, CV_8UC3 );
    Mat element = getStructuringElement( MORPH_RECT, Size( 11, 11 ), Point( 5, 5 ) );
    morphologyEx( i_img, o_img2, CV_MOP_OPEN, element );
    imshow( "Image", o_img2 );
    waitKey();

	return 0;
}

