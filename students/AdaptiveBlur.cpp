// AdaptiveBlur.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"
#include <iostream>
#include <string>
#include <conio.h> 

#include <opencv2\opencv.hpp>



using namespace cv;
using namespace std;
void BlurChannel( Mat &chImg, Mat &chInt, Mat &DistMap, short kern )//адаптивное размытие по одному каналу
{
    int dist = 0.0;
    float res = 0.0;
    for( int i = kern + 1; i < chImg.rows - kern; ++i )
    {   
        for( int j = kern + 1; j < chImg.cols - kern; ++j )
        {        
            dist =  DistMap.at< float >( i, j );            //считвыаем половину размера ядра из карты
            if( dist > kern )                               //расстояний
            {
                dist = kern; 
            }          
            res = 0.0;
            res += chInt.at< float >( i + dist + 1, j + dist + 1 );  //вычисляем среднюю яркость по каналу
            res -= chInt.at< float >( i + dist + 1, j - dist - 1 );  //по интегральной матрице
            res -= chInt.at< float >( i - dist - 1, j + dist + 1 );
            res += chInt.at< float >( i - dist - 1, j - dist - 1 );
            res /= (     4*(  dist+1  ) *  (  dist+1  ) );
            chImg.at< uchar >( i, j ) = ( uchar ) res;
        }      
    }
}
void MirrorBord( Mat & i_img, Mat & mirBrd, short kern ) //зеркально отражаем края изображения по одному каналу
{
    for( int i = 0; i < i_img.rows; ++i )
    {
        for( int j = 0; j < i_img.cols; ++j )
        {
            mirBrd.at< uchar >( i + kern - 1, j + kern - 1 ) = i_img.at< uchar >( i, j ); 
        }
        
    }
    
    for( int i = 0; i < i_img.rows; ++i )
    {
        for( int j = 0; j < kern; ++j )
        {
            mirBrd.at< uchar >( i + kern, kern - 1 - j ) = i_img.at< uchar >( i, j );
            mirBrd.at< uchar >( i + kern, i_img.cols + j - 1 + kern) = i_img.at< uchar >( i, i_img.cols - j - 1 );
        }
    }
    for( int j = 0; j < i_img.cols; ++j )
    {
        for( int i = 0; i < kern; ++i )
        {
            mirBrd.at< uchar >( kern - 1 - i, j  +  kern ) = i_img.at< uchar >( i, j );
            mirBrd.at< uchar >( i_img.rows + i - 1 + kern, j + kern ) = i_img.at< uchar >( i_img.rows - i - 1, j );
        }
    }
    for( int i = 0; i < kern; ++i )
    {
       for( int j = 0; j < kern; ++j )
       {
          mirBrd.at< uchar >(  kern - 1 - i, j  ) = i_img.at< uchar >( i + kern - 1, j );
          mirBrd.at< uchar >( i_img.rows + i - 1 + kern, j  ) = i_img.at< uchar >( i_img.rows - i  - 1, j );
          mirBrd.at< uchar >(  kern - 1 - i, j + i_img.cols + kern ) =
              i_img.at< uchar >( i + kern - 1, j + kern );
          mirBrd.at< uchar >( i_img.rows + i - 1 + kern, j + i_img.cols + kern  ) =
              i_img.at< uchar >( i_img.rows - i  - 1, j + kern );
       }
    }
}
int main(int argc, char* argv[])
{ 
   
    Mat i_img;
    short kern = 20;               //размер половины ядра по умолчанию
    
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
    if( argc > 2 )
    {
       kern = atoi( argv[ 2 ] );  //2-ой параметр - размер половины ядра размытия
    }
    //imshow( "Картинка", i_img );
    
   
    vector< Mat > mirrorBord;    
    mirrorBord.push_back( Mat( i_img.rows + 2 * kern , i_img.cols + 2 * kern , CV_8UC1 ) ); 
    mirrorBord.push_back( Mat( i_img.rows + 2 * kern , i_img.cols + 2 * kern , CV_8UC1 ) ); 
    mirrorBord.push_back( Mat( i_img.rows + 2 * kern, i_img.cols + 2 * kern , CV_8UC1 ) ); 
    vector< Mat > channelsImg;
    channelsImg.push_back( Mat( i_img.rows , i_img.cols , CV_8UC1 ) ); 
    channelsImg.push_back( Mat( i_img.rows , i_img.cols , CV_8UC1 ) ); 
    channelsImg.push_back( Mat( i_img.rows , i_img.cols , CV_8UC1 ) ); 
    split( i_img, channelsImg );                             //разбиваем на каналы
    MirrorBord( channelsImg[ 2 ], mirrorBord[ 2 ], kern );  //отражаем края картинки
    MirrorBord( channelsImg[ 1 ], mirrorBord[ 1 ], kern );
    MirrorBord( channelsImg[ 0 ], mirrorBord[ 0 ], kern );
    Mat Img( i_img.rows + 2 * kern, i_img.cols + 2 * kern, CV_8UC3 );
    merge( mirrorBord, Img );                             //сливаем каналы
    Mat Integral( Img.rows , Img.cols , CV_32F );
    integral( Img, Integral, CV_32F );                   //вычисляем интеграл по всем каналам
    
    vector< Mat > channelsInt;
    channelsInt.push_back( Mat( Img.rows , Img.cols , CV_32F ) ); 
    channelsInt.push_back( Mat( Img.rows , Img.cols , CV_32F ) ); 
    channelsInt.push_back( Mat( Img.rows , Img.cols , CV_32F ) ); 
    split( Integral, channelsInt );                         //разбиваем интеграл на каналы для размытия каждого канала отдельно
    Mat MonoChr1( Img.rows , Img.cols , CV_8UC1 );

    cvtColor( Img, MonoChr1, COLOR_BGR2GRAY, 1 ); //обесцвечиваем картинку для вычисления ребер
    //imshow( "Картинка", MonoChr );
    Mat Edges( Img.rows, Img.cols, CV_8UC1 );
    Canny( MonoChr1, Edges, 100, 240, 3, true );  //детектор ребер
    //imshow( "Картинка", Edges );
    
    for( int i = 0; i < Edges.rows; ++i )
    {
        uchar *ptr = Edges.ptr< uchar >( i );
        for( int j = 0; j < Edges.cols; ++j, ++ptr )
        {
            ptr[ 0 ] = 255 - ptr[ 0 ];    //инвертируем изображение границ для вычисления карты расстояний
        }
    }
    
    Mat DistMap( Img.rows , Img.cols , CV_32F );
    
    distanceTransform( Edges, DistMap,  CV_DIST_L1, CV_DIST_MASK_5 );  //вычисляем карту расстояний
    //imshow( "Картинка", DistMap );
    BlurChannel( mirrorBord[ 2 ], channelsInt[ 2 ], DistMap, kern ); //размываем каждый канал в соответствии с картой
    BlurChannel( mirrorBord[ 1 ], channelsInt[ 1 ], DistMap, kern );
    BlurChannel( mirrorBord[ 0 ], channelsInt[ 0 ], DistMap, kern );
    Mat o_img( Img.rows, Img.rows, CV_8UC3 ); 
    merge( mirrorBord, o_img );          //сливаем размытые каналы
    
    Rect r( kern, kern,  i_img.cols  - 1, i_img.rows  - 1 );
    Mat res( i_img.rows , i_img.cols , CV_8UC3 );
    o_img( r ).copyTo( res );           //вырезаем изображение без отражения границ
    
    imshow( "Картинка", res );         //конечная картинка
    
    waitKey();
	return 0;
}

