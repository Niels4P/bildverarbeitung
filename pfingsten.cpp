#include <iostream>
#include <stdio.h>
#include "vigra/stdimage.hxx"
#include "vigra/convolution.hxx"
#include "vigra/resizeimage.hxx"
#include "vigra/impex.hxx"
using namespace vigra; 
// Gaussian reduction to next pyramid level
template <class Image>
void reduceToNextLevel(Image & in, Image & out, float std_dev)
{    

	// image size at current level
    int width = in.width();
    int height = in.height();
    
    // define a Gaussian kernel (size 5x1)
    vigra::Kernel1D<double> filter;
    //filter.initExplicitly(-2, 2) = 0.05, 0.25, 0.4, 0.25, 0.05;
	filter.initGaussian(std_dev);
    
    vigra::BasicImage<typename Image::value_type> tmpimage1(width, height);
    vigra::BasicImage<typename Image::value_type> tmpimage2(width, height);
    
    // smooth (band limit) input image
    separableConvolveX(srcImageRange(in),
                       destImage(tmpimage1), kernel1d(filter));
    separableConvolveY(srcImageRange(tmpimage1),
		destImage(tmpimage2), kernel1d(filter));
    
	out = tmpimage2;
}
template <class Image>
void reduceToNextScale(Image & in, Image & out)
{

	// image size at current level
    int width = in.width();
    int height = in.height();
    
    // image size at next smaller level
    int newwidth = (width + 1) / 2;
    int newheight = (height + 1) / 2;
    
    // resize result image to appropriate size
    out.resize(newwidth, newheight);

	// downsample smoothed image
    resizeImageNoInterpolation(srcImageRange(in), destImageRange(out));
}

int main(int argc, char ** argv)
{
    if(argc != 7)
    {
        std::cout << "Usage: " << argv[0] << " infile outfile" << std::endl;
        std::cout << "(supported formats: " << vigra::impexListFormats() << ")" << std::endl;

        return 1;
    }
    
    try
    {
        vigra::ImageImportInfo info(argv[1]);

		/*
		std::stringstream ss;
		ss << argv[3];
		int octavenum;
		ss >> octavenum;
		ss.clear();
		ss << argv[4];
		int scalenum;
		ss >> scalenum;
        */

        if(info.isGrayscale())
        {
            vigra::BImage levels[5][5];
			vigra::BImage dog[5][4];
        
            levels[0][0].resize(info.width(), info.height());
           
            importImage(info, destImage(levels[0][0]));

			//5
			for(int i=0; i<5; ++i)
			{

				// Gau�sche Unsch�rfe (Original + 4) //5
				for(int j=1; j<5; ++j)
				{
					// reduce gray image 5 times
					reduceToNextLevel(levels[i][j-1], levels[i][j], 1);
				}
            
				// Reduktion der Aufl�sung (Original + 4)
				if (i < 4) {
					reduceToNextScale(levels[i][4], levels[i+1][0]);
					std::cout << "Bildreduktion Durchlauf: " << i << std::endl;
				}

			}

			//exportImage(srcImageRange(levels[0][3]), vigra::ImageExportInfo(argv[2]));
			//exportImage(srcImageRange(levels[0][4]), vigra::ImageExportInfo(argv[3]));

			// Difference of Gaussian
			//5
			for(int g=0; g<5; ++g)
			{
				std::cout << "Bildsubtraktion Durchlauf: " << g << std::endl;
				//5
				for(int h=1; h<5; ++h)
				{

					dog[g][h-1].resize(levels[g][h-1].size());
					//dog[g][h-1] = levels[g][h-1] - levels[g][h];

					// create image iterator that points to upper left corner 
					// of source image
					vigra::BImage::Iterator s0y = levels[g][h-1].upperLeft();
					vigra::BImage::Iterator s1y = levels[g][h].upperLeft();
            
					// create image iterator that points past the lower right corner of
					// source image (similarly to the past-the-end iterator in the STL)
					vigra::BImage::Iterator send = levels[g][h-1].lowerRight();
            
					// create image iterator that points to upper left corner 
					// of destination image
					vigra::BImage::Iterator dy = dog[g][h-1].upperLeft();
            
					// iterate down the first column of the images
					for(; s0y.y != send.y; ++s0y.y, ++s1y.y, ++dy.y)
					{
						// create image iterator that points to the first 
						// pixel of the current row of the source image
						vigra::BImage::Iterator s0x = s0y;
						vigra::BImage::Iterator s1x = s1y;
						// create image iterator that points to the first 
						// pixel of the current row of the destination image
						vigra::BImage::Iterator dx = dy;
                
						// iterate across current row
						for(; s0x.x != send.x; ++s0x.x, ++s1x.x, ++dx.x)
						{
							// calculate negative gray value
							// *20 f�r Sichtbarkeit im Bild
							*dx = std::abs(*s0x - *s1x);
						}
					}

					std::cout << "Bildreduktion innerer Durchlauf: " << h << std::endl;

				}
			}


			// finding extrema:

			// for loop �ber skalen
				//h�he und breite der dogs einer skala auslesen
				//for loop �ber dogs aus skala
					//iteration �ber pixel aus einem dog
						//Liste der benachbarten zu �berpr�fenden pixel erstellen
						//Liste mit ausgew�hltem Pixel vergleichen
						//falls Werte gr��er und kleiner gleich Pixel gefunden, Abbruch
			Size2D size;
			for(int k = 0; k<5; ++k) {

				size = dog[k][0].size();
				int width = size.width();
				int height = size.height();

				for(int l=0; l<4; ++l) {
					std::cout << "Suchen nach Extrema, Skala: " << k << " DOG: " << l << std::endl;

					for(int y=0; y<height; ++y) {
						for(int x=0; x<width; ++x) {
							vigra::UInt8 value = dog[k][l](x,y);
							bool min = true;
							bool max = true;

							for(int z=-1; z<=1; ++z){
								if(min==false && max==false){
									break;
								}
								if(z==-1 && l==0){
									continue;
								} else if(z==1 && l==3) {
									continue;
								}

								for(int cy=-1; cy<=1; ++cy){
									if(min==false && max==false){
										break;
									}
									if(y==0 && cy==-1){
										continue;
									} else if(y==height-1 && cy==1){
										continue;
									}
									for(int cx=-1; cx<=1; ++cx){
										if(min==false && max==false){
											break;
										}
										if(x==0 && cx ==-1){
											continue;
										} else if(x==width-1 && cx==1){
											continue;
										}

										if(value<dog[k][l+z](x+cx,y+cy)){
											max = false;
										} else if(value>dog[k][l+z](x+cx,y+cy)){
											min = false;
										}

									}
								}
							}
							if(min^max){
								levels[k][0](x,y) = 255;
								//std::cout << "Extrema found!" << std::endl;
							}
						}
					}
				}
			}

			exportImage(srcImageRange(levels[0][0]), vigra::ImageExportInfo(argv[2]));
			exportImage(srcImageRange(levels[1][0]), vigra::ImageExportInfo(argv[3]));
			exportImage(srcImageRange(levels[2][0]), vigra::ImageExportInfo(argv[4]));
			exportImage(srcImageRange(levels[3][0]), vigra::ImageExportInfo(argv[5]));
			exportImage(srcImageRange(levels[4][0]), vigra::ImageExportInfo(argv[6]));

			
        }
        else
        {
            vigra::BRGBImage levels[6];
        
            levels[0].resize(info.width(), info.height());
           
            importImage(info, destImage(levels[0]));

            for(int i=1; i<5; ++i)
            {
                // reduce color image 5 times
                reduceToNextLevel(levels[i-1], levels[i], 0.5);
            }

			reduceToNextScale(levels[4], levels[5]);
            
            exportImage(srcImageRange(levels[5]), vigra::ImageExportInfo(argv[2]));
        }
    }
    catch (vigra::StdException & e)
    {
        std::cout << e.what() << std::endl;
		int a;
		std::cin >> a;
        return 1;
    }
    
    return 0;
}
