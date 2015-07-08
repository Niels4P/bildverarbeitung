#include <iostream>
#include <iomanip>
#include <vector>
#include <list>
#include <string>

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>

#include <vigra/impex.hxx>
#include <vigra/multi_array.hxx>
#include <vigra/convolution.hxx>
#include <vigra/multi_math.hxx>
#include <vigra/matrix.hxx>
#include <vigra/regression.hxx>
#include <vigra/quadprog.hxx>
#include <vigra/gaussians.hxx>
#include <vigra/resampling_convolution.hxx>
#include <vigra/basicgeometry.hxx>

inline double log2(double n)
{
	return log(n)/log(2.);
}

/**
 * Very basic dogFeature structure
 */
struct dogFeature
{
    float x;    //x-coord
    float y;    //y-coord
    float s;    //scale (sigma)
    float m;    //magnitude of DoG
	double a;    //angle
};

/**
 * Inline function to determine if there is a local extremum or not
 * present in the dog stack...
 */
inline bool localExtremum(const std::vector<vigra::MultiArray<2, float> > & dog, int i, int x, int y)
{
    float my_val = dog[i-2](x,y);
    return (    (    my_val < dog[i-2](x-1,y-1) && my_val < dog[i-2](x,y-1) && my_val < dog[i-2](x+1,y-1)
                 &&  my_val < dog[i-2](x-1,y)                               && my_val < dog[i-2](x+1,y)
                 &&  my_val < dog[i-2](x-1,y+1) && my_val < dog[i-2](x,y+1) && my_val < dog[i-2](x+1,y+1)
                 
                 &&  my_val < dog[i-3](x-1,y-1) && my_val < dog[i-3](x,y-1) && my_val < dog[i-3](x+1,y-1)
                 &&  my_val < dog[i-3](x-1,y)   && my_val < dog[i-3](x-1,y) && my_val < dog[i-3](x+1,y)
                 &&  my_val < dog[i-3](x-1,y+1) && my_val < dog[i-3](x,y+1) && my_val < dog[i-3](x+1,y+1)
                 
                 &&  my_val < dog[i-1](x-1,y-1) && my_val < dog[i-1](x,y-1) && my_val < dog[i-1](x+1,y-1)
                 &&  my_val < dog[i-1](x-1,y)   && my_val < dog[i-1](x-1,y) && my_val < dog[i-1](x+1,y)
                 &&  my_val < dog[i-1](x-1,y+1) && my_val < dog[i-1](x,y+1) && my_val < dog[i-1](x+1,y+1))
           
            ||  (    my_val > dog[i-2](x-1,y-1) && my_val > dog[i-2](x,y-1) && my_val > dog[i-2](x+1,y-1)
                 &&  my_val > dog[i-2](x-1,y)                               && my_val > dog[i-2](x+1,y)
                 &&  my_val > dog[i-2](x-1,y+1) && my_val > dog[i-2](x,y+1) && my_val > dog[i-2](x+1,y+1)
                 
                 &&  my_val > dog[i-3](x-1,y-1) && my_val > dog[i-3](x,y-1) && my_val > dog[i-3](x+1,y-1)
                 &&  my_val > dog[i-3](x-1,y)   && my_val > dog[i-3](x-1,y) && my_val > dog[i-3](x+1,y)
                 &&  my_val > dog[i-3](x-1,y+1) && my_val > dog[i-3](x,y+1) && my_val > dog[i-3](x+1,y+1)
                 
                 &&  my_val > dog[i-1](x-1,y-1) && my_val > dog[i-1](x,y-1) && my_val > dog[i-1](x+1,y-1)
                 &&  my_val > dog[i-1](x-1,y)   && my_val > dog[i-1](x-1,y) && my_val > dog[i-1](x+1,y)
                 &&  my_val > dog[i-1](x-1,y+1) && my_val > dog[i-1](x,y+1) && my_val > dog[i-1](x+1,y+1)));
}

/**
 * Inline function to determine the exact angle of orientation
 *
 */
inline void calcParabolaVertex(int x1, int y1, int x2, int y2, int x3, int y3, double& xv) //double& yv
{
	double denom = (x1 - x2) * (x1 - x3) * (x2 - x3);
	double A     = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom;
	double B     = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denom;
	double C     = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom;

	xv = -B / (2*A);
	//yv = C - B*B / (4*A);
}

/**
 * Function to locate extrema on subpixel-level
 * using first derivative and hessian matrix
 */
bool subpixel(const std::vector<vigra::MultiArray<2, float> > & dog, int i, int x, int y, float threshold, float ratio, float height, float width, float& off_x, float& off_y, float& off_s)
{
		using namespace vigra;
		using namespace vigra::linalg;

		int s = i-2;
		// derivative of first order
		float dx = (dog[s](x+1,y)-dog[s](x-1,y))/2;
		float dy = (dog[s](x,y+1)-dog[s](x,y-1))/2;
		float ds = (dog[s+1](x,y)-dog[s-1](x,y))/2;

		// partial derivatives of second order
		float d2 = 2*dog[s](x,y);
		float dxx = dog[s](x+1,y)+dog[s](x-1,y)-d2;
		float dyy = dog[s](x,y+1)+dog[s](x,y-1)-d2;
		float dss = dog[s+1](x,y)+dog[s-1](x,y)-d2;

		float dxy = (dog[s](x+1,y+1)-dog[s](x-1,y+1)-dog[s](x+1,y-1)+dog[s](x-1,y-1))/4;
		float dxs = (dog[s+1](x+1,y)-dog[s+1](x-1,y)-dog[s-1](x+1,y)+dog[s-1](x-1,y))/4;
		float dys = (dog[s+1](x,y+1)-dog[s+1](x,y-1)-dog[s-1](x,y+1)+dog[s-1](x,y-1))/4;

		// hessian matrix
		float H_data[] = {
			 dxx, dxy, dxs,
			 dxy, dyy, dys,
			 dxs, dys, dss
		};

		// vector with derivatives of first order
		float D_data[] = {
			dx,
			dy,
			ds
		};

		Matrix<float> H(Shape2(3,3), H_data);
		Matrix<float> D(Shape2(3,1), D_data);
		Matrix<float> offset(Shape2(3,1));

		// trace of the hessian to the power of 2 divided by the determinant of the hessian
		float soe = pow(trace(H),2)/determinant(H);
		
		//elimination of keypoints with a ratio below the given value between the principal curvatures
		if (soe>=(pow((ratio+1),2))/ratio){
			return false;
		}
		
		// calculation of the offset in every dimension (x,y,sigma)
		bool solution = linearSolve(H, D, offset);
		if (solution){
			// rejection of unstable extrema with low contrast below the threshold
			if ((std::abs(dog[s](x,y)) + dot(D.transpose(),-offset)*0.5) < threshold*256){
				return false;
			}

			off_x = offset(0,0);
			off_y = offset(1,0);
			off_s = offset(2,0);
			
			// no change of offset, if keypoint is at the edge of any dimension
			if ((x == 1 && off_x >= 0.5f) || (x == width && off_x <= -0.5f))
				off_x = 0.0f;
			if ((y == 1 && off_y >= 0.5f) || (y == height && off_y <= -0.5f))
				off_y = 0.0f;
			
			// keypoint will be added
			return true;
		}
		return false;

}

/**
 * Inline function to determine the exact angle of orientation
 *
 */
bool orientation(const std::vector<vigra::MultiArray<2, float> > & octave, int i, int x, int y, double& angle, double& angle2){
	using namespace std;
	using namespace vigra;
	using namespace vigra::multi_math;

	// Gaussian weighted circular window (8 pixel radius)
	Gaussian<double> gauss(8.0/3.0);
	
	vector<double> bin(36);

	//check, if keypoint is too close to the edge of any dimension
	if (x < 8 || x > octave[i].width() - 8)
		return false;

	if (y < 8 || y > octave[i].height() - 8)
		return false;

	// orientation of neighbouring pixel is weighted by the gaussian and its magnitude and added to a histogram (bin-array)
	for(int yo=-7; yo<=8; ++yo){
		for(int xo=-7; xo<=8; ++xo){
			double r = sqrt(xo*xo + yo*yo);
			double weight = gauss(r);

			double magnitude = sqrt(pow(octave[i](x+xo+1,y+yo)-octave[i](x+xo-1,y+yo),2)+pow(octave[i](x+xo,y+yo+1)-octave[i](x+xo,y+yo-1),2));
			int theta = int(((atan2(octave[i](x+xo,y+yo+1)-octave[i](x+xo,y+yo-1), octave[i](x+xo+1,y+yo)-octave[i](x+xo-1,y+yo))+M_PI)/M_PI)*18);

			bin[theta] += magnitude * weight;
		}
	}

	double peak_value = 0.0, peak2_value = 0.0;
	int peak_element = 0, peak2_element = 0;
	bool found2 = false;

	//searching for the peak value in the histogram
	for(int b=0; b<36; ++b){
		if (bin[b]>peak_value){
			peak_value = bin[b];
			peak_element = b;
		}
	}

	//check for a second peak element with at least 80% value of the peak element
	//-> will be added as an extra feature point with exact same coordinates but different angle
	for(int b=0; b<36; ++b){
		if (b==peak_element){
			continue;
		} else if (bin[b]>peak_value*0.8){
			peak2_value = bin[b];
			peak2_element = b;
			found2 = true;
		}
	}

	//neighbours of peak element will be needed for parabola computation of the exact angle
	int n1, n2;
	n1 = peak_element - 1;
	n2 = peak_element + 1;
	if (n1 < 0){
		n1 = 35;
	}
	if (n2 > 35){
		n2 = 0;
	}

	//calculation of exact angle with parabola computation
	//5 degrees is added to every bin to use the mean of every bin
	calcParabolaVertex(n1*10+5,bin[n1],peak_element*10+5,bin[peak_element],n2*10+5,bin[n2],angle);

	//a second peak element was found and needs to be calculated like the first
	if (found2){
		n1 = peak2_element - 1;
		n2 = peak2_element + 1;
		if (n1 < 0){
			n1 = 35;
		}
		if (n2 > 35){
			n2 = 0;
		}
		calcParabolaVertex(n1*10+5,bin[n1],peak2_element*10+5,bin[peak2_element],n2*10+5,bin[n2],angle2);
	}

	return true;

}


/**
 * The main method - will be called at program execution
 */
int main(int argc, char** argv)
{
    using namespace std;
    using namespace vigra;
    using namespace vigra::multi_math;
	using namespace vigra::linalg;
    
    namespace po = boost::program_options;
    
    //Parameter variables
    string image_filename;
    float  sigma,
           dog_threshold;
    bool   double_image_size,
           iterative_interval_creation;
	float  threshold = 0.03f;
	float  ratio = 10.0f;
    
    //Program (argument) options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("image_filename",          po::value<string>(&image_filename)->required(),                 "input image filename")
        ("sigma",                   po::value<float>(&sigma)->default_value(1.0),                   "sigma of scale step")
        ("dog_threshold",           po::value<float>(&dog_threshold)->default_value(5.0),           "keypoint threshold")
        ("double_image_size",       po::value<bool>(&double_image_size)->default_value(true),       "double image size for 0th scale")
        ("iterative_interval_creation",  po::value<bool>(&iterative_interval_creation)->default_value(true),  "use iterative gaussian convolution for each octave interval");
    
    //Additional options for determining the options without name but by order of appearance
    po::positional_options_description p;
        p.add("image_filename",1);
        p.add("sigma",1);
        p.add("dog_threshold",1);
        p.add("double_image_size",1);
    
    //This map will hold the parser results: We don't need it, becaus we directly linked our variables
    //to the program options abouv, (&VARIABLE)...
    po::variables_map vm;
    
    try
    {
        //Store the command line args in the program options
        po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
        
        //and do the parsing
        po::notify(vm);

        //Load the given image
        MultiArray<2, float> image;
        importImage(image_filename, image);

		int width = image.width();
		int height = image.height();
        
        //If we rescale the image (double in each direction), we need to adjust the
        //octave offset - thus resulting DoG positions will be divided by two at the
        //lowermost scale if needed
        int o_offset=0;
        
        if(double_image_size)
        {
            MultiArray<2, float> tmp(image.shape()*2);
            
            resizeImageLinearInterpolation(image, tmp);
            image.reshape(tmp.shape());
            gaussianSmoothing(tmp, image, sigma);
            o_offset=-1;
        }
        
        //Determine the number of Octaves
        int octaves=log2(min(image.width(),image.height()))-3;
        
        //Further parameters
        int         s = 2;
        float       k = pow(2.0,1.0/s);
        const int  intervals = s+3;
        
        //Data containers:
        vector<MultiArray<2, float> > octave(intervals);
        vector<MultiArray<2, float> > dog(intervals-1);
        list<dogFeature> dogFeatures;
        
        //initialise first octave with current (maybe doubled) image
        octave[0] = image;

		//vector to store the sigma-value of every scale
		vector<double> scales;
        
        //Run the loop
        for(int o=0; o<octaves; ++o)
        {
            string file_basename = string("sift_octave") + boost::lexical_cast<string>(o);
            //exportImage(octave[0], file_basename+ + "_level0.png");

			//clear for every octave and add zero element
			scales.clear();
			scales.push_back(0.0);
            
            for (int i=1; i<intervals; ++i)
            {
                //initialize curent interval image array
                octave[i].reshape(octave[i-1].shape());

                //Create each interval step image using the previous one
                if(iterative_interval_creation)
                {
                    // (total_sigma)^2 = sigma^2 + (last_sigma)^2
                    // --> sigma = sqrt((total_sigma)^2 - (last_sigma)^2)!
                    //determine the last sigma
                    double last_sigma  = pow(k, i-1.0)*sigma,
                           total_sigma = last_sigma*k;
                    
					//scales[i] = sqrt(total_sigma*total_sigma - last_sigma*last_sigma);
					scales.push_back(sqrt(total_sigma*total_sigma - last_sigma*last_sigma));
                    gaussianSmoothing(octave[i-1], octave[i], scales[i]);
                }
                //Create each interval step using the base image
                else
                {
					//scales[i] = pow(k,(double)i)*sigma;
					scales.push_back(pow(k,(double)i)*sigma);
                    gaussianSmoothing(octave[0], octave[i], scales[i]);
                }
                
                //Compute the dog
                dog[i-1].reshape(octave[i].shape());
                dog[i-1] = octave[i]-octave[i-1];
                
                //if we have at least three DoGs, we can search for local extrema
                if(i>2)
                {
                    //Determine current sigma of this dog step at this octave:
                    double current_sigma  = o*sigma + pow(k, i-2.0)*sigma;

                    
                    for (int y=1; y<dog[i-2].height()-1; ++y)
                    {
                        for (int x=1; x<dog[i-2].width()-1; ++x)
                        {
                            float my_val = dog[i-2](x,y);
                            if ( abs(my_val) > dog_threshold && localExtremum(dog, i,x,y))
                            {
								//calling subpixel function to determine subpixel-offset of extrema
								float off_x, off_y, off_s;
								if (subpixel(dog,i,x,y,threshold,ratio,dog[i-2].height()-1,dog[i-2].width()-1,off_x,off_y,off_s)){

									int scale = 0;
									double sigscale = 10.0;
									//calculating the scale, where to look for orientation
									//sigma of dog, where the extrema was found, has to be used to find the matching image with similar sigma in octave array
									for(int s=0; s<scales.size(); ++s){
										double check = abs(scales[s]-(current_sigma-off_s));
										if (sigscale > check){
											sigscale = check;
											scale = s;
										}
									}
									
									double angle, angle2;
									bool save;
									//calculation of orientation of keypoint
									//keypoint will be dismissed, if save is false
									//may cause the need to add a second one (if angle2 is given back)
									save = orientation(octave, scale, x, y, angle, angle2);

									//adding keypoint(s) to list of feature keypoints
									if (save) {
										dogFeature new_feature = { (x-off_x)*pow(2,o+o_offset),
																   (y-off_y)*pow(2,o+o_offset),
																   (current_sigma-off_s),
																   abs(my_val),
																   angle};
										dogFeatures.push_back(new_feature);
									
										if (angle2){
											dogFeature new_feature = { (x-off_x)*pow(2,o+o_offset),
																	   (y-off_y)*pow(2,o+o_offset),
																	   (current_sigma-off_s),
																	   abs(my_val),
																	   angle2};
											dogFeatures.push_back(new_feature);
										}
									}
								}
                            }
                        }
                    }
                }
                //exportImage(dog[i-1], file_basename + "_dog" + boost::lexical_cast<string>(i-1) + ".png");
                //exportImage(octave[i], file_basename + "_level" + boost::lexical_cast<string>(i) + ".png");
            }
            
            //rescale for next pyramid step and resize old image (3rd from top)
            octave[0].reshape(octave[0].shape()/2);
            resizeImageNoInterpolation(octave[s], octave[0]);
        }
        /*
        cout << "       x;        y;       s;         m\n";
        for(const dogFeature& f : dogFeatures)
        {
            cout << setw(8) << fixed << setprecision(3) << f.x << "; "
                 << setw(8) << fixed << setprecision(1) << f.y << ";"
                 << setw(8) << fixed << setprecision(1) << f.s << ";"
                 << setw(8) << fixed << setprecision(3) << f.m << "\n";
        }
        cout << "Found: " << dogFeatures.size() << " candidates.\n";
		*/

		//output will be in scalable vector graphics format
		//with link to original image file in same directory
		cout << "<svg height=\"" << height << "\" width=\"" << width << "\">\n";
		cout << "<g>\n" << "<image y=\"0.0\" x=\"0.0\" xlink:href=\"" << image_filename << "\" height=\"" << height << "\" width=\"" << width << "\" />\n" << "</g>\n";
		cout << "<g>\n";
		for(const dogFeature& f : dogFeatures)
        {
			cout << "<rect x=\"" << -(f.s/2) << "\" y=\"" << -(f.s/2) << "\" height=\"" << f.s << "\" width=\"" << f.s << "\" stroke=\"red\" fill=\"none\" transform=\"translate(" << f.x << "," << f.y << ") rotate(" << f.a << ")\" />\n";
        }
		cout << "</g>\n";
		cout << "</svg>"; 
		//transform=\"rotate(" << f.a << "," << f.x-(f.s/2) << "," << f.y-(f.s/2) << ")\"

    }
    catch(po::required_option& e)
    {
        cerr << "Error: " << e.what()
             << std::endl
             << std::endl
             << desc << std::endl;
        return 1;
    }
    catch(po::error& e)
    {
        cerr << "Error: " << e.what()
             << std::endl
             << std::endl
             << desc << std::endl;
        return 1;
    }
    catch(exception & e)
    {
        cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}