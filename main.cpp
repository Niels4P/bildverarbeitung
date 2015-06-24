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

inline double log2(double n)
{
	return log(n)/log(2.);
}

/**
 * Very basic dogFeature structure
 */
struct dogFeature
{
    float x;    //y-coord
    float y;    //x-coord
    float s;    //scale (sigma)
    float m;    //magnitude of DoG
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

bool subpixel(const std::vector<vigra::MultiArray<2, float> > & dog, int i, int x, int y, float threshold, float ratio, float height, float width, float& off_x, float& off_y, float& off_s)
{
		using namespace vigra;
		using namespace vigra::linalg;

		int s = i-2;
		float dx = (dog[s](x+1,y)-dog[s](x-1,y))/2;
		float dy = (dog[s](x,y+1)-dog[s](x,y-1))/2;
		float ds = (dog[s+1](x,y)-dog[s-1](x,y))/2;


		if (sqrt(dog[s](x,y)+(pow(dx,2)+pow(dy,2)+pow(ds,2))*1.5f)<threshold*256)
		{
			return false;
		}

		float d2 = 2*dog[s](x,y);
		float dxx = dog[s](x+1,y)+dog[s](x-1,y)-d2;
		float dyy = dog[s](x,y+1)+dog[s](x,y-1)-d2;
		float dss = dog[s+1](x,y)+dog[s-1](x,y)-d2;

		float dxy = (dog[s](x+1,y+1)-dog[s](x-1,y+1)-dog[s](x+1,y-1)+dog[s](x-1,y-1))/2;
		float dxs = (dog[s+1](x+1,y)-dog[s+1](x-1,y)-dog[s-1](x+1,y)+dog[s-1](x-1,y))/2;
		float dys = (dog[s+1](x,y+1)-dog[s+1](x,y-1)-dog[s-1](x,y+1)+dog[s-1](x,y-1))/2;

		float H_data[] = {
			 dxx, dxy, dxs,
			 dxy, dyy, dys,
			 dxs, dys, dss
		};

		float D_data[] = {
			dx,
			dy,
			ds
		};

		Matrix<float> H(Shape2(3,3), H_data);
		Matrix<float> D(Shape2(3,1), D_data);
		Matrix<float> offset(Shape2(3,1));

		float soe = pow(trace(H),2)/determinant(H);
		
		if (soe>=(pow((ratio+1),2))/ratio){
			return false;
		}
		
		bool solution = linearSolve(H, D, offset);
		if (solution){
			off_x = offset(0,0);
			off_y = offset(1,0);
			off_s = offset(2,0);

			if ((x == 1 && off_x >= 0.5f) || (x == width && off_x <= -0.5f))
				off_x = 0.0f;
			if ((y == 1 && off_y >= 0.5f) || (y == height && off_y <= -0.5f))
				off_y = 0.0f;

			return true;
		}
		return false;


		//TODO:
		//offset nicht größer als 0,5: wie sieht das bei sigma aus?
		//negative Koordinaten?! siehe Subpixel- und Rechteckbilder
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
        int intervals = s+3;
        
        //Data containers:
        vector<MultiArray<2, float> > octave(intervals);
        vector<MultiArray<2, float> > dog(intervals-1);
        list<dogFeature> dogFeatures;
        
        //initialise first octave with current (maybe doubled) image
        octave[0] = image;
        
        //Run the loop
        for(int o=0; o<octaves; ++o)
        {
            string file_basename = string("sift_octave") + boost::lexical_cast<string>(o);
            //exportImage(octave[0], file_basename+ + "_level0.png");
            
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
                    
                    gaussianSmoothing(octave[i-1], octave[i], sqrt(total_sigma*total_sigma - last_sigma*last_sigma));
                }
                //Create each interval step using the base image
                else
                {
                    gaussianSmoothing(octave[0], octave[i],  pow(k,(double)i)*sigma);
                }
                
                //Compute the dog
                dog[i-1].reshape(octave[i].shape());
                dog[i-1] = octave[i]-octave[i-1];
                
                //if we have at least three DoGs, we can search for local extrema
                if(i>2)
                {
                    //Determine current sigma of this dog step at this octave:
                    double current_sigma  = o*sigma + pow(k, i-2.0)*sigma;

					//logk(current_sigma/sigma - o) + 2.0 = i
                    
                    for (int y=1; y<dog[i-2].height()-1; ++y)
                    {
                        for (int x=1; x<dog[i-2].width()-1; ++x)
                        {
                            float my_val = dog[i-2](x,y);
                            if ( abs(my_val) > dog_threshold && localExtremum(dog, i,x,y))
                            {
								float off_x, off_y, off_s;
								if (subpixel(dog,i,x,y,threshold,ratio,dog[i-2].height()-1,dog[i-2].width()-1,off_x,off_y,off_s)){

									dogFeature new_feature = { (x-off_x)*pow(2,o+o_offset),
															   (y-off_y)*pow(2,o+o_offset),
															   (current_sigma-off_s),
															   abs(my_val)};
									dogFeatures.push_back(new_feature);
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

		cout << "<svg height=\"" << height << "\" width=\"" << width << "\">\n";
		cout << "<g>\n" << "<image y=\"0.0\" x=\"0.0\" xlink:href=\"" << image_filename << "\" height=\"" << height << "\" width=\"" << width << "\" />\n" << "</g>\n";
		cout << "<g>\n";
		for(const dogFeature& f : dogFeatures)
        {
			cout << "<rect x=\"" << f.x-(f.s/2) << "\" y=\"" << f.y-(f.s/2) << "\" height=\"" << f.s << "\" width=\"" << f.s << "\" stroke=\"yellow\" fill=\"none\" />\n";
        }
		cout << "</g>\n";
		cout << "</svg>"; 


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