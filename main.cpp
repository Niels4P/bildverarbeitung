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

/**
 * The main method - will be called at program execution
 */
int main(int argc, char** argv)
{
    using namespace std;
    using namespace vigra;
    using namespace vigra::multi_math;
    
    namespace po = boost::program_options;
    
    //Parameter variables
    string image_filename;
    float  sigma,
           dog_threshold;
    bool   double_image_size,
           iterative_interval_creation;
    
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
                    
                    for (int y=1; y<dog[i-2].height()-1; ++y)
                    {
                        for (int x=1; x<dog[i-2].width()-1; ++x)
                        {
                            float my_val = dog[i-2](x,y);
                            if ( abs(my_val) > dog_threshold && localExtremum(dog, i,x,y))
                            {
                                dogFeature new_feature = { x*pow(2,o+o_offset),
                                                           y*pow(2,o+o_offset),
                                                           current_sigma,
                                                           abs(my_val)};
								dogFeatures.push_back(new_feature);
                            }
                        }
                    }
                }
                exportImage(dog[i-1], file_basename + "_dog" + boost::lexical_cast<string>(i-1) + ".png");
                exportImage(octave[i], file_basename + "_level" + boost::lexical_cast<string>(i) + ".png");
            }
            
            //rescale for next pyramid step and resize old image (3rd from top)
            octave[0].reshape(octave[0].shape()/2);
            resizeImageNoInterpolation(octave[s], octave[0]);
        }
        
        cout << "       x;        y;       s;         m\n";
        for(const dogFeature& f : dogFeatures)
        {
            cout << setw(8) << fixed << setprecision(3) << f.x << "; "
                 << setw(8) << fixed << setprecision(1) << f.y << ";"
                 << setw(8) << fixed << setprecision(1) << f.s << ";"
                 << setw(8) << fixed << setprecision(3) << f.m << "\n";
        }
        cout << "Found: " << dogFeatures.size() << " candidates.\n";

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