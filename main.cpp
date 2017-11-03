#include <iostream>
#include <string>
#include <stdlib.h>
#include <signal.h>
#include <math.h>
#include <vector>

#include <boost/program_options.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "types_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/core/types_c.h"
#include "gurobi_c++.h"

#include "graph.h"
#include "callback.h"
   
#define DPRINT(X) std::cerr << #X << ": " << X << std::endl;
#define EPS 0.00001

namespace po = boost::program_options;
namespace b = boost;

cv::Mat src, src_gray;
cv::Mat dst;
cv::Size picture_size;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3; // 1:2 or 1:3
int kernel_size = 3;
const char* window_name = "Edge Map";

std::vector<GRBModel>* models_ref = NULL;
void my_handler(int s) {
  if(models_ref != NULL) {
    for(auto it = models_ref->begin(); it!=models_ref->end(); ++it) {
        it->terminate();
    }
    models_ref = NULL;
  }
  else {
    exit(2);
  }
}
////////////////////////////////////////////////////////////////////////////////
/**
 * Copyright (c) 2016, David Stutz
 * Contact: david.stutz@rwth-aachen.de, davidstutz.de
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
//returns number of merged superpixels with a neighbour
int enforceMinimumSuperpixelSize(const cv::Mat &image, cv::Mat &labels, int size) {
    int max_label = 0;
    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            if (labels.at<int>(i, j) > max_label) {
                max_label = labels.at<int>(i, j);
            }
        }
    }
    
    //std::vector<cv::Vec3b> means(max_label + 1, cv::Vec3b(0, 0, 0));
    std::vector<float> means(max_label + 1, 0.f);
    std::vector<int> counts(max_label + 1, 0);
    
    std::vector< std::vector<int> > neighbors(max_label + 1);
    for (unsigned int k = 0; k < neighbors.size(); k++) {
        neighbors[k] = std::vector<int>(0);
    }
    
    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            int label = labels.at<int>(i, j);
            /*
            means[label][0] += image.at<cv::Vec3b>(i, j)[0];
            means[label][1] += image.at<cv::Vec3b>(i, j)[1];
            means[label][2] += image.at<cv::Vec3b>(i, j)[2];
            */
            means[label] += (float)image.at<uchar>(i, j);
       
            counts[label]++;
           //kinda neighbours 
            int neighbor_label = labels.at<int>(std::min(i + 1, labels.rows - 1), j);
            if (neighbor_label != label) {
                if (std::find(neighbors[label].begin(), neighbors[label].end(), neighbor_label) == std::end(neighbors[label])) {
                    neighbors[label].push_back(neighbor_label);
                }
            }
            
            neighbor_label = labels.at<int>(std::max(0, i - 1), j);
            if (neighbor_label != label) {
                if (std::find(neighbors[label].begin(), neighbors[label].end(), neighbor_label) == std::end(neighbors[label])) {
                    neighbors[label].push_back(neighbor_label);
                }
            }
            
            neighbor_label = labels.at<int>(i, std::min(j + 1, labels.cols - 1));
            if (neighbor_label != label) {
                if (std::find(neighbors[label].begin(), neighbors[label].end(), neighbor_label) == std::end(neighbors[label])) {
                    neighbors[label].push_back(neighbor_label);
                }
            }
            
            neighbor_label = labels.at<int>(i, std::max(0, j - 1));
            if (neighbor_label != label) {
                if (std::find(neighbors[label].begin(), neighbors[label].end(), neighbor_label) == std::end(neighbors[label])) {
                    neighbors[label].push_back(neighbor_label);
                }
            }
        }
    }
    
    for (unsigned int k = 0; k < counts.size(); k++) {
        if (counts[k] > 0) {
           /* means[k][0] /= counts[k];
            means[k][1] /= counts[k];
            means[k][2] /= counts[k];
            */
            means[k] /= counts[k];
        }
    }
    
    int count = 0;
    std::vector<int> new_labels(max_label + 1, -1);
    
    for (unsigned int k = 0; k < counts.size(); k++) {
        if (counts[k] < size) {
            
            float min_distance = std::numeric_limits<float>::max();
            for (unsigned int kk = 0; kk < neighbors[k].size(); kk++) {
            /*  float distance = (means[k][0] - means[neighbors[k][kk]][0])*(means[k][0] - means[neighbors[k][kk]][0])
                        + (means[k][1] - means[neighbors[k][kk]][1])*(means[k][1] - means[neighbors[k][kk]][1])
                        + (means[k][2] - means[neighbors[k][kk]][2])*(means[k][2] - means[neighbors[k][kk]][2]);
            */

                float distance = (means[k] - means[neighbors[k][kk]])*(means[k] - means[neighbors[k][kk]]);
                if (distance < min_distance && new_labels[neighbors[k][kk]] < 0) {
                    min_distance = distance;
                    new_labels[k] = neighbors[k][kk];
                }
            }
          
            count++;
        }
    }
    
    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            
            int label = labels.at<int>(i, j);
            if (new_labels[label] >= 0) {
                
                int new_label = new_labels[label];
                while (new_labels[new_label] >= 0) {
                    new_label = new_labels[new_label];
                }
                
                labels.at<int>(i, j) = new_labels[label];
            }
        }
    }
    
    return count;
}

int enforceMinimumSuperpixelSizeUpTo(const cv::Mat &image, cv::Mat &labels, int number) {
    int max_label = 0;
    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            if (labels.at<int>(i, j) > max_label) {
                max_label = labels.at<int>(i, j);
            }
        }
    }
    
    std::vector<cv::Vec3b> means(max_label + 1, cv::Vec3b(0, 0, 0));
    std::vector<int> counts(max_label + 1, 0);
    
    std::vector< std::vector<int> > neighbors(max_label + 1);
    for (unsigned int k = 0; k < neighbors.size(); k++) {
        neighbors[k] = std::vector<int>(0);
    }
    
    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            int label = labels.at<int>(i, j);
            
            /*
            means[label][0] += image.at<cv::Vec3b>(i, j)[0];
            means[label][1] += image.at<cv::Vec3b>(i, j)[1];
            means[label][2] += image.at<cv::Vec3b>(i, j)[2];
            */
            means[label][0] += image.at<uchar>(i, j);
            means[label][1] += image.at<uchar>(i, j);
            means[label][2] += image.at<uchar>(i, j);
            counts[label]++;
            
            int neighbor_label = labels.at<int>(std::min(i + 1, labels.rows - 1), j);
            if (neighbor_label != label) {
                if (std::find(neighbors[label].begin(), neighbors[label].end(), neighbor_label) == std::end(neighbors[label])) {
                    neighbors[label].push_back(neighbor_label);
                }
            }
            
            neighbor_label = labels.at<int>(std::max(0, i - 1), j);
            if (neighbor_label != label) {
                if (std::find(neighbors[label].begin(), neighbors[label].end(), neighbor_label) == std::end(neighbors[label])) {
                    neighbors[label].push_back(neighbor_label);
                }
            }
            
            neighbor_label = labels.at<int>(i, std::min(j + 1, labels.cols - 1));
            if (neighbor_label != label) {
                if (std::find(neighbors[label].begin(), neighbors[label].end(), neighbor_label) == std::end(neighbors[label])) {
                    neighbors[label].push_back(neighbor_label);
                }
            }
            
            neighbor_label = labels.at<int>(i, std::max(0, j - 1));
            if (neighbor_label != label) {
                if (std::find(neighbors[label].begin(), neighbors[label].end(), neighbor_label) == std::end(neighbors[label])) {
                    neighbors[label].push_back(neighbor_label);
                }
            }
        }
    }
    
    for (unsigned int k = 0; k < counts.size(); k++) {
        if (counts[k] > 0) {
            means[k][0] /= counts[k];
            means[k][1] /= counts[k];
            means[k][2] /= counts[k];
        }
    }
    
    std::vector<int> ids(max_label + 1, 0);
    for (unsigned int k = 0; k < ids.size(); k++) {
        ids[k] = k;
    }
    
    std::sort(ids.begin(), ids.end(), [&counts](int i, int j) {
        return counts[i] < counts[j];
    });
    
//    for (unsigned int k = 0; k < counts.size(); k++) {
//        std::cout << ids[k] << ": " << counts[ids[k]] << std::endl;
//    }
    
    int count = 0;
    std::vector<int> new_labels(max_label + 1, -1);
    
    for (unsigned int k = 0; k < std::min((int) counts.size(), number); k++) {
        int label = ids[k];
        
        float min_distance = std::numeric_limits<float>::max();
        for (unsigned int kk = 0; kk < neighbors[label].size(); kk++) {
            float distance = (means[label][0] - means[neighbors[label][kk]][0])*(means[label][0] - means[neighbors[label][kk]][0])
                    + (means[label][1] - means[neighbors[label][kk]][1])*(means[label][1] - means[neighbors[label][kk]][1])
                    + (means[label][2] - means[neighbors[label][kk]][2])*(means[label][2] - means[neighbors[label][kk]][2]);

            if (distance < min_distance && new_labels[neighbors[label][kk]] < 0) {
                min_distance = distance;
                new_labels[label] = neighbors[label][kk];
            }
        }
        
        count++;
    }
    
    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            
            int label = labels.at<int>(i, j);
            if (new_labels[label] >= 0) {
                
                int new_label = new_labels[label];
                while (new_labels[new_label] >= 0) {
                    new_label = new_labels[new_label];
                }
                std::cout << "r: " << i << " c: " << j <<  " : " << "before "  << labels.at<int>(i, j) << " after " << new_labels[label] << std::endl;
                labels.at<int>(i, j) = new_labels[label];
            }
        }
    }
    
    return count;
}
////////////////////////////////////////////////////////////////////////////////
int count_labels(const cv::Mat& labels) {
    int count;
    std::set<int> set;
    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            set.insert(labels.at<int>(i, j));
        }
    }
    return set.size();
}

//kernel size must be odd
template <typename MatType>
double calculate_global_contrast(const cv::Mat& m, int kernel_size) {
    cv::Mat m2;
    m2 = m.clone();
    cv::boxFilter(m, m2, -1, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    int rows = m.rows;
    int cols = m.cols;
    double tresh = 0.0;
    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            tresh += m2.at<MatType>(r, c)/255.f;
        }
    }
    return tresh / (rows*cols);
}

template <typename MatType>
void count_jumps(const cv::Mat& m, std::vector<int>& rowcount, std::vector<int>& colcount, double tresh) {
    int rows = m.rows;
    int cols = m.cols;
    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols - 1; ++c) {
            if(fabs(m.at<MatType>(r, c+1)/255.f - m.at<MatType>(r, c)/255.f) >= tresh){
                rowcount[r]++;
            }
        }
    }
    for(int c = 0; c < cols; ++c) {
        for(int r = 0; r < rows - 1; ++r) {
            if(fabs(m.at<MatType>(r+1, c)/255.f - m.at<MatType>(r, c)/255.f) >= tresh){
                colcount[c]++;
            }
        }
    }
}
void CannyThreshold(int, void*)
{
    cv::Mat detected_edges;
    /// Reduce noise with a kernel 3x3
    //cv::blur( src_gray, detected_edges, cv::Size(3,3) );
    cv::GaussianBlur( src_gray, detected_edges, cv::Size(3,3), 9.5);

    /// Canny detector
    cv::Canny( detected_edges, detected_edges, 0, lowThreshold*ratio, kernel_size );

    /// Using Canny's output as a mask, we display our result
    dst = cv::Scalar::all(0); // set all pixels of dst to greyscale color 0

    src.copyTo( dst, detected_edges); // copy with detected_edges as mask
    cv::imshow( window_name, dst );
}

void saveMatToCsv(cv::Mat &matrix, std::string filename){
    std::ofstream outputFile(filename);
    outputFile << format(matrix, "CSV");
    outputFile.close();
}

int main(int argc, char** argv) {
    // set the handler
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = (void(*)(int))my_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    //commandline
    po::variables_map vm;
    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("grid-size,gs", po::value<int>()->default_value(264), "set number of grid elements. Only approximate")
            ("input-file", po::value< std::string >(), "picture to process")
            ("output-file,o", po::value< std::string >()->default_value("out.jpg"), "picture to process")
            ("interactive", po::value<bool>()->default_value(true), "interactive mode")
            ("csv", po::value< std::string >()->default_value("out.csv"), "csv file name to write to")
            ("lambda", po::value< double >() ->default_value(0.1), "parameter for multicut criterion")
            ("jumps", po::value< double >() ->default_value(0.05), "the lower the number the more cuts are allowed")
            ("enforce-size", po::value< double >() ->default_value(0.25), "post processing, relative to segment size")
        ;

        po::positional_options_description p;
        p.add("input-file", -1);
        po::store(po::command_line_parser(argc, argv).
          options(desc).positional(p).run(), vm);
        //po::notify(vm);    
        if(vm.count("help")){
            std::cout << desc << std::endl;
            return EXIT_SUCCESS;
        }
        if(!vm.count("input-file")) {
            std::cerr << "no input file given" << std::endl;
            return EXIT_FAILURE;
        }
        
    }
    catch (po::error &ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    //parse image and build graph
    // Load an image
    src = cv::imread( vm["input-file"].as<std::string>(), CV_LOAD_IMAGE_COLOR );

    if( !src.data )
    { 
        std::cerr << "bad input file" << std::endl;
        return EXIT_FAILURE; 
    }
    picture_size = src.size();

    dst.create( src.size(), src.type() );
    cv::cvtColor( src, src_gray, CV_BGR2GRAY );

    /// Create a window
    if(vm["interactive"].as<bool>()) {
        cv::namedWindow( window_name, CV_WINDOW_AUTOSIZE );

        /// Create a Trackbar for user to enter threshold
        cv::createTrackbar( "Low Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );

        /// Show the image
        CannyThreshold(0, 0);
            
        cv::waitKey(0);
    }
    // do some blur
    cv::GaussianBlur( src_gray, src_gray, cv::Size(3,3), 0);


    //build submatrix per grid segment

    double ratio = ((double)picture_size.width)/picture_size.height;
    double root = sqrt(vm["grid-size"].as<int>());
    cv::Size segment_dim(floor(ratio * root), floor((1/ratio) * root));
    int number_of_segments = segment_dim.width*segment_dim.height;
    

    //segments should like this:
    //D-A A A A A-C C C C
    //| | | | | | | | | |
    //B-X X X X-Y Y Y Y Y- ...
    //    S1       S2
    //B-X X X X-Y Y Y Y Y-
    //
    //B-X X X X-Y Y Y Y Y-
    // ...

    cv::Size2f segment_size_approx(picture_size.width / (float)segment_dim.width, picture_size.height / (float)segment_dim.height);
    if(vm["interactive"].as<bool>()) {
        DPRINT(picture_size.width)
        DPRINT(picture_size.height)
        DPRINT(segment_size_approx.width) 
        DPRINT(segment_size_approx.height) 
        DPRINT(number_of_segments)
        DPRINT(segment_dim.width)
        DPRINT(segment_dim.height)
    }

    
    double global_contrast = calculate_global_contrast<uchar>(src_gray, 5);
    if(vm["interactive"].as<bool>()) {
        DPRINT(global_contrast)
    }
    std::vector<Graph*> segment_graphs(number_of_segments);
    //GRBEnv env = GRBEnv();
    std::vector<GRBEnv> envs(number_of_segments); // we need an environment per model
    std::vector<GRBModel> models;
    for(auto it = envs.begin(); it!=envs.end();++it) {
        auto env = *it;
        models.emplace_back(env);
    }
    models_ref = &models;

    int start_row = 0;
    int start_col = 0;

    for(int i = 0; i < segment_dim.height; ++i) {
        cv::Size segment_size;
        segment_size.height = (int)(floor(segment_size_approx.height*(i+1))-floor(segment_size_approx.height*i));
        if(i == segment_dim.height-1) segment_size.height = picture_size.height - start_row;
        
        for(int j = 0; j < segment_dim.width; ++j) {
            segment_size.width = (int)(floor(segment_size_approx.width*(j+1))-floor(segment_size_approx.width*j));
            if(j == segment_dim.width-1) segment_size.width = picture_size.width - start_col;

            int seg_num = xy_to_index(j, i, segment_dim);

            GRBModel& model = models[seg_num];
            // Turn off display and heuristics and enable adding constraints in our callback function
            model.set(GRB_IntParam_OutputFlag, 0); 
            model.set(GRB_DoubleParam_Heuristics, 1);
            model.set(GRB_IntParam_LazyConstraints, 1);
            
            segment_graphs[seg_num]= new Graph(segment_size.width*segment_size.height);
            Graph& g = *segment_graphs[seg_num];
            g[b::graph_bundle].size = segment_size;
            g[b::graph_bundle].row = cv::Range(start_row, start_row+segment_size.height);
            g[b::graph_bundle].col = cv::Range(start_col, start_col+segment_size.width);
            //build_grid
            GRBLinExpr obj1(0.0); // left side of term
            GRBLinExpr obj2(0.0); // right side of term
            GRBLinExpr objective(0.0);
            cv::Mat submatrix = src_gray(g[b::graph_bundle].row, g[b::graph_bundle].col);
            int edge_index_counter = 0;
            for(int y = 0; y < segment_size.height; ++y) {
                for(int x = 0; x < segment_size.width; ++x) {
                    if(x != segment_size.width-1) {
                        int a = xy_to_index(x, y, segment_size);
                        int b = xy_to_index(x+1, y, segment_size);
                        Graph::edge_descriptor e;
                        bool inserted;
                        b::tie(e, inserted) = b::add_edge(a, b, g);
                        g[e].index = edge_index_counter++;
                        g[e].var = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "edge");
                        //obj2 += lambda_row[y]*g[e].var;
                        obj2 += g[e].var;
                        //DPRINT(fabs(submatrix.at<uchar>(y,x)/255.f-submatrix.at<uchar>(y,x+1)/255.f))
                        obj1 += fabs(submatrix.at<uchar>(y,x)/255.f-submatrix.at<uchar>(y,x+1)/255.f)*g[e].var;
                    }
                    if(y != segment_size.height-1) {
                        int a = xy_to_index(x, y, segment_size);
                        int b = xy_to_index(x, y+1, segment_size);
                        Graph::edge_descriptor e;
                        bool inserted;
                        b::tie(e, inserted) = b::add_edge(a, b, g);
                        g[e].index = edge_index_counter++;
                        g[e].var = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "edge");
                        //obj2 += lambda_col[x]*g[e].var;
                        obj2 += g[e].var;
                        obj1 += fabs(submatrix.at<uchar>(y,x)/255.f-submatrix.at<uchar>(y+1,x)/255.f)*g[e].var;
                    }
                    // edge insertion order is up, left, right, down
                }
            }

            obj2 *= vm["lambda"].as<double>()*global_contrast;
            objective = obj1 - obj2;
            model.setObjective(objective, GRB_MAXIMIZE);

            

            //initial constraints
            for(int x = 0; x < segment_size.width-1; ++x) {
                for(int y = 0; y < segment_size.height-1; ++y) {
                    //a---b
                    //|   |
                    //d---c
                    GRBVar& a_b = g[b::edge(xy_to_index(x, y, segment_size), xy_to_index(x+1,y, segment_size), g).first].var;
                    GRBVar& b_c = g[b::edge(xy_to_index(x+1, y, segment_size), xy_to_index(x+1,y+1, segment_size), g).first].var;
                    GRBVar& c_d = g[b::edge(xy_to_index(x, y+1, segment_size), xy_to_index(x+1,y+1, segment_size), g).first].var;
                    GRBVar& d_a = g[b::edge(xy_to_index(x, y+1, segment_size), xy_to_index(x,y, segment_size), g).first].var;
                    model.addConstr(a_b + b_c + c_d >= d_a);
                    model.addConstr(b_c + c_d + d_a >= a_b);
                    model.addConstr(c_d + d_a + a_b >= b_c);
                    model.addConstr(d_a + a_b + b_c >= c_d);
                }
            }
            // the <= k constraints:
            std::vector<int> rowcount(segment_size.height, 0);
            std::vector<int> colcount(segment_size.width, 0);
            count_jumps<uchar>(submatrix, rowcount, colcount, vm["jumps"].as<double>()*global_contrast);
            for(int r = 0; r < segment_size.height; ++r) {
               GRBLinExpr sum(0);
               for(int c = 0; c < segment_size.width-1; ++c) {
                    GRBVar& v = g[b::edge(xy_to_index(c, r, segment_size), xy_to_index(c+1,r, segment_size), g).first].var;
                    sum += v;
               }
               model.addConstr(sum <= rowcount[r]);
            }
            for(int c = 0; c < segment_size.width; ++c) {
               GRBLinExpr sum(0);
               for(int r = 0; r < segment_size.height-1; ++r) {
                    GRBVar& v = g[b::edge(xy_to_index(c, r, segment_size), xy_to_index(c,r+1, segment_size), g).first].var;
                    sum += v;
               }
               model.addConstr(sum <= colcount[c]);
            }
            


            #if 1
            myGRBCallback* cb = new myGRBCallback(g);
            model.setCallback(cb);
            #endif
            #if 1
            model.optimizeasync();
            #endif

            start_col += segment_size.width; 
        }
        start_col = 0;
        start_row += segment_size.height;
    }
    int modelnum = 0;
    while(modelnum < number_of_segments) {
        if(models[modelnum].get(GRB_IntAttr_Status) != GRB_INPROGRESS) modelnum++;
        sleep(0.5);
    }

    // done

    // write labels as csv
    cv::Mat labels;
    labels.create(picture_size, CV_32SC1);
    labels = cv::Scalar(0); 
    int label_offset = 0;

    for(int i = 0; i < number_of_segments; ++i){
        GRBModel& model = models[i];
        Graph& g = *segment_graphs[i];
        cv::Size& segment_size = g[b::graph_bundle].size;
        cv::Mat submatrix = labels(g[b::graph_bundle].row, g[b::graph_bundle].col);
        int max_label = 0;
        for(int x = 0; x < segment_size.width; ++x) {
          for(int y = 0; y < segment_size.height; ++y) {
             int label = g[(Graph::vertex_descriptor)xy_to_index(x,y,segment_size)].multicut_label;
             max_label = std::max(max_label, label); 
             submatrix.at<int>(y, x) = label + label_offset;
          }
        }
        label_offset = label_offset + max_label + 1;//right?
    }
    std::cout << "num labels: " << label_offset << std::endl;
    std::cout << "merging..." << std::endl;
    for(int i = 0; i < 20; ++i) {
    enforceMinimumSuperpixelSize(src_gray, labels, (int)(vm["enforce-size"].as<double>()*segment_size_approx.width*segment_size_approx.height)); 
    }
//  enforceMinimumSuperpixelSizeUpTo(src_gray, labels, 3320);
    // ... maybe just fuse them to segment?
    int count = count_labels(labels);
    std::cout << "num labels after enforcing sp size: " << count << std::endl;
    saveMatToCsv(labels, vm["csv"].as<std::string>());


    // write image
    #if 0
    cv::Mat red;
    cv::Mat border_mask;
    border_mask.create(picture_size, CV_8UC1);
    border_mask = cv::Scalar(0); 

    for(int i = 0; i < number_of_segments; ++i){
        GRBModel& model = models[i];
        Graph& g = *segment_graphs[i];
        cv::Size& segment_size = g[b::graph_bundle].size;
        cv::Mat submatrix = border_mask(g[b::graph_bundle].row, g[b::graph_bundle].col);
        //segment border
        for(int x = 0; x < segment_size.width; ++x) {
            submatrix.at<uchar>(0, x) = 1;
            if(index_to_xy(i, segment_dim).y == segment_dim.height-1) submatrix.at<uchar>(segment_size.height-1, x) = 1;
        }
        for(int y = 0; y < segment_size.height; ++y) {
            submatrix.at<uchar>(y, 0) = 1;
            if(index_to_xy(i, segment_dim).x == segment_dim.width-1) submatrix.at<uchar>(y, segment_size.width-1) = 1;
        }
        #if 1
        //horizontal edges
        for(int x = 0; x < (segment_size.width-1); ++x) {
          for(int y = 0; y < segment_size.height; ++y) {
            if(std::abs(g[b::edge(xy_to_index(x, y, segment_size), xy_to_index(x+1,y,segment_size), g).first].var.get(GRB_DoubleAttr_X) - 1.0) < EPS) {
                submatrix.at<uchar>(y, x) = 1; // left
            }
          }
        }

        //vertical edges
        for(int x = 0; x < segment_size.width; ++x) {
          for(int y = 0; y < (segment_size.height-1); ++y) {
            if(std::abs(g[b::edge(xy_to_index(x, y, segment_size), xy_to_index(x,y+1,segment_size), g).first].var.get(GRB_DoubleAttr_X) - 1.0) < EPS) {
                submatrix.at<uchar>(y+1, x) = 1; // down 
            }
          }
        }
        #endif
    }
    #endif
    //use labels instead:
    cv::Mat red;
    cv::Mat border_mask;
    border_mask.create(picture_size, CV_8UC1);
    border_mask = cv::Scalar(0); 

    for(int r = 0; r < picture_size.height; ++r) {
        for(int c = 0; c < picture_size.width-1; ++c) {
            if(labels.at<int>(r, c) != labels.at<int>(r, c+1)) {
                border_mask.at<uchar>(r, c) = 1;
            }
        }
    }
    for(int r = 0; r < picture_size.height-1; ++r) {
        for(int c = 0; c < picture_size.width; ++c) {
            if(labels.at<int>(r, c) != labels.at<int>(r+1, c)) {
                border_mask.at<uchar>(r, c) = 1;
            }
        }
    }
    red.create(picture_size, CV_8UC3);
    red = cv::Scalar(0, 0, 255);
    cv::cvtColor(src_gray, dst, CV_GRAY2BGR);
    red.copyTo( dst, border_mask ); 
    cv::imwrite( vm["output-file"].as<std::string>(), dst);

    if(vm["interactive"].as<bool>()) {
        cv::imshow( window_name, dst );
        cv::waitKey(0);
    }

    return EXIT_SUCCESS;
}
