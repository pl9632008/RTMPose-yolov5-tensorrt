#include <fstream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <ctime>
#include "NvInfer.h"
#include "cuda_runtime_api.h"

using namespace nvinfer1;
using namespace std;
using namespace cv;

float data[3*256*192];
float simccX[17*384];
float simccY[17*512];
float yoloin[3*640*640];
float yoloout[25200*85];

struct KEYPOINT{
    cv::Point2f p;
    float prob;
};

struct  Object{
  Rect_<float> rect;
  int label;
  float prob;
};

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

void qsort_descent_inplace(vector<Object>&faceobjects,int left, int right){
    int i = left;
    int j = right;
    float p = faceobjects[(left+right)/2].prob;
    while (i<=j){
        while (faceobjects[i].prob>p ){
            i++;
        }
        while (faceobjects[j].prob<p){
            j--;
        }
        if(i<=j){
            swap(faceobjects[i],faceobjects[j]);
            i++;
            j--;
        }

    }
#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void  qsort_descent_inplace(vector<Object>&faceobjects){
    if(faceobjects.empty()){
        return ;
    }
    qsort_descent_inplace(faceobjects,0,faceobjects.size()-1);
}

float intersection_area(Object & a,Object&b) {
    Rect2f inter = a.rect&b.rect;
    return inter.area();

}

void nms_sorted_bboxes(std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
         Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
          Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

Mat preprocess_img(cv::Mat& img, int input_w, int input_h,int & padw,int& padh) {
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows))); 
    padw = (input_w - w)/2;
    padh = (input_h - h)/2;
    return out;
}

static float vector_2d_angle(cv::Point p1,cv::Point p2) {
  float angle = 0.0;
  float radian_value = acos((p1.x*p2.x+p1.y*p2.y)/(sqrt(p1.x*p1.x+p1.y*p1.y)*sqrt(p2.x*p2.x+p2.y*p2.y)));
  angle = 180*radian_value/3.1415;
  return angle;
 
}

static void draw_objects(const cv::Mat& image, const std::vector<vector<KEYPOINT>>& ans)
{
    
    static const int joint_pairs[16][2] = {
        {0, 1}, {1, 3}, {0, 2}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}
    };
     static const unsigned char colors[81][3] = {
        {56, 0, 255},
        {226, 255, 0},
        {0, 94, 255},
        {0, 37, 255},
        {0, 255, 94},
        {255, 226, 0},
        {0, 18, 255},
        {255, 151, 0},
        {170, 0, 255},
        {0, 255, 56},
        {255, 0, 75},
        {0, 75, 255},
        {0, 255, 169},
        {255, 0, 207},
        {75, 255, 0},
        {207, 0, 255},
        {37, 0, 255},
        {0, 207, 255},
        {94, 0, 255},
        {0, 255, 113},
        {255, 18, 0},
        {255, 0, 56},
        {18, 0, 255},
        {0, 255, 226},
        {170, 255, 0},
        {255, 0, 245},
        {151, 255, 0},
        {132, 255, 0},
        {75, 0, 255},
        {151, 0, 255},
        {0, 151, 255},
        {132, 0, 255},
        {0, 255, 245},
        {255, 132, 0},
        {226, 0, 255},
        {255, 37, 0},
        {207, 255, 0},
        {0, 255, 207},
        {94, 255, 0},
        {0, 226, 255},
        {56, 255, 0},
        {255, 94, 0},
        {255, 113, 0},
        {0, 132, 255},
        {255, 0, 132},
        {255, 170, 0},
        {255, 0, 188},
        {113, 255, 0},
        {245, 0, 255},
        {113, 0, 255},
        {255, 188, 0},
        {0, 113, 255},
        {255, 0, 0},
        {0, 56, 255},
        {255, 0, 113},
        {0, 255, 188},
        {255, 0, 94},
        {255, 0, 18},
        {18, 255, 0},
        {0, 255, 132},
        {0, 188, 255},
        {0, 245, 255},
        {0, 169, 255},
        {37, 255, 0},
        {255, 0, 151},
        {188, 0, 255},
        {0, 255, 37},
        {0, 255, 0},
        {255, 0, 170},
        {255, 0, 37},
        {255, 75, 0},
        {0, 0, 255},
        {255, 207, 0},
        {255, 0, 226},
        {255, 245, 0},
        {188, 255, 0},
        {0, 255, 18},
        {0, 255, 75},
        {0, 255, 151},
        {255, 56, 0},
        {245, 255, 0}
    };

    srand((unsigned)time(NULL));
   
    for(int idx = 0 ; idx < ans.size(); idx ++){

        const unsigned char* color = colors[rand() % 81];
        vector<KEYPOINT>keypoints = ans[idx];

            for (int i = 0; i < 16; i++)
            {
                const KEYPOINT& p1 = keypoints[joint_pairs[i][0]];
                const KEYPOINT& p2 = keypoints[joint_pairs[i][1]];

                if (p1.prob < 0.3f || p2.prob < 0.3f)
                    continue;

                cv::line(image, p1.p, p2.p, cv::Scalar(color[0], color[1], color[2]), 2);
            }

            // draw joint
            color = colors[rand() % 81];

            for (size_t i = 0; i < keypoints.size(); i++)
            {
                const KEYPOINT& keypoint = keypoints[i];

                fprintf(stderr, "%.2f %.2f = %.5f\n", keypoint.p.x, keypoint.p.y, keypoint.prob);

                if (keypoint.prob < 0.3f)
                    continue;

                cv::circle(image, keypoint.p, 3, cv::Scalar(color[0],color[1],color[2]), -1);
            }

            //calculate angle
            // auto youshou_youzhou = keypoints[10].p  -keypoints[8].p;
            // auto youzhou_youjian = keypoints[6].p - keypoints[8].p;
            // auto youzhou = keypoints[8].p;
            // float ang = vector_2d_angle( youshou_youzhou,youzhou_youjian);

            // String text = to_string((int)ang)+ "^o";
            // int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
            // double fontScale = 1;
            // int thickness = 1;
            // putText(image, text, youzhou, fontFace, fontScale,
            // Scalar(188,0,255), thickness, 1);
        
            cout<<"-------------------------------"<<endl;
    }

}


int main(int argc,char ** argv){

        size_t size{0};
        char * trtModelStream{nullptr};
        ifstream file("/wangjiadong/yolov5/yolov5x.engine", ios::binary);
        
        if(file.good()){
            file.seekg(0,ios::end);
            size = file.tellg();
            file.seekg(0,ios::beg);
            trtModelStream = new char[size];
            file.read(trtModelStream,size);
            file.close();
        }
    
        IRuntime * runtime = createInferRuntime(logger);
        ICudaEngine * engine = runtime->deserializeCudaEngine(trtModelStream,size);
        IExecutionContext *context = engine->createExecutionContext();
        delete[] trtModelStream;

        int BATCH_SIZE=1;
        int INPUT_H=640;
        int INPUT_W=640;

        const char * images = "images";
        const char * output0 = "output0";

        int32_t images_index = engine->getBindingIndex(images);
        int32_t output0_index = engine->getBindingIndex(output0);

        cout<<images_index<<" "
            <<output0_index<<" "
            <<endl;
        cout<<engine->getNbBindings()<<endl;
      
        void * buffers[2];
        cudaMalloc(&buffers[images_index],BATCH_SIZE*3*INPUT_W*INPUT_H*sizeof(float));
        cudaMalloc(&buffers[output0_index],BATCH_SIZE*25200*85*sizeof(float));

        Mat org_img = imread(argv[1]);

        int padw,padh;
        Mat pr_img = preprocess_img(org_img,INPUT_W,INPUT_H,padw,padh);

        for(int i = 0 ; i < INPUT_W*INPUT_H;i++){
            yoloin[i] = pr_img.at<Vec3b>(i)[2]/255.0;
            yoloin[i+INPUT_W*INPUT_H] = pr_img.at<Vec3b>(i)[1]/255.0;
            yoloin[i+2*INPUT_W*INPUT_H]=pr_img.at<Vec3b>(i)[0]/255.0;
        }

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaMemcpyAsync(buffers[images_index],yoloin,BATCH_SIZE*3*INPUT_W*INPUT_H*sizeof(float),cudaMemcpyHostToDevice,stream);
        context->enqueueV2(buffers,stream, nullptr);
        cudaMemcpyAsync(yoloout,buffers[output0_index],1*25200*85*sizeof(float),cudaMemcpyDeviceToHost,stream);

        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        cudaFree(buffers[images_index]);
        cudaFree(buffers[output0_index]);

        delete context;
        delete runtime;
        delete engine;

        vector<Object> objects;
       
        for(int i = 0 ; i<25200;++i){
            if(yoloout[85*i+4]>0.5){
            
                int l,r,t,b;
                float r_w = INPUT_W/(org_img.cols*1.0);
                float r_h = INPUT_H/(org_img.rows*1.0);

                float x = yoloout[85*i+0];
                float y = yoloout[85*i+1];
                float w = yoloout[85*i+2];
                float h = yoloout[85*i+3];
                float score = yoloout[85*i+4];

                if(r_h>r_w){
                    l = x-w/2.0;
                    r = x+w/2.0;
                    t = y-h/2.0-(INPUT_H-r_w*org_img.rows)/2;
                    b = y+h/2.0-(INPUT_H-r_w*org_img.rows)/2;
                    l=l/r_w;
                    r=r/r_w;
                    t=t/r_w;
                    b=b/r_w;
                }else{
                    l = x-w/2.0-(INPUT_W-r_h*org_img.cols)/2;
                    r = x+w/2.0-(INPUT_W-r_h*org_img.cols)/2;
                    t = y-h/2.0;
                    b = y+h/2.0;
                    l=l/r_h;
                    r=r/r_h;
                    t=t/r_h;
                    b=b/r_h;
                }
                int label_index = max_element(yoloout+85*i+5,yoloout+85*(i+1)) - (yoloout+85*i+5);
                if(label_index==0){
                    Object obj;
                    obj.rect.x = l;
                    obj.rect.y = t;
                    obj.rect.width=r-l;
                    obj.rect.height=b-t;
                    obj.label = label_index;
                    obj.prob = score;
                    objects.push_back(obj);
                }
      
            }

        }

        qsort_descent_inplace(objects);
        vector<int> picked;
        nms_sorted_bboxes(objects,picked,0.45);
        int count = picked.size();
        cout<<"count="<<count<<endl;
        vector<Object>obj_out(count);
        for(int i = 0 ; i <count ; ++i){
            obj_out[i] = objects[picked[i]];
    }

//------------------------RTMPose----------------------------------
        size_t size2{0};
        char * trtModelStream2{nullptr};
        ifstream file2("/wangjiadong/mmdeploy-main/workdir/RTMPose.engine", ios::binary);
        
        if(file2.good()){
            file2.seekg(0,ios::end);
            size2 = file2.tellg();
            file2.seekg(0,ios::beg);
            trtModelStream2 = new char[size2];
            file2.read(trtModelStream2,size2);
            file2.close();
        }
    
        IRuntime * runtime2 = createInferRuntime(logger);
        ICudaEngine * engine2 = runtime2->deserializeCudaEngine(trtModelStream2,size2);
        IExecutionContext *context2 = engine2->createExecutionContext();
        delete[] trtModelStream2;

        int BATCH_SIZE2=1;
        int INPUT_H2=256;
        int INPUT_W2=192;

        const char * input = "input";
        const char * simcc_x = "simcc_x";
        const char * simcc_y = "simcc_y";

        int32_t input_index = engine2->getBindingIndex(input);
        int32_t simcc_x_index = engine2->getBindingIndex(simcc_x);    
        int32_t simcc_y_index = engine2->getBindingIndex(simcc_y);

        cout<<input_index<<" "
            <<simcc_x_index<<" "
            <<simcc_y_index<<" "
            <<endl;
        cout<<engine2->getNbBindings()<<endl;

        vector<vector<KEYPOINT>>ans;

        for(int idx = 0 ; idx < obj_out.size(); idx++){
    
            void * buffers2[3];
            cudaMalloc(&buffers2[input_index],BATCH_SIZE2*3*INPUT_W2*INPUT_H2*sizeof(float));
            cudaMalloc(&buffers2[simcc_x_index],BATCH_SIZE2*17*384*sizeof(float));
            cudaMalloc(&buffers2[simcc_y_index],BATCH_SIZE2*17*512*sizeof(float));

            Object &obj = obj_out[idx];
            obj.rect.x = max((int)obj.rect.x,0);
            obj.rect.y = max((int)obj.rect.y,0);
            if(obj.rect.x+obj.rect.width>org_img.cols){
                obj.rect.width = org_img.cols - obj.rect.x;

            }
            if(obj.rect.y+obj.rect.height>org_img.rows){
                obj.rect.height = org_img.rows - obj.rect.y;
            }

            Mat img = org_img(obj.rect);

            int padw2,padh2;
            Mat pr_img2 = preprocess_img(img,INPUT_W2,INPUT_H2,padw2,padh2);

            for(int i = 0 ; i < INPUT_W2*INPUT_H2;i++){
                data[i] = pr_img2.at<Vec3b>(i)[2]/255.0;
                data[i+INPUT_W2*INPUT_H2] = pr_img2.at<Vec3b>(i)[1]/255.0;
                data[i+2*INPUT_W2*INPUT_H2]=pr_img2.at<Vec3b>(i)[0]/255.0;
            }

            cudaStream_t stream2;
            cudaStreamCreate(&stream2);
            cudaMemcpyAsync(buffers2[input_index],data,BATCH_SIZE2*3*INPUT_W2*INPUT_H2*sizeof(float),cudaMemcpyHostToDevice,stream2);
            context2->enqueueV2(buffers2,stream2, nullptr);
            cudaMemcpyAsync(simccX,buffers2[simcc_x_index],BATCH_SIZE2*17*384*sizeof(float),cudaMemcpyDeviceToHost,stream2);
            cudaMemcpyAsync(simccY,buffers2[simcc_y_index],BATCH_SIZE2*17*512*sizeof(float),cudaMemcpyDeviceToHost,stream2);

            cudaStreamSynchronize(stream2);
            cudaStreamDestroy(stream2);
            cudaFree(buffers2[input_index]);
            cudaFree(buffers2[simcc_x_index]);
            cudaFree(buffers2[simcc_y_index]);

            float r_w = INPUT_W2/(img.cols*1.0);
            float r_h = INPUT_H2/(img.rows*1.0);

            vector<KEYPOINT> keypoints;
            for(int c = 0 ; c < 17 ; c++){
                auto x_index = max_element(simccX+c*384, simccX+(c+1)*384) - (simccX+c*384);
                auto y_index = max_element(simccY+c*512, simccY+(c+1)*512) - (simccY+c*512);
                auto prob = max(  *max_element(simccX+c*384, simccX+(c+1)*384) ,  *max_element(simccY+c*512, simccY+(c+1)*512)    );
                KEYPOINT keypoint;
                x_index/=2;
                y_index/=2;

                if(r_h>r_w){
                    x_index = x_index/r_w;
                    y_index = (y_index - ( INPUT_H2 - r_w * img.rows )/2 )/r_w; 
                }else{

                    x_index = (x_index - ( INPUT_W2 - r_h * img.cols )/2 )/r_h;
                    y_index = y_index/r_h;
            }

            keypoint.p = cv::Point2f(x_index +obj.rect.x  ,y_index + obj.rect.y);
            keypoint.prob = prob;
            keypoints.push_back(keypoint);
        }
        ans.push_back(keypoints);
    
        }

        draw_objects(org_img,ans);
        imwrite("./testout.jpg",org_img);
        delete context2;
        delete runtime2;
        delete engine2;
    
  }
