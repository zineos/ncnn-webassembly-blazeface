//
// Created by dl on 19-7-19.
//

#ifndef BLAZEFACE_H
#define BLAZEFACE_H

#include <string>
#include <stack>
#include "net.h"
#include <chrono>

using namespace std::chrono;

class Timer
{
public:
    std::stack<high_resolution_clock::time_point> tictoc_stack;

    void tic()
    {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        tictoc_stack.push(t1);
    }

    double toc(std::string msg = "", bool flag = true)
    {
        double diff = duration_cast<milliseconds>(high_resolution_clock::now() - tictoc_stack.top()).count();
        if(msg.size() > 0){
            if (flag)
                printf("%s time elapsed: %f ms\n", msg.c_str(), diff);
        }

        tictoc_stack.pop();
        return diff;
    }
    void reset()
    {
        tictoc_stack = std::stack<high_resolution_clock::time_point>();
    }
};

struct Point{
    float _x;
    float _y;
};
struct bbox{
    float x1;
    float y1;
    float x2;
    float y2;
    float s;
    Point point[5];
};

struct box{
    float cx;
    float cy;
    float sx;
    float sy;
};

class Detector
{

public:
    Detector();

    void Init(const std::string &model_param, const std::string &model_bin);

    Detector(const std::string &model_param, const std::string &model_bin, bool retinaface = false);

    inline void Release();

    void nms(std::vector<bbox> &input_boxes, float NMS_THRESH);

    void Detect(const unsigned char* rgba_data, int width, int height, std::vector<bbox>& boxes);

    void create_anchor(std::vector<box> &anchor, int w, int h);

    inline void SetDefaultParams();

    static inline bool cmp(bbox a, bbox b);

    ~Detector();

public:
    float _nms;
    float _threshold;
    float _mean_val[3];
    bool _retinaface;
    ncnn::Net *Net;
};
#endif //
