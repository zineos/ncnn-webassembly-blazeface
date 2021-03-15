#include <algorithm>
// #include "omp.h"
#include "blazeface.h"

Detector::Detector():
        _nms(0.4),
        _threshold(0.6),
        _mean_val{104.f, 117.f, 123.f},
        Net(new ncnn::Net())
{
}

inline void Detector::Release(){
    if (Net != nullptr)
    {
        delete Net;
        Net = nullptr;
    }
}

Detector::Detector(const std::string &model_param, const std::string &model_bin, bool retinaface):
        _nms(0.5),
        _threshold(0.5),
        _mean_val{104.f, 117.f, 123.f},
        Net(new ncnn::Net())
{
    Init(model_param, model_bin);
}

void Detector::Init(const std::string &model_param, const std::string &model_bin)
{
    int ret = Net->load_param(model_param.c_str());
    ret = Net->load_model(model_bin.c_str());
}

void Detector::Detect(const unsigned char* rgba_data, int width, int height, std::vector<bbox>& boxes)
{
    // Timer timer;
    // timer.tic();
    const int target_size = 128;

    // letterbox pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;

    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgba_data, ncnn::Mat::PIXEL_RGBA2RGB, width, height, w, h);

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = (w + 15) / 16 * 16 - w;
    int hpad = (h + 15) / 16 * 16 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    in_pad.substract_mean_normalize(_mean_val, 0);
    // timer.toc("precoss:");
    // timer.tic();
    ncnn::Extractor ex = Net->create_extractor();

    ex.set_light_mode(true);
    // ex.set_num_threads(4);

    ex.input(0, in_pad);
    ncnn::Mat out, out1, out2;

    // loc
    ex.extract("boxes", out);

    // class
    ex.extract("scores", out1);

    //landmark
    ex.extract("landmark", out2);

    // timer.toc("det:");

    std::vector<box> anchor;
    // timer.tic();
    create_anchor(anchor,  w, h);
    // timer.toc("anchor:");

    std::vector<bbox > total_box;
    float *ptr = out.channel(0);
    float *ptr1 = out1.channel(0);
    float *landms = out2.channel(0);

    // #pragma omp parallel for num_threads(2)
    for (int i = 0; i < anchor.size(); ++i)
    {
        if (*(ptr1+1) > _threshold)
        {
            box tmp = anchor[i];
            box tmp1;
            bbox result;

            // loc and conf
            tmp1.cx = tmp.cx + *ptr * 0.1 * tmp.sx;
            tmp1.cy = tmp.cy + *(ptr+1) * 0.1 * tmp.sy;
            tmp1.sx = tmp.sx * exp(*(ptr+2) * 0.2);
            tmp1.sy = tmp.sy * exp(*(ptr+3) * 0.2);

            result.x1 = (tmp1.cx - tmp1.sx/2) * in.w ;
            if (result.x1<0)
                result.x1 = 0;
            result.y1 = (tmp1.cy - tmp1.sy/2) * in.h ;
            if (result.y1<0)
                result.y1 = 0;
            result.x2 = (tmp1.cx + tmp1.sx/2) * in.w ;
            if (result.x2>in.w)
                result.x2 = in.w;
            result.y2 = (tmp1.cy + tmp1.sy/2)* in.h ;
            if (result.y2>in.h)
                result.y2 = in.h;
            result.s = *(ptr1 + 1);

            // landmark
            for (int j = 0; j < 5; ++j)
            {
                result.point[j]._x =( tmp.cx + *(landms + (j<<1)) * 0.1 * tmp.sx ) * in.w;
                result.point[j]._y =( tmp.cy + *(landms + (j<<1) + 1) * 0.1 * tmp.sy ) * in.h;
            }

            total_box.push_back(result);
        }
        ptr += 4;
        ptr1 += 2;
        landms += 10;
    }

    std::sort(total_box.begin(), total_box.end(), cmp);
    nms(total_box, _nms);
    printf("%d\n", (int)total_box.size());

    for (int j = 0; j < total_box.size(); ++j)
    {   
        total_box[j].x1 = (total_box[j].x1 - (wpad / 2)) / scale;
        total_box[j].y1 = (total_box[j].y1 - (hpad / 2)) / scale;
        total_box[j].x2 = (total_box[j].x2 - (wpad / 2)) / scale;
        total_box[j].y2 = (total_box[j].y2 - (hpad / 2)) / scale;
        for (int k = 0; k < 5; ++k)
        {
            total_box[j].point[k]._x = (total_box[j].point[k]._x  - (wpad / 2)) / scale;
            total_box[j].point[k]._y = (total_box[j].point[k]._y  - (hpad / 2)) / scale;
        }
        boxes.push_back(total_box[j]);
    }
}

inline bool Detector::cmp(bbox a, bbox b) {
    if (a.s > b.s)
        return true;
    return false;
}

inline void Detector::SetDefaultParams(){
    _nms = 0.4;
    _threshold = 0.6;
    _mean_val[0] = 104;
    _mean_val[1] = 117;
    _mean_val[2] = 123;
    Net = nullptr;

}

Detector::~Detector(){
    Release();
}

void Detector::create_anchor(std::vector<box> &anchor, int w, int h)
{
//    anchor.reserve(num_boxes);
    anchor.clear();
    std::vector<std::vector<int> > feature_map(2), min_sizes(4);
    float steps[] = {8, 16};
    for (int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h/steps[i]));
        feature_map[i].push_back(ceil(w/steps[i]));
    }
    std::vector<int> minsize1 = {8, 11};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {14, 19, 26, 38, 64, 149};
    min_sizes[1] = minsize2;


    for (int k = 0; k < feature_map.size(); ++k)
    {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i)
        {
            for (int j = 0; j < feature_map[k][1]; ++j)
            {
                for (int l = 0; l < min_size.size(); ++l)
                {
                    float s_kx = min_size[l]*1.0/w;
                    float s_ky = min_size[l]*1.0/h;
                    float cx = (j + 0.5) * steps[k]/w;
                    float cy = (i + 0.5) * steps[k]/h;
                    box axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }

    }

}


void Detector::nms(std::vector<bbox> &input_boxes, float NMS_THRESH)
{
    std::vector<float>vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float   h = std::max(float(0), yy2 - yy1 + 1);
            float   inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}


std::string param = "blaceface.param";
std::string bin = "blaceface.bin";

Detector detector(param, bin, false);


static int detect_blazeface(const unsigned char* rgba_data, int width, int height, std::vector<bbox>& boxes)
{   
    if (detector.Net == nullptr)
    {
       detector.Init("blaceface.param","blaceface.bin");
    }
    detector.Detect(rgba_data, width, height, boxes);

    return 0;
}


#include <emscripten.h>

static const unsigned char* rgba_data = 0;
static int w = 0;
static int h = 0;
static float* result_buffer = 0;

static ncnn::Mutex lock;
static ncnn::ConditionVariable condition;

static ncnn::Mutex finish_lock;
static ncnn::ConditionVariable finish_condition;

static void worker()
{
    while (1)
    {
        lock.lock();
        while (rgba_data == 0)
        {
            condition.wait(lock);
        }

        std::vector<bbox> objects;
        detect_blazeface(rgba_data, w, h, objects);

//         print_objects(objects);

        // result_buffer max 20 objects
        if (objects.size() > 20)
            objects.resize(20);

        size_t i = 0;
        for (; i < objects.size(); i++)
        {
            const bbox& obj = objects[i];

            result_buffer[0] = 0;
            result_buffer[1] = obj.s;
            result_buffer[2] = obj.x1;
            result_buffer[3] = obj.y1;
            result_buffer[4] = obj.x2 - obj.x1;
            result_buffer[5] = obj.y2 - obj.y1;
            result_buffer[6] = obj.point[0]._x;
            result_buffer[7] = obj.point[0]._y;
            result_buffer[8] = obj.point[1]._x;
            result_buffer[9] = obj.point[1]._y;
            result_buffer[10] = obj.point[2]._x;
            result_buffer[11] = obj.point[2]._y;
            result_buffer[12] = obj.point[3]._x;
            result_buffer[13] = obj.point[3]._y;
            result_buffer[14] = obj.point[4]._x;
            result_buffer[15] = obj.point[4]._y;


            result_buffer += 16;
        }
        for (; i < 20; i++)
        {
            result_buffer[0] = -233;
            result_buffer[1] = -233;
            result_buffer[2] = -233;
            result_buffer[3] = -233;
            result_buffer[4] = -233;
            result_buffer[5] = -233;
            result_buffer[6] = -233;
            result_buffer[7] = -233;
            result_buffer[8] = -233;
            result_buffer[9] = -233;
            result_buffer[10] = -233;
            result_buffer[11] = -233;
            result_buffer[12] = -233;
            result_buffer[13] = -233;
            result_buffer[14] = -233;
            result_buffer[15] = -233;


            result_buffer += 16;
        }

        rgba_data = 0;

        lock.unlock();

        finish_lock.lock();
        finish_condition.signal();
        finish_lock.unlock();
    }
}

#include <thread>
static std::thread t(worker);

extern "C" {

void blazeface_ncnn(const unsigned char* _rgba_data, int _w, int _h, float* _result_buffer)
{
    lock.lock();
    while (rgba_data != 0)
    {
        condition.wait(lock);
    }

    rgba_data = _rgba_data;
    w = _w;
    h = _h;
    result_buffer = _result_buffer;

    lock.unlock();

    condition.signal();

    // wait for finished
    finish_lock.lock();
    while (rgba_data != 0)
    {
        finish_condition.wait(finish_lock);
    }
    finish_lock.unlock();
}

}