#include "net.h"

#include <algorithm>
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <stdio.h>
#include <vector>
#include <cmath>

static int detect_simpleconv5net(const ncnn::Net &simpleconv5net,const cv::Mat& bgr, std::vector<float>& cls_scores)
{

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, 224, 224);
    // transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    const float mean_vals[3] = {0.5f*255.f, 0.5f*255.f, 0.5f*255.f};
    const float norm_vals[3] = {1/0.5f/255.f, 1/0.5f/255.f, 1/0.5f/255.f};

    in.substract_mean_normalize(mean_vals, norm_vals); //ncnn中的substract_mean_normalize函数像素范围在[0,255]

    ncnn::Extractor ex = simpleconv5net.create_extractor();

    ex.input("input.1", in); //填充数据

    ncnn::Mat out;
    float start_time = cv::getTickCount(); //计算模型推理时间
    ex.extract("59", out); //获得模型推理结果
    float end_time = cv::getTickCount();
    fprintf(stderr, "%s = %f %s\n", "inference time = ", (end_time-start_time)/cv::getTickFrequency()*1000, " ms");
 
    cls_scores.resize(out.w); //取softmax分类概率结果，指数减去固定值防止溢出处理
    float maxscore = 0.0;
    for (int j = 0; j < out.w; j++)
    {
        if(out[j] >= maxscore) maxscore = out[j]; 
        cls_scores[j] = out[j];
    }
    float sum = 0.0;

    for (int j = 0; j < out.w; j++)
    {
        //fprintf(stderr, "%s %i=%f\n", "raw score",j,cls_scores[j]);
        cls_scores[j] = std::exp(cls_scores[j]-maxscore);
        //fprintf(stderr, "%s %i=%f\n", "after exp score",j,cls_scores[j]);
        sum += cls_scores[j]; 
    }

    //fprintf(stderr, "%s=%f %s=%f\n", "max fc score = ",maxscore, "sum score=", sum);
    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = cls_scores[j] / sum;
    }
    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s%s%s [modelparam modelbin imagepath resultpath]\n", argv[0], argv[1], argv[2], argv[3]);
        return -1;
    }

    const char* modelparam = argv[1];
    const char* modelbin = argv[2];
    const char* imagepath = argv[3];
    const char* resultpath = argv[4];

    //初始化模型
    ncnn::Net simpleconv5net;
    simpleconv5net.opt.use_vulkan_compute = true;
    simpleconv5net.load_param(modelparam);
    simpleconv5net.load_model(modelbin);

    cv::Mat image = cv::imread(imagepath, 1);

    if (image.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    //获得topk的分类概率
    std::vector<float> cls_scores;
    detect_simpleconv5net(simpleconv5net, image, cls_scores);
    int topk = 1;
    int size = cls_scores.size();
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    for (int i = 0; i < topk; i++)
    {
        fprintf(stderr, "%d = %f\n", vec[i].second, floor(vec[0].first * 10000.000f + 0.5) / 10000.000f);
    }

    //绘制结果
    std::string text;
    std::stringstream ss;
    ss.setf(std::ios::fixed);
    ss.precision(3);
    ss << vec[0].first;
    double score = atof(ss.str().c_str());
    std::string label = "c="+std::to_string(vec[0].second);
    std::string prob = "prob="+std::to_string(floor(vec[0].first * 10000.000f + 0.5) / 10000.000f);
    text.assign(label+" ");
    text.append(prob);

    int font_face = cv::FONT_HERSHEY_COMPLEX; 
    double font_scale = 0.75;
    int thickness = 2;

    //将文本框居中绘制
    cv::Mat showimage = image.clone();
    cv::resize(showimage,showimage,cv::Size(256,256));
    cv::Point origin; 
    origin.x = showimage.cols / 20;
    origin.y = showimage.rows / 2;
    cv::putText(showimage, text, origin, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 8, 0);
    cv::namedWindow("image",0);
    cv::imshow("image",showimage);
    //cv::waitKey(0);
    cv::imwrite(resultpath,showimage);

    return 0;
}
