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

static int detect_simpleconv3net(const ncnn::Net &simpleconv3net,const cv::Mat& bgr, std::vector<float>& cls_scores)
{

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 48, 48);

    const float mean_vals[3] = {104.f, 117.f, 123.f};
    in.substract_mean_normalize(mean_vals, 0); //减均值，归一化

    ncnn::Extractor ex = simpleconv3net.create_extractor();

    ex.input("data", in); //填充数据

    ncnn::Mat out;
    ex.extract("prob", out); //获得结果
 
    cls_scores.resize(out.w); //取结果
    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
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
    ncnn::Net simpleconv3net;
    simpleconv3net.opt.use_vulkan_compute = true;
    simpleconv3net.load_param(modelparam);
    simpleconv3net.load_model(modelbin);

    cv::Mat image = cv::imread(imagepath, 1);

    if (image.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    //获得topk的分类概率
    std::vector<float> cls_scores;
    detect_simpleconv3net(simpleconv3net, image, cls_scores);
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
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    //绘制结果
    std::string text;
    std::string prob = std::to_string(vec[0].first);
    if(vec[0].second == 0)
        text.assign("neural ");
    else
        text.assign("smile ");
    text.append(prob);

    int font_face = cv::FONT_HERSHEY_COMPLEX; 
    double font_scale = 0.5;
    int thickness = 2;

    //将文本框居中绘制
    cv::Mat showimage = image.clone();
    cv::resize(showimage,showimage,cv::Size(256,256));
    cv::Point origin; 
    origin.x = showimage.cols / 3;
    origin.y = showimage.rows / 2;
    cv::putText(showimage, text, origin, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 8, 0);
    cv::namedWindow("image",0);
    cv::imshow("image",showimage);
    cv::waitKey(0);
    cv::imwrite(resultpath,showimage);

    return 0;
}
