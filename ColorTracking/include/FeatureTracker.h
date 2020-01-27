#pragma once

#include <opencv2/core.hpp>

#include "lkt/Features.hpp"

class FeatureTracker
{
    lkt::Features frame_features;
    lkt::FeatureInfoList feature_list;
    cv::Mat1b prev_frame;
    int feature_id = -1;
    std::vector<cv::Mat1b> prev_frames;
    std::vector<lkt::FeatureInfo> prev_features;

public:
    FeatureTracker(const cv::Size &frame_size);

    void handleFrame(cv::Mat3b &frame);
};