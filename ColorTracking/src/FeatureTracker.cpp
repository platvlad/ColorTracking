#include "FeatureTracker.h"
#include <opencv2/highgui.hpp>

#include "lkt/optflow.hpp"

#include <iostream>


FeatureTracker::FeatureTracker(const cv::Size &frame_size) : 
    frame_features(lkt::Features(frame_size.area())) {}

void FeatureTracker::handleFrame(cv::Mat3b &frame)
{
    cv::Mat1b gray_frame;
    cv::cvtColor(frame, gray_frame, CV_RGB2GRAY);
    
    if (prev_frame.empty())
    {
        lkt::Features::FeaturesAndCorners features = frame_features.detectFeaturesAndCorners(gray_frame);
        
        feature_list = features.first;
        double max_x = 0;
        double max_y = 0;
        for (int i = 0; i < feature_list.size(); ++i)
        {
            if (feature_list[i].y >= 300 && feature_list[i].y < 600 && feature_list[i].x >= 650 && feature_list[i].x < 800)
            {
                feature_id = i;
                prev_features.push_back(feature_list[i]);
                cv::Rect window_rect(200, 500, 500, 500);
                cv::Mat1b gray_window = gray_frame(window_rect);
                prev_frames.push_back(gray_window);
                break;
            }
            if (feature_list[i].x > max_x)
            {
                max_x = feature_list[i].x;
            }
            if (feature_list[i].y > max_y)
            {
                max_y = feature_list[i].y;
            }
        }
        prev_frame = gray_frame;
        return;
    }
    
    feature_list = lkt::moveFeaturesBySparseFlow(prev_frame, gray_frame, feature_list);
    lkt::FeatureInfo feature = feature_list[feature_id];
    if (feature.flowQuality.lukasKanadeSuccess && feature.y >= 300 && feature.y < 600 && feature.x >= 650 && feature.x < 800)
    {
        cv::Rect window_rect(200, 500, 500, 500);
        cv::Mat1b window = gray_frame(window_rect);
        prev_features.push_back(feature);
        prev_frames.push_back(window);
        int oldValue = gray_frame(feature.y, feature.x);
        gray_frame(feature.y, feature.x) = 255;
        cv::imwrite("data/foxes/debug_frames/Frame" + std::to_string(prev_frames.size()) + ".png", gray_frame);
        gray_frame(feature.y, feature.x) = oldValue;
    }
    bool smth = true;
    prev_frame = gray_frame;
}