#pragma once

#include "Tracker2.h"

#include "LkPoseGetter.h"

class LktTracker : public Tracker2
{
public:
    LktTracker(const std::string &directory_name) : Tracker2(directory_name) {}
    void plotEnergy(const LkPoseGetter& f_tracker, const glm::mat4& pose, int frame_number);
    void run() override;
};