#pragma once

#include "Tracker.h"

class GroundTruthTracker : public Tracker
{
public:
    GroundTruthTracker(const std::string &directory_name) : Tracker(directory_name) {}

    void run() override;
};