#pragma once

#include "Tracker.h"

class LktInitTracker: public Tracker
{
public:
    LktInitTracker(const std::string &directory_name) : Tracker(directory_name) {}
    void run() override;
};