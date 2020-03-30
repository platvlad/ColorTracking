#pragma once

#include "Tracker.h"

class LktTracker : public Tracker
{
public:
    LktTracker(const std::string &directory_name) : Tracker(directory_name) {}
    void run() override;
};