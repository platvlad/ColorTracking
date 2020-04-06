#pragma once

#include "tracker/Tracker.h"

class SlsqpLktTracker : public Tracker
{
public:
    SlsqpLktTracker(const std::string &directory_name) : Tracker(directory_name) {}
    void run() override;
};