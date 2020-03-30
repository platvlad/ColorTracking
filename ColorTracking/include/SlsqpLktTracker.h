#pragma once

#include "Tracker.h"

class SlsqpLktTracker : public Tracker
{
public:
    SlsqpLktTracker(const std::string &directory_name) : Tracker(directory_name) {}
    void run() override;
};