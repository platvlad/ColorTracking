#pragma once

#include "Tracker2.h"

class LktInitTracker: public Tracker2
{
public:
    LktInitTracker(const std::string &directory_name) : Tracker2(directory_name) {}
    void run() override;
};