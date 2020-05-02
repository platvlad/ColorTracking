#pragma once

#include "Tracker2.h"

class LktTracker : public Tracker2
{
public:
    LktTracker(const std::string &directory_name) : Tracker2(directory_name) {}
    void run() override;
};