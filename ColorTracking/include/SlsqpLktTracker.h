#pragma once

#include "Tracker2.h"

class SlsqpLktTracker : public Tracker2
{
public:
    SlsqpLktTracker(const std::string &directory_name) : Tracker2(directory_name) {}
    void run() override;
};