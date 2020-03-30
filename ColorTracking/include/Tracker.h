#include "DataIO.h"

class Tracker
{
    DataIO data;

public:
    Tracker(const std::string &directory_name);

    virtual void run() = 0;
};