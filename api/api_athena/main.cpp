#include <iostream>
#include "basics/log.h"
#include "basics/types.h"
using namespace std;

int main()
{
    Log.setLogLevel(LOG_INFO);

    Log(LOG_INFO) << "Program starts";


    Log(LOG_INFO) << "Program ends normally";
    return 0;
}
