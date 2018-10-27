#include <iostream>
#include "basics/log.h"
#include "basics/types.h"
#include "messenger/msg.h"
using namespace std;

int main()
{
    Log.setLogLevel(LOG_INFO);

    Log(LOG_INFO) << "Program starts";


    Message msg;
    Log(LOG_INFO) << "Program ends normally";
    return 0;
}
