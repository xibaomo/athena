/*
 * =====================================================================================
 *
 *       Filename:  main.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  12/02/2018 03:00:36 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "app.h"

int main(int argc, char* argv[])
{
    // argv[1] - AppType
    // argv[2] - configFile
    AppType atp = (AppType)stoi(String(argv[1]));
    String cfg = String(argv[2]);
    App* app = createApp(atp,cfg);

    app->prepare();

    app->execute();

    app->finish();

    return 0;
}
