#include "ma_cal.h"
#include  "ma_lwma.h"
MACalculator* createMACalculator(MA_TYPE mt)
{
    MACalculator* mc(nullptr);
    switch(mt) {
    case MA_TYPE::LWMA:
        mc = &MA_lwma::getInstance();
        break;
    default:
        Log(LOG_FATAL) << "MA type not supported";
        break;
    }

    return mc;
}
