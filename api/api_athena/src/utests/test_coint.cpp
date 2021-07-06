#include "basics/utils.h"
#include "basics/types.h"
#include <vector>
#define N 100
using namespace std;
using namespace athena;
int main() {
    PyEnviron::getInstance();
    vector<real64> v1;
    vector<real64> v2;

    for (int i=0; i < N; i++) {
        v1.push_back(i*.1);
        v2.push_back(i*i*0.2+1);
    }

    double r = testADF(&v1[0],N);
    cout<<r<<endl;

    return 0;
}

