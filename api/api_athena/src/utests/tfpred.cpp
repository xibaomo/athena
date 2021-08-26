#include "predictor/tf23_wrapper.h"
#include <iostream>
using namespace std;

int main(int argc, char** argv)
{
    TFModel model;

    string pbfile(argv[1]);

    model.loadModel(pbfile);

    const std::vector<std::int64_t> input_dims = {1, 5, 12};
    const std::vector<float> input_vals =
    {
        -0.4809832f, -0.3770838f, 0.1743573f, 0.7720509f, -0.4064746f, 0.0116595f, 0.0051413f, 0.9135732f, 0.7197526f, -0.0400658f, 0.1180671f, -0.6829428f,
            -0.4810135f, -0.3772099f, 0.1745346f, 0.7719303f, -0.4066443f, 0.0114614f, 0.0051195f, 0.9135003f, 0.7196983f, -0.0400035f, 0.1178188f, -0.6830465f,
            -0.4809143f, -0.3773398f, 0.1746384f, 0.7719052f, -0.4067171f, 0.0111654f, 0.0054433f, 0.9134697f, 0.7192584f, -0.0399981f, 0.1177435f, -0.6835230f,
            -0.4808300f, -0.3774327f, 0.1748246f, 0.7718700f, -0.4070232f, 0.0109549f, 0.0059128f, 0.9133330f, 0.7188759f, -0.0398740f, 0.1181437f, -0.6838635f,
            -0.4807833f, -0.3775733f, 0.1748378f, 0.7718275f, -0.4073670f, 0.0107582f, 0.0062978f, 0.9131795f, 0.7187147f, -0.0394935f, 0.1184392f, -0.6840039f,
        };

    model.setInputNodeName("input_4");
    model.setOutputNodeName("output_node0");
    model.predict(input_vals.data(),input_dims);

    TF_Tensor* pred = model.getOutputTensor();

    float* data = model.getPredictedResult();
    std::cout << "Output vals: " << data[0] << ", " << data[1] << ", " << data[2] << ", " << data[3] << std::endl;

    std::vector<int64_t> dims = model.getOutputShape();
    for(auto v : dims) {
        cout<<v<<endl;
    }
    return 0;
}
