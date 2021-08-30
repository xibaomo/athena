#include "predictor/tf23_wrapper.h"
#include <iostream>
using namespace std;

int main(int argc, char** argv)
{
    TFModel model;

    string pbfile(argv[1]);

    model.loadModel(pbfile);

    std::vector<std::int64_t> input_dims = {28, 28};
    const std::vector<float> input_vals =
    {
       0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.011764706,0.003921569,0.0,0.0,0.02745098,0.0,0.14509805,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.003921569,0.007843138,0.0,0.105882354,0.32941177,0.043137256,0.0,0.0,0.0,0.0,0.0,0.0,0.46666667,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.003921569,0.0,0.0,0.34509805,0.56078434,0.43137255,0.0,0.0,0.0,0.0,0.08627451,0.3647059,0.41568628,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.015686275,0.0,0.20784314,0.5058824,0.47058824,0.5764706,0.6862745,0.6156863,0.6509804,0.5294118,0.6039216,0.65882355,0.54901963,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.007843138,0.0,0.043137256,0.5372549,0.50980395,0.5019608,0.627451,0.6901961,0.62352943,0.654902,0.69803923,0.58431375,0.5921569,0.5647059,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.003921569,0.0,0.007843138,0.003921569,0.0,0.011764706,0.0,0.0,0.4509804,0.44705883,0.41568628,0.5372549,0.65882355,0.6,0.6117647,0.64705884,0.654902,0.56078434,0.6156863,0.61960787,0.043137256,0.0,0.0,0.0,0.0,0.0,0.003921569,0.0,0.0,0.0,0.0,0.0,0.011764706,0.0,0.0,0.34901962,0.54509807,0.3529412,0.36862746,0.6,0.58431375,0.5137255,0.5921569,0.6627451,0.6745098,0.56078434,0.62352943,0.6627451,0.1882353,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.007843138,0.015686275,0.003921569,0.0,0.0,0.0,0.38431373,0.53333336,0.43137255,0.42745098,0.43137255,0.63529414,0.5294118,0.5647059,0.58431375,0.62352943,0.654902,0.5647059,0.61960787,0.6627451,0.46666667,0.0,0.0,0.0,0.007843138,0.007843138,0.003921569,0.007843138,0.0,0.0,0.0,0.0,0.101960786,0.42352942,0.45882353,0.3882353,0.43529412,0.45882353,0.53333336,0.6117647,0.5254902,0.6039216,0.6039216,0.6117647,0.627451,0.5529412,0.5764706,0.6117647,0.69803923,0.0,0.011764706,0.0,0.0,0.0,0.0,0.0,0.0,0.08235294,0.20784314,0.36078432,0.45882353,0.43529412,0.40392157,0.4509804,0.5058824,0.5254902,0.56078434,0.6039216,0.64705884,0.6666667,0.6039216,0.5921569,0.6039216,0.56078434,0.5411765,0.5882353,0.64705884,0.16862746,0.0,0.0,0.09019608,0.21176471,0.25490198,0.29803923,0.33333334,0.4627451,0.5019608,0.48235294,0.43529412,0.44313726,0.4627451,0.49803922,0.49019608,0.54509807,0.52156866,0.53333336,0.627451,0.54901963,0.60784316,0.6313726,0.5647059,0.60784316,0.6745098,0.6313726,0.7411765,0.24313726,0.0,0.26666668,0.36862746,0.3529412,0.43529412,0.44705883,0.43529412,0.44705883,0.4509804,0.49803922,0.5294118,0.53333336,0.56078434,0.49411765,0.49803922,0.5921569,0.6039216,0.56078434,0.5803922,0.49019608,0.63529414,0.63529414,0.5647059,0.5411765,0.6,0.63529414,0.76862746,0.22745098,0.27450982,0.6627451,0.5058824,0.40784314,0.38431373,0.39215687,0.36862746,0.38039216,0.38431373,0.4,0.42352942,0.41568628,0.46666667,0.47058824,0.5058824,0.58431375,0.6117647,0.654902,0.74509805,0.74509805,0.76862746,0.7764706,0.7764706,0.73333335,0.77254903,0.7411765,0.72156864,0.14117648,0.0627451,0.49411765,0.67058825,0.7372549,0.7372549,0.72156864,0.67058825,0.6,0.5294118,0.47058824,0.49411765,0.49803922,0.57254905,0.7254902,0.7647059,0.81960785,0.8156863,1.0,0.81960785,0.69411767,0.9607843,0.9882353,0.9843137,0.9843137,0.96862745,0.8627451,0.80784315,0.19215687,0.0,0.0,0.0,0.047058824,0.2627451,0.41568628,0.6431373,0.7254902,0.78039217,0.8235294,0.827451,0.8235294,0.8156863,0.74509805,0.5882353,0.32156864,0.03137255,0.0,0.0,0.0,0.69803923,0.8156863,0.7372549,0.6862745,0.63529414,0.61960787,0.5921569,0.043137256,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0

        };

    vector<float> features;
    int nf = 20;
    for(int i=0;i < nf; i++)
        features.insert(features.end(),input_vals.begin(),input_vals.end());
    model.setInputNodeName("x");
    model.setOutputNodeName("Identity");
    auto vv = model.predict(features.data(), nf,input_dims);

    cout<<vv.size()<<endl;
    for(auto v : vv[nf-1]) {
        cout<<v<<endl;
    }
    return 0;
}
