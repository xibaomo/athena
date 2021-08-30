#pragma once

#include <string>
#include <vector>
#include<unordered_map>
#include<map>
#include <tensorflow/c/c_api.h>

class TFModel
{
public:
    template <typename T>
    struct Array2D
    {
        T* data;
        size_t rows;
        size_t cols;
        bool m_own;
        Array2D() : data(nullptr), rows(0), cols(0), m_own(true) {;}
        virtual ~Array2D()
        {
            release();
        }
        void release()
        {
            if(m_own && data) delete[] data;
        }
        int channels() const
        {
            return 1;
        }
        int total() const
        {
            return rows*cols;
        }
    };
    typedef Array2D<float> Real2D; // includes 1D
private:
    TF_Graph*    m_graph;
    TF_Session*  m_session;

    TF_Output    m_input_op;
    TF_Output    m_output_op;

    std::string m_input_op_name;
    std::string m_output_op_name;

    TF_Tensor*  m_output_tensor;
public:
    TFModel();
    ~TFModel();

    TF_Tensor* scalarStringTensor(const char* str, TF_Status* status);

    void loadGraph(const char* graph_file, const char* checkpoint_prefix);

    TF_Session* createSession(TF_Graph* graph);

    void deleteSession(TF_Session* sess);

    TF_Tensor* createEmptyTensor(TF_DataType data_type, const std::int64_t* dims, std::size_t num_dims, std::size_t len)
    {
        return TF_AllocateTensor(data_type,dims,(int)num_dims,len);
    }

    TF_Tensor* createEmptyTensor(TF_DataType data_type, const std::vector<std::int64_t>& dims, std::size_t len)
    {
        return createEmptyTensor(data_type, dims.data(),dims.size(),len);
    }

    TF_Tensor* createTensor(TF_DataType data_type,
                            const std::int64_t* dims, std::size_t num_dims, const void* data, std::size_t len);

    bool setTensorData(TF_Tensor* tensor, const void* data, std::size_t len);

    std::vector<std::int64_t> getTensorShape(TF_Graph* graph, const TF_Output& output);
    std::vector<std::vector<std::int64_t>> getTensorsShape(TF_Graph* graph, const std::vector<TF_Output>& outputs);
    std::vector<int64_t> getTensorShape( TF_Tensor * tensor );

    // public interface

    void loadModel(const std::string& fn);

    void setInputNodeName(const std::string& inputName);
    void setOutputNodeName(const std::string& outputName);

    void predict_singleInput(const float* input, const std::vector<int64_t>& dims);

    // dims[0] is number of inputs, dims[1:] is dimension of single input
    std::vector<std::vector<float>> predict(const float* input, int nInputs,
                                            const std::vector<int64_t>& dims);
    TF_Tensor* getOutputTensor() { return m_output_tensor; }

    std::vector<int> getOutputShape() {
        std::vector<int64_t> vv = getTensorShape(m_graph,m_output_op);
        std::vector<int> v;
        v.assign(vv.begin()+1,vv.end());
        return v;
    }

    float* getPredictedResult() {
        return static_cast<float*>(TF_TensorData(m_output_tensor));
    }
};


