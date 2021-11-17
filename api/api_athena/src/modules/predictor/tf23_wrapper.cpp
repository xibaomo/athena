#include "tf23_wrapper.h"
#include "basics/log.h"
#include <fstream>
#include <numeric>

#define CHECK_TF_STATUS(func_name) if(TF_GetCode(tfstat)!=TF_OK){Log(LOG_FATAL) << "TF status fails: " + string("func_name") <<std::endl;}
using namespace std;

static void deallocateBuffer(void* data, size_t)
{
    std::free(data);
}

static TF_Buffer* readBufferFromFile(const char* file)
{
    std::ifstream f(file,std::ios::binary);
    if(f.fail() || !f.is_open())
    {
        Log(LOG_FATAL) << "Cannot open file: " + string(file) <<std::endl;
        return nullptr;
    }
    if(f.seekg(0,std::ios::end).fail())
    {
        return nullptr;
    }

    auto fsize = f.tellg();
    if(f.seekg(0,std::ios::beg).fail())
    {
        return nullptr;
    }
    if(fsize<0)
    {
        return nullptr;
    }

    char* data = (char*)malloc(fsize);
    if(f.read(data,fsize).fail())
    {
        Log(LOG_FATAL) << "Fail to read file: " + string(file) <<std::endl;
        return nullptr;
    }

    TF_Buffer* buf = TF_NewBuffer();
    buf->data = data;
    buf->length = fsize;
    buf->data_deallocator = deallocateBuffer;

    return buf;

}

TFModel::TFModel():m_graph(nullptr),m_session(nullptr),m_output_tensor(nullptr)
{
}

TFModel::~TFModel()
{
    if(!m_graph)  TF_DeleteGraph(m_graph);
    if(!m_session) deleteSession(m_session);
    if(!m_output_tensor) TF_DeleteTensor(m_output_tensor);
}

TF_Tensor*
TFModel::scalarStringTensor(const char* str, TF_Status* status)
{
    size_t str_len = std::strlen(str);
    size_t nbytes = 8 + TF_StringEncodedSize(str_len); // 8 extra bytes - for start offset
    TF_Tensor* tensor = TF_AllocateTensor(TF_STRING, nullptr,0,nbytes);
    char* data = (char*)(TF_TensorData(tensor));

    std::memset(data,0,8);
    TF_StringEncode(str,str_len,data+8,nbytes-8,status);
    return tensor;
}

TF_Session*
TFModel::createSession(TF_Graph* graph)
{
    if (!graph) return nullptr;

    auto tfstat = TF_NewStatus();
    auto options = TF_NewSessionOptions();

    auto session = TF_NewSession(graph,options,tfstat);
    CHECK_TF_STATUS(TF_NewSession);

    TF_DeleteSessionOptions(options);
    TF_DeleteStatus(tfstat);
    return session;
}

void
TFModel::deleteSession(TF_Session* sess)
{
    if (!sess) return;

    auto tfstat = TF_NewStatus();

    TF_CloseSession(sess,tfstat);
    CHECK_TF_STATUS(TF_CloseSession);

    TF_DeleteSession(sess,tfstat);
    CHECK_TF_STATUS(TF_DeleteSession);

    TF_DeleteStatus(tfstat);
}

void
TFModel::loadGraph(const char* graph_file, const char* checkpoint_prefix)
{
    TF_Buffer* buf = readBufferFromFile(graph_file);
    if(!buf)
    {
        Log(LOG_FATAL) << "Fail to read graph file: " + string(graph_file) <<std::endl;
    }

    auto tfstat = TF_NewStatus();

    if(m_graph)
    {
        TF_DeleteGraph(m_graph);
        m_graph = nullptr;
    }
    m_graph = TF_NewGraph();

    auto opts = TF_NewImportGraphDefOptions();

    TF_GraphImportGraphDef(m_graph,buf,opts,tfstat);
    TF_DeleteImportGraphDefOptions(opts);
    TF_DeleteBuffer(buf);

    if(TF_GetCode(tfstat)!=TF_OK)
    {
        Log(LOG_FATAL) << "Fail to load graph from buffer" <<std::endl;
    }

    if(!checkpoint_prefix) return;

    // usually useless
    auto checkpoint_tensor = scalarStringTensor(checkpoint_prefix,tfstat);
    if(TF_GetCode(tfstat)!=TF_OK)
    {
        Log(LOG_FATAL) << "Fail to create checkpoint tensor" <<std::endl;
    }

    auto input = TF_Output{TF_GraphOperationByName(m_graph,"save/Const"),0};
    auto restore_op = TF_GraphOperationByName(m_graph,"save/restore_all");

    TF_Session* session = createSession(m_graph);

    TF_SessionRun(session,
                  nullptr, // RUn options
                  &input,&checkpoint_tensor,1, // input tensors,input tensor values, number of inputs
                  nullptr,nullptr,0, //output tensors, output tensor values, number of outputs.
                  &restore_op,1,// target operations, number of targets.
                  nullptr, // run metadata
                  tfstat);

    if(TF_GetCode(tfstat)!=TF_OK)
    {
        Log(LOG_FATAL) << "TF_SessionRun fails" <<std::endl;
    }

    if(session)
    {
        deleteSession(session);
    }
    if(checkpoint_tensor) TF_DeleteTensor(checkpoint_tensor);
}

TF_Tensor* TFModel::createTensor(TF_DataType data_type,
                                 const std::int64_t* dims, std::size_t num_dims, const void* data, std::size_t len)
{
    TF_Tensor* tensor = createEmptyTensor(data_type, dims, num_dims, len);
    auto tensor_data = TF_TensorData(tensor);
    if(!tensor_data)
    {
        Log(LOG_FATAL) << "Fail to create tensor data" <<std::endl;
    }

    len = std::min(len,TF_TensorByteSize(tensor));
    if(data && len != 0)
    {
        memcpy(tensor_data,data,len);
    }

    return tensor;
}

bool TFModel::setTensorData(TF_Tensor* tensor, const void* data, std::size_t len)
{
    auto tensor_data = TF_TensorData(tensor);
    len = std::min(len,TF_TensorByteSize(tensor));
    if(tensor_data && data && len != 0)
    {
        memcpy(tensor_data,data,len);
        return true;
    }
    return false;
}

std::vector<std::int64_t>
TFModel::getTensorShape(TF_Graph* graph, const TF_Output& output)
{
    auto tfstat = TF_NewStatus();

    auto num_dims = TF_GraphGetTensorNumDims(graph,output,tfstat);
    CHECK_TF_STATUS(TF_GraphGetTensorNumDims);

    vector<std::int64_t> result(num_dims);
    TF_GraphGetTensorShape(graph,output,result.data(),num_dims,tfstat);
    CHECK_TF_STATUS(TF_GraphGetTensorShape);

    TF_DeleteStatus(tfstat);

    return result;
}

std::vector<std::vector<std::int64_t>>
                                    TFModel::getTensorsShape(TF_Graph* graph, const std::vector<TF_Output>& outputs)
{
    std::vector<std::vector<std::int64_t>> result;
    result.reserve(outputs.size());

    for(const auto& o : outputs)
    {
        result.push_back(getTensorShape(graph,o));
    }

    return result;
}

std::vector<int64_t> TFModel::getTensorShape( TF_Tensor * tensor )
{
    auto ndims = TF_NumDims( tensor );

    std::vector<int64_t> dims;

    for ( int i = 0; i < ndims; i++ )
    {
        dims.push_back( TF_Dim( tensor, i ) );
    }

    return dims;
}

static const char* DataTypeToString(TF_DataType data_type)
{
    switch (data_type)
    {
    case TF_FLOAT:
        return "TF_FLOAT";
    case TF_DOUBLE:
        return "TF_DOUBLE";
    case TF_INT32:
        return "TF_INT32";
    case TF_UINT8:
        return "TF_UINT8";
    case TF_INT16:
        return "TF_INT16";
    case TF_INT8:
        return "TF_INT8";
    case TF_STRING:
        return "TF_STRING";
    case TF_COMPLEX64:
        return "TF_COMPLEX64";
    case TF_INT64:
        return "TF_INT64";
    case TF_BOOL:
        return "TF_BOOL";
    case TF_QINT8:
        return "TF_QINT8";
    case TF_QUINT8:
        return "TF_QUINT8";
    case TF_QINT32:
        return "TF_QINT32";
    case TF_BFLOAT16:
        return "TF_BFLOAT16";
    case TF_QINT16:
        return "TF_QINT16";
    case TF_QUINT16:
        return "TF_QUINT16";
    case TF_UINT16:
        return "TF_UINT16";
    case TF_COMPLEX128:
        return "TF_COMPLEX128";
    case TF_HALF:
        return "TF_HALF";
    case TF_RESOURCE:
        return "TF_RESOURCE";
    case TF_VARIANT:
        return "TF_VARIANT";
    case TF_UINT32:
        return "TF_UINT32";
    case TF_UINT64:
        return "TF_UINT64";
    default:
        return "Unknown";
    }
}

static const char* CodeToString(TF_Code code)
{
    switch (code)
    {
    case TF_OK:
        return "TF_OK";
    case TF_CANCELLED:
        return "TF_CANCELLED";
    case TF_UNKNOWN:
        return "TF_UNKNOWN";
    case TF_INVALID_ARGUMENT:
        return "TF_INVALID_ARGUMENT";
    case TF_DEADLINE_EXCEEDED:
        return "TF_DEADLINE_EXCEEDED";
    case TF_NOT_FOUND:
        return "TF_NOT_FOUND";
    case TF_ALREADY_EXISTS:
        return "TF_ALREADY_EXISTS";
    case TF_PERMISSION_DENIED:
        return "TF_PERMISSION_DENIED";
    case TF_UNAUTHENTICATED:
        return "TF_UNAUTHENTICATED";
    case TF_RESOURCE_EXHAUSTED:
        return "TF_RESOURCE_EXHAUSTED";
    case TF_FAILED_PRECONDITION:
        return "TF_FAILED_PRECONDITION";
    case TF_ABORTED:
        return "TF_ABORTED";
    case TF_OUT_OF_RANGE:
        return "TF_OUT_OF_RANGE";
    case TF_UNIMPLEMENTED:
        return "TF_UNIMPLEMENTED";
    case TF_INTERNAL:
        return "TF_INTERNAL";
    case TF_UNAVAILABLE:
        return "TF_UNAVAILABLE";
    case TF_DATA_LOSS:
        return "TF_DATA_LOSS";
    default:
        return "Unknown";
    }
}

void
TFModel::setInputNodeName(const std::string& inputName)
{
    m_input_op_name = inputName;
    m_input_op = TF_Output{TF_GraphOperationByName(m_graph,m_input_op_name.c_str()),0};

    if(!m_input_op.oper)
        Log(LOG_FATAL) << "Cannot init input_op" <<std::endl;
}

void
TFModel::setOutputNodeName(const std::string& outputName)
{
    m_output_op_name = outputName;
    m_output_op = TF_Output{TF_GraphOperationByName(m_graph,m_output_op_name.c_str()),0};

    if(!m_output_op.oper)
        Log(LOG_FATAL) << "Cannot init output_op" <<std::endl;
}

void
TFModel::loadModel(const std::string& pbfile)
{
    loadGraph(pbfile.c_str(),nullptr);
}

void
TFModel::predict_singleInput(const float* input, const std::vector<int64_t>& dims)
{
    auto tfstat = TF_NewStatus();

    if(!m_session)
    {
        m_session = createSession(m_graph);
    }

    size_t num = std::accumulate(dims.begin(), dims.end(), 1, multiplies<int64_t>());

    auto input_tensor = createTensor(TF_FLOAT,dims.data(),dims.size(),input,num*sizeof(float));

    if(m_output_tensor)
    {
        TF_DeleteTensor(m_output_tensor);
        m_output_tensor = nullptr;
    }

    TF_SessionRun(m_session,
                  nullptr,//run options
                  &m_input_op,&input_tensor,1,//input node, input value, num of inputs
                  &m_output_op,&m_output_tensor,1,
                  nullptr,0,//target operations, num of targets
                  nullptr, // run metadata
                  tfstat);
    CHECK_TF_STATUS(TF_SessionRun);

    TF_DeleteStatus(tfstat);
}
std::vector<std::vector<float>>
TFModel::predict(const float* input, int nInputs, const std::vector<int64_t>& dims) {
    auto shape = getOutputShape();
    int len_output = std::accumulate(shape.begin(), shape.end(), 1, multiplies<int>());
    int nElems = std::accumulate(dims.begin(), dims.end(), 1, multiplies<int64_t>());

    vector<vector<float>> vv;
    for(int i=0; i < nInputs; i++) {
        const float* data = input + i*nElems;
        predict_singleInput(data,dims);
        float* pred = getPredictedResult();
        vector<float> v(pred,pred+len_output);
        vv.push_back(std::move(v));
    }

    return vv;
}


