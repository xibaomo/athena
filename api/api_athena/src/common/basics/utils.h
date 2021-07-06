/*
 * =====================================================================================
 *
 *       Filename:  utils.h
 *
 *    Description:  common utilities
 *
 *        Version:  1.0
 *        Created:  10/27/2018 02:31:18 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#ifndef  _BASIC_UTILS_H_
#define  _BASIC_UTILS_H_

#include <sys/socket.h>
#include <netinet/ip.h>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include <chrono>
#include <fstream>
#include "basics/log.h"
#include "types.h"
#include "pyhelper.hpp"
#include "minbar_predictor/mb_base/mb_base_pred.h"
#include <gsl/gsl_statistics_double.h>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
namespace athena {

struct SerializePack {
    std::vector<int>   int32_vec;
    std::vector<float> real32_vec;
    std::vector<double> real64_vec;
    std::vector<std::string> str_vec;
    std::vector<int>   int32_vec1;
    std::vector<float> real32_vec1;
    std::vector<double> real64_vec1;
    std::vector<std::string> str_vec1;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & int32_vec;
        ar & real32_vec;
        ar & real64_vec;
        ar & str_vec;
        ar & int32_vec1;
        ar & real32_vec1;
        ar & real64_vec1;
        ar & str_vec1;
    }
};
inline
std::string serialize(SerializePack& pack) {
    std::ostringstream ss;
    boost::archive::text_oarchive oa(ss);
    oa << pack;
    return ss.str();
}
inline void
unserialize(const std::string& str, SerializePack& pack) {
    std::stringstream ss(str);
    boost::archive::text_iarchive ia(ss);
    ia >> pack;
}
/*-----------------------------------------------------------------------------
 *  Execute system call by popen and return result as a string
 *-----------------------------------------------------------------------------*/
String execSysCall_block(const String& cmd);

class NonBlockSysCall {
  private:
    std::vector<FILE*> m_fhs;
    char m_buffer[1024];
    NonBlockSysCall() {;}
  public:

    static NonBlockSysCall& getInstance() {
        static NonBlockSysCall _inst;
        return _inst;
    }
    void exec(const String& cmd) {
        FILE* fh = popen(cmd.c_str(), "r");
        int d = fileno(fh);
        fcntl(d, F_SETFL, O_NONBLOCK);
        m_fhs.push_back(fh);
    }
    virtual ~NonBlockSysCall() {
        for ( auto fh: m_fhs) {
            pclose(fh);
        }
    }
//    bool checkFinished() {
//        int d = fileno(m_fh);
//        ssize_t r = read(d, m_buffer, 1024);
//        if ( r == -1 && errno == EAGAIN ) {
//            //Log(LOG_VERBOSE) << m_cmd + " not finished";
//            return false;
//        } else if (r > 0) {
//            return true;
//        } else
//            Log(LOG_ERROR) << "Pipe closed";
//        return false;
//    }

    String getResult() {
        String res(m_buffer);
        return res;
    }
};

extern NonBlockSysCall* gNBSysCall;
/*-----------------------------------------------------------------------------
 *  Sleep in units of ms
 *-----------------------------------------------------------------------------*/
void sleepMilliSec(int num_ms);

/*-----------------------------------------------------------------------------
 *  Get local host name
 *-----------------------------------------------------------------------------*/
String getHostName();

/*-----------------------------------------------------------------------------
 *  Split a string
 *-----------------------------------------------------------------------------*/
std::vector<String>
splitString(const String& str, const String delimiters=":");

/**
 * Case-insensitive string comparison
 */

bool
compareStringNoCase(const String& str1, const String& str2);

/**
 * Get time difference t1 - t2 in minutes
 */
long
getTimeDiffInMin(const String& t1, const String& t2);

/**
 * Convert time to string with given format
 */
String
convertTimeString(const String& timeStr, const String& format="%Y.%m.%d %H:%M");


class Timer {
  protected:
    std::chrono::time_point<std::chrono::system_clock> m_start;
  public:
    Timer() {
        m_start = std::chrono::system_clock::now();
    }
    double getElapsedTime() {
        auto now = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed = now - m_start;
        return elapsed.count();
    }

};

int getIntFromPyobject(CPyObject& pyobj);
String getStringFromPyobject(CPyObject& pyobj);

template <typename T>
void freeSTL(T& t) {
    T tmp;
    t.swap(tmp);
}

void showMinBar(MinBar& mb);

String getFileFolder(const String& fp);

String getFileStem(const String& fp);

void getPythonFunction(const String& modFile, const String& funcName,CPyObject& func);

template <typename T>
void savgol_smooth1D(std::vector<T>& invec, int width, int order, std::vector<real64>& ov) {
    String mhome = getenv("ATHENA_HOME");
    String utilscript = mhome + "/modules/basics/common/utils.py";
    CPyObject func;
    getPythonFunction(utilscript,"savgol_smooth1D",func);

    CPyObject lst = PyList_New(invec.size());
    for (size_t i=0; i < invec.size(); i++) {
        PyList_SetItem(lst,i,Py_BuildValue("d",(real64)invec[i]));
    }

    CPyObject args = Py_BuildValue("(Oii)",lst.getObject(),5,3);
    CPyObject res = PyObject_CallObject(func,args.getObject());

    ov.clear();
    PyArrayObject* arr =  (PyArrayObject*)res.getObject();
    int dim = arr->dimensions[0];
    real64* data = (real64*)arr->data;
//    for (int i=0; i < dim; i++) {
//        ov.push_back(data[i]);
//    }
    ov.assign(data,data+dim);

}

template <typename T>
real64
computePairCorr(std::vector<T>& v1, std::vector<T>& v2) {
    int len = v1.size();
    real64* x = new real64[len];
    real64* y = new real64[len];
    for (int i=0; i<len; i++) {
        x[i] = v1[i];
        y[i] = v2[i];
    }
    real64 corr = gsl_stats_correlation(x,1,y,1,len);

    delete[] x;
    delete[] y;

    return corr;
}

bool
test_coint(std::vector<real32>& v1, std::vector<real32>& v2, real32 pval=0.01);

/**
 * Compute p-value of adf test (stationarity)
 */
real64
testADF(real64* v, int len);

real64
testADF(std::vector<real64>& v, int start=-1, int last=-1);

real64
hurst(real64* v, int len);
/**
 * Dump arbitrary number of vectors
 */
inline
void dumpVectors_aux(std::ofstream& ofs, int) {
    ofs<<"\n";
}

template <typename V, typename... T>
void dumpVectors_aux(std::ofstream& ofs, int i, V& v, T&... args) {
    ofs << std::setprecision(10)<<v[i] <<",";
    dumpVectors_aux(ofs,i,args...);
}

template <typename V, typename ... T>
void dumpVectors(const String& csvfile, V& v, T&... args) {
    std::ofstream ofs(csvfile);
    for (size_t i = 0; i < v.size(); i++) {
        dumpVectors_aux(ofs, i, v, args...);
    }
    ofs.close();
}

/**
 * Use Z-test to compare two normal distributions
 * Z = 2.33 corresponds to p=0.01
 */
template <typename T>
T
compZtest(T m1, T m2, T sd1, T sd2) {
    real64 z = fabs(m1-m2)/sqrt(sd1*sd1+sd2*sd2);

    return z;
}

template <typename T>
int
searchVector(std::vector<T>& v, T tgt) {
    for(size_t i=0; i < v.size(); i++) {
        if (v[i] == tgt)
            return i;
    }
    return -1;
}

real64
compHalfLife(real64* ts, int len);
}
#endif   /* ----- #ifndef _BASIC_UTILS_H_  ----- */
