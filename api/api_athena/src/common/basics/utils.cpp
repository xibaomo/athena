/*
 * =====================================================================================
 *
 *       Filename:  utils.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  10/27/2018 02:33:24 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (),
 *   Organization:
 *
 * =====================================================================================
 */

#include "utils.h"
#include "pyrunner/pyrunner.h"
#include "linreg/linreg.h"
#include <memory>
#include <array>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <locale>
#include <string>
#include <iostream>
#include <boost/algorithm/string.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
using namespace std;
namespace athena{
String
execSysCall_block(const String& cmd)
{
    array<char, 128> buffer;
    String result;
    shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);
    if ( !pipe )
        throw runtime_error("popen() failed");

    while ( !feof(pipe.get()) ) {
        if ( fgets(buffer.data(), 128, pipe.get()) != nullptr )
            result += buffer.data();
    }

    result.erase(result.find('\n'));

    return result;
}

void sleepMilliSec(int num_ms)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(num_ms));
}

String
getHostName()
{
    char hn[128];
    int err = gethostname(hn, 128);

    if ( err<0 )
        throw runtime_error("cannot get host name");

    return String(hn);
}

vector<String>
splitString(const String& str, const String delimiters)
{
    vector<String> res;
    boost::split(res, str, [delimiters](char c) {
                 for (auto a : delimiters) {
                    if (c == a) return true;
                 }
                 return false;});
    return res;
}

bool
compareStringNoCase(const String& str1, const String& str2)
{
    return boost::iequals(str1,str2);
}

String
getStringFromPyobject(CPyObject& pyobj)
{
    PyObject* objrepr = PyObject_Repr(pyobj.getObject());
    if (!objrepr) {
        throw runtime_error("Failed to get string from pyobject");
    }

    const char* cp = PyBytes_AsString(objrepr);
    Py_XDECREF(objrepr);

    String s = String(cp);
    return s.substr(1,s.size()-2);
}

int
getIntFromPyobject(CPyObject& pyobj)
{
    PyObject* objrepr = PyObject_Repr(pyobj.getObject());
    if (!objrepr) {
        throw runtime_error("Failed to get string from pyobject");
    }

    const char* cp = PyBytes_AsString(objrepr);

    if (!cp) throw runtime_error("Get Null from PyBytes_AsString");
    int p = stoi(String(cp));
    Py_XDECREF(objrepr);
    return p;
}
NonBlockSysCall* gNBSysCall = &NonBlockSysCall::getInstance();

long
getTimeDiffInMin(const String& st1, const String& st2)
{
    boost::posix_time::ptime t1(boost::posix_time::time_from_string(st1));
    boost::posix_time::ptime t2(boost::posix_time::time_from_string(st2));

    boost::posix_time::time_duration td = t1 - t2;
    long diffmin = td.total_seconds()/60;

    return diffmin;
}

String
convertTimeString(const String& timeStr, const String& fmt)
{
    boost::posix_time::ptime t(boost::posix_time::time_from_string(timeStr));
    std::stringstream stream;
    boost::posix_time::time_facet* facet = new boost::posix_time::time_facet();
    facet->format(fmt.c_str());
    stream.imbue(std::locale(std::locale::classic(),facet));
    stream << t;

    String resStr = stream.str();
    delete facet;
    return resStr;
}

void
showMinBar(MinBar& mb)
{
    Log(LOG_INFO) << to_string(mb.open) + " " +
                         to_string(mb.high) + " " +
                         to_string(mb.low)  + " " +
                         to_string(mb.close) + " " +
                         to_string(mb.tickvol);
}

String
getFileFolder(const String& fp)
{
    boost::filesystem::path p(fp.c_str());
    boost::filesystem::path dir = p.parent_path();

    return dir.string();
}

String
getFileStem(const String& fp)
{
    boost::filesystem::path p(fp.c_str());
    boost::filesystem::path s = p.stem();

    return s.string();
}

void getPythonFunction(const String& modFile, const String& funcName,CPyObject& func)
{
    String mod_folder = getFileFolder(modFile);
    PyEnviron::getInstance().appendSysPath(mod_folder);
    String mod_name = getFileStem(modFile);
    CPyObject mod = PyImport_ImportModule(mod_name.c_str());
    if (!mod) {
        Log(LOG_FATAL) << "Failed to import module: " + modFile;
    }

    func = PyObject_GetAttrString(mod.getObject(), funcName.c_str());

    if (!func || !PyCallable_Check(func)) {
        Log(LOG_FATAL) << "Failed to get function or it's not callable: " + funcName;
    }

}

static void import_numpy()
{
//    static int a = 0;
//    if (a>0) return;
//    a++;
//    PyRunner& pyrun = PyRunner::getInstance();
//    (void)pyrun;
//    if(PyArray_API==NULL)
//        import_array();
}
bool
test_coint(std::vector<real32>& v1, std::vector<real32>& v2, real32 pval)
{
    CPyObject lx = PyList_New(v1.size());
    CPyObject ly = PyList_New(v2.size());
    CPyObject pv = Py_BuildValue("f",pval);
    for (size_t i = 0; i < v1.size(); i++) {
        PyList_SetItem(lx,i,Py_BuildValue("f",v1[i]));
        PyList_SetItem(ly,i,Py_BuildValue("f",v2[i]));
    }

    auto& pyrun = PyRunner::getInstance();

//    npy_intp dims[1];
//    dims[0] = v1.size();
//
//
//
//    import_numpy();
//
//    CPyObject pyv1 = PyArray_SimpleNewFromData(1,dims,NPY_FLOAT32,(void*)&v1[0]);
//    CPyObject pyv2 = PyArray_SimpleNewFromData(1,dims,NPY_FLOAT32,(void*)&v2[0]);

//    CPyObject args = Py_BuildValue("(OO)",pyv1.getObject(),pyv2.getObject());
    CPyObject args = PyTuple_New(3);
    PyTuple_SetItem(args.getObject(),0,lx.getObject());
    PyTuple_SetItem(args.getObject(),1,ly.getObject());
    PyTuple_SetItem(args.getObject(),2,pv.getObject());

    CPyObject res = pyrun.runAthenaPyFunc("coint","coint_verify",args);

    if (PyLong_AsLong(res) == 1) {
        return true;
    }

    return false;

}

bool
__test_coint(std::vector<real32>& v1, std::vector<real32>& v2)
{
    auto& pyrun=PyRunner::getInstance();

    String athenaHome = String(getenv("ATHENA_HOME"));
    String modulePath = athenaHome + "/pyapi";
    PyRun_SimpleString("import sys");
    std::string cmd = "sys.path.append(\'" + modulePath + "\')";
        PyRun_SimpleString(cmd.c_str());

    String modName = "coint";
    String funcName="test_test";

    PyObject* mod = PyImport_ImportModule(modName.c_str());
    if (!mod)
        Log(LOG_FATAL) << "Failed to import module: " + modName;

    CPyObject func = PyObject_GetAttrString(mod,funcName.c_str());
    if(!func)
        Log(LOG_FATAL) << "Failed to find py function: " + funcName;
    npy_intp dims[1];
    dims[0] = v1.size();

    static int a=0;
    if (a==0) {import_numpy();a++;}
//
    PyObject* pyv1 = PyArray_SimpleNewFromData(1,dims,NPY_FLOAT32,reinterpret_cast<void*>(&v1[0]));
    PyObject* pyv2 = PyArray_SimpleNewFromData(1,dims,NPY_FLOAT32,reinterpret_cast<void*>(&v2[0]));

//    CPyObject args = Py_BuildValue("(OO)",pyv1.getObject(),pyv2.getObject());
    PyObject* args = PyTuple_New(2);
    PyTuple_SetItem(args,0,pyv1);
    PyTuple_SetItem(args,1,pyv2);

    //CPyObject res = pyrun.runAthenaPyFunc("coint","coint_verify",args);

    CPyObject res = PyObject_CallObject(func,args);

    if (PyLong_AsLong(res) == 1) {
        return true;
    }

    Py_XDECREF(args);
    Py_XDECREF(mod);
    Py_XDECREF(pyv1);
    Py_XDECREF(pyv2);

    return false;
}

real64
testADF(real64* v, int len)
{
    PyObject* lx = PyList_New(len);
    for(int i=0; i < len; i++) {
        PyList_SetItem(lx,i,Py_BuildValue("d",v[i]));
    }
    PyObject* args = Py_BuildValue("(O)",lx);
//
    PyObject* res = PyRunner::getInstance().runAthenaPyFunc("coint","test_adf",args);
//
    return PyFloat_AsDouble(res);

}

real64
hurst(real64* v, int len)
{
    CPyObject lx = PyList_New(len);
    for(int i=0; i < len; i++) {
        PyList_SetItem(lx,i,Py_BuildValue("d",v[i]));
    }

    CPyObject args = Py_BuildValue("(O)",lx.getObject());

    CPyObject res = PyRunner::getInstance().runAthenaPyFunc("coint","hurst",args);

    return PyFloat_AsDouble(res);
}

real64
testADF(vector<real64>& vec, int start, int last)
{
    real64* v;
    int len;
    if (start < 0 || last < 0) {
        v = &vec[0];
        len = vec.size();
    } else {
        v = &vec[start];
        len = last-start;
    }
    real64 pv = testADF(v,len);

    return pv;
}

real64
compHalfLife(real64* y, int len)
{
    real64* dy = new real64[len-1];
    for (int i=0; i < len-1; i++) {
        dy[i] = y[i+1] - y[i];
    }

    LRParam pm = linreg(y,dy,len-1);
    delete[] dy;

    real64 lambda = -log(2.)/pm.c1;
    return lambda;
}

}
