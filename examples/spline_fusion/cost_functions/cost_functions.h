#include "ceres/ceres.h"
#include <iostream>
using namespace std;

using ceres::CostFunction;
using ceres::SizedCostFunction;
using ceres::AutoDiffCostFunction;
using ceres::NumericDiffCostFunction;
using ceres::DynamicNumericDiffCostFunction;
using ceres::CENTRAL;

//NumericDiffCostFunction<NumericDiffCostFunctor, ceres::CENTRAL, 1, 1>


class SimpleCostFunction
  : public SizedCostFunction<1 /* number of residuals */,
                             1 /* size of first parameter */> {
 public:
  virtual ~SimpleCostFunction() {}
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    double x = parameters[0][0];

    // f(x) = 10 - x.
    residuals[0] = 10 - x;

    // f'(x) = -1. Since there's only 1 parameter and that parameter
    // has 1 dimension, there is only 1 element to fill in the
    // jacobians.
    if (jacobians != NULL && jacobians[0] != NULL) {
      jacobians[0][0] = -1;
    }
    return true;
  }
};

//struct CostFunctor2 {
//  template <typename T> bool operator()(const T* const x, T* residual) const {
//    residual[0] = T(10.0) - x[0];
//    return true;
//  }
//};


typedef void (*Method_1x1f8)(PyObject* pyfun, void* x0, void* residual, PyObject* args);
typedef void (*Method_2x1f8)(PyObject* pyfun, void* x0, void* x1, void* residual, PyObject* args);
typedef void (*Method_dyn_num_f8)(PyObject* pyfun, void* x0, void* residual, PyObject* args, int nParams, int nRes);


template<typename M>
struct  _PyCallBackBase{
	M				cb;
	PyObject*		pyfun;
	PyObject*		args;

	_PyCallBackBase(M cb, PyObject* pyfun, PyObject* args): cb(cb), pyfun(pyfun), args(args) {
		Py_INCREF(pyfun);
		Py_XINCREF(args);
	}
	~_PyCallBackBase(){
		Py_DECREF(pyfun);
		Py_XDECREF(args);
	}
};


struct  F_1x1{
	typedef Method_1x1f8	MethodType;
	_PyCallBackBase<MethodType>	_obj;

	F_1x1(MethodType cb, PyObject* pyfun, PyObject* args): _obj(cb,pyfun,args) {}
	  
	//bool operator()(const double* const x0, double* residual) const {
	template <typename T> bool operator()(const T* const x0, T* residual) const {
		_obj.cb(_obj.pyfun, (void*)x0, (void*)residual, _obj.args);
		return true;
	}
};


struct  F_Dyn{
	typedef Method_dyn_num_f8	MethodType;
	_PyCallBackBase<MethodType>	_obj;
	int nParams;
	int nRes;

	F_Dyn(MethodType cb, PyObject* pyfun, PyObject* args, int nParams, int nRes):
		_obj(cb,pyfun,args),
		nParams(nParams),
		nRes(nRes){
		
	}
	  
	template <typename T> bool operator()(T const* const* x0, T* residual) const {
		_obj.cb(_obj.pyfun, (void*)x0, (void*)residual, _obj.args, nParams, nRes);
		return true;
	}
};



struct  F_2x1{
	typedef Method_2x1f8	MethodType;
	_PyCallBackBase<MethodType>	_obj;

	F_2x1(MethodType cb, PyObject* pyfun, PyObject* args): _obj(cb,pyfun,args) {}
	  
	//bool operator()(const double* const x0,const double* const x1,  double* residual) const {
	template <typename T> bool operator()(const T* const x0,const T* const x1, T* residual) const {
		_obj.cb(_obj.pyfun, (void*)x0, (void*)x1, (void*)residual, _obj.args);
		return true;
	}
};





//CostFunction* createCostFunction(){
//  return new SimpleCostFunction();
//}



CostFunction* createCostFunAutoDiff_1x1f8(Method_1x1f8 cb, PyObject* pyfun, PyObject* args){	// fix it later!!
  return new AutoDiffCostFunction<F_1x1,1,1>(new F_1x1(cb, pyfun, args));
}

CostFunction* createCostFunAutoDiff_2x1f8(Method_2x1f8 cb, PyObject* pyfun, PyObject* args){	// fix it later!!
  return new AutoDiffCostFunction<F_2x1,1,1,1>(new F_2x1(cb, pyfun, args));
}


CostFunction* createNumericDiffCostFun_1x1f8(Method_1x1f8 cb, PyObject* pyfun, PyObject* args){
  return new NumericDiffCostFunction<F_1x1, CENTRAL,1,1>(new F_1x1(cb, pyfun, args));
}

CostFunction* createNumericDiffCostFun_2x1f8(Method_2x1f8 cb, PyObject* pyfun, PyObject* args){
  return new NumericDiffCostFunction<F_2x1, CENTRAL,1,1,1>(new F_2x1(cb, pyfun, args));
}

CostFunction* createDynamicNumericDiffCostFun(Method_dyn_num_f8 cb, PyObject* pyfun, PyObject* args, int param_num, int res_num){
  typedef DynamicNumericDiffCostFunction<F_Dyn> CostFunType;
  CostFunType* _cost = new CostFunType(new F_Dyn(cb, pyfun, args, param_num, res_num));
  _cost->AddParameterBlock(param_num);		// assume only one block
  _cost->SetNumResiduals(res_num);
  return _cost;
}

