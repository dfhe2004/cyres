#ifndef SE3_SPLINE_H
#define SE3_SPLINE_H

#include <vector>
#include <map>
#include <iostream>
#include <cstring>
using std::cout;
using std::endl;
using std::memcpy;
#include <math.h>

#include <ceres/jet.h>

#include <sophus/se3.hpp>
#include <Eigen/Dense>
using Sophus::SE3Group;
using Sophus::SE3;

const Eigen::Matrix4d C = (Eigen::Matrix4d() << 6.0 / 6.0 , 0.0       , 0.0        , 0.0        ,
        5.0 / 6.0 , 3.0 / 6.0 , -3.0 / 6.0 , 1.0 / 6.0  ,
        1.0 / 6.0 , 3.0 / 6.0 , 3.0 / 6.0  , -2.0 / 6.0 ,
        0.0       , 0.0       , 0.0        ,  1.0 / 6.0).finished();

template<typename T>
Eigen::Matrix<T, 4, 1> spline_B(T u) {
    Eigen::Matrix<T, 4, 1> U(T(1.0), u, u * u, u * u * u);
    return C.cast<T>() * U;
}


template<typename T>
class UniformSpline {
public:
    typedef SE3Group<T> SE3Type;
    typedef Eigen::Matrix<T, 4, 4> SE3DerivType;
    typedef Eigen::Matrix<T, 3, 1> Vec3;

    size_t num_knots() const { return knots_.size(); }
    void add_knot(T* data){
		knots_.push_back(data);
	}

    Eigen::Map<SE3Type> get_knot(size_t i) const {
        return Eigen::Map<SE3Type>(knots_[i]);
    }

    double* get_knot_data(size_t i) const{
		if (i < knots_.size()) {
            return knots_[i];
        }
        else {
            throw std::out_of_range("Knot does not exist");
        }
    }


    /** Evaluate spline (pose and its derivative)
     * This gives the current pose and derivative of the spline.
     * The Pose P = [R | t] is such that it moves a point from
     * the spline coordinate frame, to the world coordinate frame.
     * X_world = P X_spline
     */
    void evaluate(T t, SE3Type& P) const;

    double min_time() const {
		return 0.0;
    };
	
    double max_time() const {
        if (num_knots() > 0)
            return num_knots() - 3;
        else
            return 0.0;
    };

protected:
	std::vector<T*> knots_;
};

template<typename T>
void UniformSpline<T>::evaluate(T t, SE3Type &P) const {
    typedef Eigen::Matrix<T, 4, 4> Mat4;
    typedef Eigen::Matrix<T, 4, 1> Vec4;
    typedef Eigen::Map<SE3Type> KnotMap;
	
	// Remove offset
	size_t i0 = floor(t);			 // [0,n-3)
    T u = t - i0;				

    P = Eigen::Map<SE3Type>(knots_[i0]);
    Vec4 B = spline_B(u);

    for(size_t j=1; j!=4; ++j) {
        KnotMap knot1 = Eigen::Map<SE3Type>(knots_[i0+j-1]);
        KnotMap knot2 = Eigen::Map<SE3Type>(knots_[i0+j]);
        typename SE3Type::Tangent omega = SE3Type::log(knot1.inverse() * knot2);
        Mat4 omega_hat = SE3Type::hat(omega);
        SE3Type Aj = SE3Type::exp(B(j) * omega);
        P *= Aj;
    }
}

void spline_evaluate(double* out, double* se3, const double* xyz, const double* weights,  const double* knots){
    typedef UniformSpline<double>::SE3Type	_SE3Type;
	typedef Eigen::Map<_SE3Type>			_MapType;
	typedef Eigen::Matrix<double, 3, 1>		_M3x1;
	
	const size_t stride = _SE3Type::num_parameters;
	
	_SE3Type P = _MapType((double*)knots);		// inplace ??
	knots += stride;
    for(size_t j=1; j!=4; ++j) {
        _SE3Type knot1 = _MapType((double*)(knots-stride));
        _SE3Type knot2 = _MapType((double*)knots);
        _SE3Type::Tangent omega = _SE3Type::log(knot1.inverse() * knot2);
        Eigen::Matrix<double, 4, 4> omega_hat = _SE3Type::hat(omega);
        _SE3Type Aj = _SE3Type::exp(weights[j] * omega);
        P *= Aj;
		knots += stride;
    }
	
	_M3x1 pt = _M3x1(xyz);
	pt = P*pt;

	memcpy(out, pt.data(), sizeof(pt));
	if (se3!=NULL){
		memcpy(se3, P.data(), sizeof(se3[0])*_SE3Type::num_parameters );
	}
}



#endif //SE3_SPLINE_H
