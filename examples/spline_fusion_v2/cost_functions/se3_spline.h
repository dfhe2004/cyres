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
    typedef SE3Group<T>			SE3Type;
	typedef Eigen::Map<SE3Type> MapType;
    typedef Eigen::Matrix<T, 4, 4> Mat4;
    typedef Eigen::Matrix<T, 4, 1> Vec4;
    typedef Eigen::Matrix<T, 3, 1> Vec3;

    void add_knot(T* data){
		knots_.push_back(MapType(data));
	}

    /** Evaluate spline (pose and its derivative)
     * This gives the current pose and derivative of the spline.
     * The Pose P = [R | t] is such that it moves a point from
     * the spline coordinate frame, to the world coordinate frame.
     * X_world = P X_spline
     */
    void evaluate(SE3Type& P, size_t i, const Vec4& weights) const;


protected:
	std::vector<SE3Type> knots_;
};

template<typename T>
void UniformSpline<T>::evaluate(SE3Type& P, size_t i, const Vec4& weights) const {
    P = knots_[i];				

    for(size_t j=1; j!=4; ++j) {
        SE3Type knot1 = knots_[i+j-1]; 
        SE3Type knot2 = knots_[i+j];	
        typename SE3Type::Tangent omega = SE3Type::log(knot1.inverse() * knot2);
        Mat4 omega_hat = SE3Type::hat(omega);
        SE3Type Aj = SE3Type::exp(weights(j) * omega);
        P *= Aj;
    }
}

void spline_evaluate(double* out, double* se3, const double* xyz, const double* weights,  const double* knots){
    typedef UniformSpline<double>	_SplineType;
	typedef _SplineType::SE3Type	SE3Type;
	typedef _SplineType::MapType	MapType;
	typedef _SplineType::Vec3		Vec3;
	
	const size_t stride = SE3Type::num_parameters;
	
	SE3Type P = MapType((double*)knots);		// inplace ??
	knots += stride;
    for(size_t j=1; j!=4; ++j) {
        SE3Type knot1 = MapType((double*)(knots-stride));
        SE3Type knot2 = MapType((double*)knots);
        SE3Type::Tangent omega = SE3Type::log(knot1.inverse() * knot2);
        Eigen::Matrix<double, 4, 4> omega_hat = SE3Type::hat(omega);
        SE3Type Aj = SE3Type::exp(weights[j] * omega);
        P *= Aj;
		knots += stride;
    }
	
	Vec3 pt = P*Vec3(xyz);
	memcpy(out, pt.data(), sizeof(pt));
	if (se3!=NULL){
		memcpy(se3, P.data(), sizeof(se3[0])*SE3Type::num_parameters );
	}
}

void se3_to_matrix(double* out, double* se3){
    typedef UniformSpline<double>::MapType	MapType; 
	memcpy(out, MapType(se3).matrix().data(), sizeof(se3[0])*16);
}

#endif //SE3_SPLINE_H
