#ifndef SE3_SPLINE_H
#define SE3_SPLINE_H

#include <vector>
#include <map>
#include <iostream>
using std::cout;
using std::endl;
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

    UniformSpline(const double offset=0.0, const double dt=1.0) : offset_(offset), dt_(dt){};

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

    double get_dt() const { return dt_; };

    double get_offset() const { return offset_; };

    double min_time() const {
        if (num_knots() > 0)
            return offset_;
        else
            return 0.0;
    };
	
    double max_time() const {
        if (num_knots() > 0)
            return offset_ + (dt_ * (num_knots() - 3));
        else
            return 0.0;
    };

protected:
    double dt_;
    double offset_;
	std::vector<T*> knots_;
};

template<typename T>
void UniformSpline<T>::evaluate(T t, SE3Type &P) const {
    typedef Eigen::Matrix<T, 4, 4> Mat4;
    typedef Eigen::Matrix<T, 4, 1> Vec4;
    typedef Eigen::Map<SE3Type> KnotMap;
	
	// Remove offset
    T s = (t - T(offset_)) / T(dt_); // Spline normalized time (offset aware)
    T u = s - floor(s);				
	size_t i0 = floor(s);			 // [0,n-3)

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

/*
void spline_evaluate(double* out, const double* weights,  const double* knots){
    const size_t stride = SE3Type::num_parameters;
	//const double* itr = knots+i*stride;
	
	SE3Type P = Eigen::Map<SE3Type>(knots);		// inplace ??
	knots += stride;
    for(size_t j=1; j!=4; ++j) {
        Eigen::Map<SE3Type> knot1 = Eigen::Map<SE3Type>(knots-stride);
        Eigen::Map<SE3Type> knot2 = Eigen::Map<SE3Type>(knots);
        typename SE3Type::Tangent omega = SE3Type::log(knot1.inverse() * knot2);
        Eigen::Matrix<T, 4, 4> omega_hat = SE3Type::hat(omega);
        SE3Type Aj = SE3Type::exp(weight[j] * omega);
        P *= Aj;
		knots += stride;
    }
	memcopy(out, P.data(), sizeof(out[0])*SE3Type::num_parameters);
}
*/

#endif //SE3_SPLINE_H
