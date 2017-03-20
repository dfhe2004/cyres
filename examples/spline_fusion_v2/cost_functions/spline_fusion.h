//#pragma  once 

#include <iostream>
#include <vector>
#include <set>
#include <cmath>

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <pydbg.h>

#include "se3_spline.h"
#include "local_parameterization_se3.hpp"

using std::cout;
using std::endl;

static const size_t STRIDE = Sophus::SE3d::num_parameters;


struct SplineConstraint {
    typedef UniformSpline<double>   SplineType;
	typedef SplineType::Vec3		Vec3;
	typedef SplineType::Vec4		Vec4;
	typedef ceres::DynamicNumericDiffCostFunction<SplineConstraint> CostFunType;

    SplineConstraint(size_t tile_idx, size_t num_points): 
        tile_idx_(tile_idx), num_points_(num_points){}

    void set_arr_ref(const double* ref,const double* ref_ts){
        ref_	= ref;
        ref_ts_ = ref_ts;
    }
    
    void set_arr_src(const double* src, const double* src_norm,  const double* src_ts){
        src_		= src;
        src_norm_	= src_norm;
        src_ts_		= src_ts;
    }

	void set_arr_params(size_t num_params, const int* data, size_t num_data){
		for(size_t i=0; i!=num_data; ++i){
			ts2params_.push_back(data[i]);	// ref|src
	    }
        num_params_= num_params;
    }

	void prepare_spline_weights(){
		ref_w_.clear();
		src_w_.clear();

		for(size_t i=0; i!=num_points_; ++i){
			Vec4 val = spline_B(ref_ts_[i] - floor(ref_ts_[i]));
			ref_w_.push_back(val);

			val = spline_B(src_ts_[i] - floor(src_ts_[i]));
			src_w_.push_back(val);
		}
		_var_dump5(103, tile_idx_, num_points_, ref_, ref_ts_, ref_w_.data());
		_var_dump4(104, src_, src_norm_, src_ts_, src_w_.data());
		_var_dump3(105, num_params_, ts2params_.size(), ts2params_.data() );
	}


    template<typename T>
    bool operator()(T const* const* parameters, T* residuals) const {
        SplineType _spline;
		for (size_t i=0; i < num_params_; ++i) {
			_spline.add_knot((T*)parameters[i]);
        }
		
		_var_dump1(100, tile_idx_);
		for (size_t i=0; i!=num_points_; ++i ){
			_calculate_res(&residuals[3*i], i, _spline);
		}

		_var_dump1(102, residuals);
		return true;
	}


	//-- utils 
	void _calculate_res(double* res, size_t i, const SplineType& spline) const{
			Vec3 cur_ref  = Vec3(&ref_[i*3]);
			Vec3 cur_src  = Vec3(&src_[i*3]);

			_var_dump3(201, i, cur_ref.data(), cur_src.data());

            SE3Group<double> P0, P1;
            spline.evaluate(P0, ts2params_[0],	src_w_[i]);		// src 
            spline.evaluate(P1, ts2params_[i+1],ref_w_[i]);
			
			_var_dump5(202, &ts2params_[0], &src_w_[i], &ref_w_[i], P0.data(), P1.data());

			//-- apply P0, P1 to cur_ref ,cur_norm and cur_src
			cur_src  = P0*cur_src;
			cur_ref  = P1*cur_ref;
			
			_var_dump2(203, cur_ref.data(), cur_src.data());
			
			Vec3 diff = cur_ref-cur_src;
			res[0] = diff(0);
			res[1] = diff(1);
			res[2] = diff(2);
    }

	const double* ref_; 
	const double* ref_ts_;
		
	const double* src_; 
	const double* src_norm_;
	const double* src_ts_;

	size_t num_points_;
    size_t num_params_;
	size_t tile_idx_;		// only for debug

	std::vector<int>  ts2params_;
	std::vector<Vec4> ref_w_;
	std::vector<Vec4> src_w_;
};


struct SplineFusion{
	
	SplineFusion(size_t max_solver_time=60){
		max_solver_time_ = max_solver_time;
	}

	void set_ref(const double* ref, const double* ref_ts){
		ref_ = ref;
		ref_ts_ = ref_ts;
	}
	
	void set_src(const double* src, const double* src_norm, const double* src_ts){
		src_ = src;
		src_norm_ = src_norm;
		src_ts_ = src_ts;
	}

	void set_tile(const int* tiles, size_t num){
		tiles_ = tiles;
		num_tiles_ = num;
	}

	void set_params(double* data, size_t num){
		params_data_= data;
		num_params_= num;
	}

	void add_cost_functor(size_t tile_idx, const int* params_idx, size_t num_params_idx, const int* ts2param, size_t num_ts2params);

	void set_local_parameterization(const int* idx, size_t num);

	void run_solver();

	const double* ref_;		// for constrain
	const double* ref_ts_;
	const double* src_;
	const double* src_norm_;
	const double* src_ts_;
	const int* tiles_;
	size_t num_tiles_;

	double* params_data_;	// output [num_params,7]
	size_t num_params_;
	
	//-- ceres		
	size_t max_solver_time_;
	ceres::Problem problem_;
};


void SplineFusion::add_cost_functor(size_t tile_idx, const int* params_idx, size_t num_params_idx, const int* ts2params, size_t num_ts2params){
	const size_t begin = tiles_[tile_idx*2];
	const size_t end = tiles_[tile_idx*2+1];
			
	SplineConstraint* constraint = new SplineConstraint(tile_idx, end-begin);
	constraint->set_arr_ref(&ref_[begin*3], &ref_ts_[begin]);
	constraint->set_arr_src(&src_[begin*3], &src_norm_[begin*3], &src_ts_[begin]);
	constraint->set_arr_params(num_params_idx, ts2params, num_ts2params);
	constraint->prepare_spline_weights();

	auto cost_functor = new SplineConstraint::CostFunType(constraint);
	
	std::vector<double*> param_blocks;
	int at = 0;
	for(size_t i=0; i!=num_params_idx; ++i){
		at = params_idx[i]*STRIDE;
		param_blocks.push_back(&params_data_[at]);
        cost_functor->AddParameterBlock(STRIDE); 
	}
    cost_functor->SetNumResiduals(3*(end-begin));
    problem_.AddResidualBlock(cost_functor, NULL, param_blocks);	// debug later, try other loss functions!!
}


void SplineFusion::set_local_parameterization(const int* idx, size_t num){
    int at = 0;
	for (size_t i=0; i < num; ++i) {
		at = idx[i]*STRIDE;
        problem_.AddParameterBlock(&params_data_[at],
                                  STRIDE,
                                  new Sophus::test::LocalParameterizationSE3);
    }
}

void SplineFusion::run_solver(){
    // Solver options
    ceres::Solver::Options solver_options;
    solver_options.max_solver_time_in_seconds = max_solver_time_;
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    solver_options.minimizer_progress_to_stdout = true;
    solver_options.parameter_tolerance = 1e-4;
    //solver_options.max_num_iterations = 500;
    //solver_options.num_threads = 1;


    // 1) Numeric differentiation
    cout << "\n\n------------------------- NUMERIC ------------------------------------" << endl;
    
	_var_dump5(300, ref_, ref_ts_, src_, src_norm_, src_ts_);
	_var_dump4(301, tiles_, num_tiles_, params_data_, num_params_);

    ceres::Solver::Summary summary;
    cout << "Solving..." << endl;
    ceres::Solve(solver_options, &problem_, &summary);
    cout << summary.FullReport() << endl;
}

