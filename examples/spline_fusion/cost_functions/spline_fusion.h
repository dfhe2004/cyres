//#pragma  once 

#include <iostream>
#include <vector>
#include <cmath>

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <pydbg.h>

#include "se3_spline.h"
#include "local_parameterization_se3.hpp"

using std::cout;
using std::endl;


struct SplineConstraint {
	typedef Eigen::Matrix<double, 3, 1> M3x1;
    typedef UniformSpline<double>       SplineType;
    //typedef UniformSpline<const double> SplineConstType;

    SplineConstraint(double ts_offset, double ts_step, size_t num_params): 
        ts_offset_(ts_offset), ts_step_(ts_step), num_params_(num_params){}

    void set_arr_ref(const double* ref, const double* ref_norm, const double* ref_ts){
        ref_ = ref;
        ref_norm_ = ref_norm;
        ref_ts_ = ref_ts;
    }
    
    void set_arr_src(const double* src, const double* src_ts){
        src_= src;
        src_ts_ = src_ts;
    }

    void set_arr_tiles(const int* tiles, size_t num_tiles){
        tiles_ = tiles;
        num_tiles_ = num_tiles;
    }

	size_t num_tiles() const {	return num_tiles_; }

    template<typename T>
    bool operator()(T const* const* parameters, T* residuals) const {
		// update spline
        SplineType _spline(ts_offset_, ts_step_);
        for (size_t i=0; i < num_params_; ++i) {
            _spline.add_knot((T*)(&parameters[i][0]));
        }
		
		_var_dump0(100);

		for (size_t i=0; i!=num_tiles_; ++i){
			size_t begin = tiles_[i*2];
			size_t end = tiles_[i*2+1];

			_var_dump3(101, i, begin, end );		//--dbg
			residuals[i] = _calculate_res(_spline, begin,end);
		}
		return true;
	}

	double _calculate_res(const SplineType& spline, size_t begin, size_t end) const{
		double rs = 0;
		for(size_t i=begin; i!=end; ++i){
			M3x1 cur_ref = M3x1(&ref_[i*3]);
			M3x1 cur_norm = M3x1(&ref_norm_[i*3]);
			M3x1 cur_src  = M3x1(&src_[i*3]);

			//_var_dump4(200, i, ref_, ref_norm_, src_);
			_var_dump4(201, i, cur_ref.data(), cur_norm.data(), cur_src.data());

            SE3Group<double> P0, P1;
            spline.evaluate( ref_ts_[i], P0);
            spline.evaluate( src_ts_[i], P1);
			
			_var_dump2(202, P0.matrix().data(), P1.matrix().data());

			//-- apply P0, P1 to cur_ref ,cur_norm and cur_src
			cur_ref  = P0*cur_ref;
			cur_src  = P1*cur_src;
			cur_norm = P0.so3()*cur_norm;
			
			
			_var_dump3(203, cur_ref.data(), cur_src.data(), cur_norm.data());
			
			cur_norm.normalize();
			
			_var_dump1(204, cur_norm.data());

			double diff = cur_norm.dot(cur_ref-cur_src); 
			rs += abs(diff);
			_var_dump2(205, &rs, &diff);
		}
		return rs;
    }

    double ts_offset_;
    double ts_step_;

	const double* ref_; 
	const double* ref_norm_;
	const double* ref_ts_;
		
	const double* src_; 
	const double* src_ts_;
			
	const int* tiles_;
	size_t num_tiles_;

    size_t num_params_;
};

void init_params(double* data, size_t num_parmas){
    UniformSpline<double>::SE3Type identity;
    for (size_t i=0; i < num_parmas; ++i) {
        memcpy(data, identity.data(), sizeof(data[0]) * Sophus::SE3d::num_parameters);
        data +=Sophus::SE3d::num_parameters; 
    }
}


typedef ceres::DynamicNumericDiffCostFunction<SplineConstraint> CostFunType; 

//template<typename T>
void run_solver(CostFunType* cost_func, ceres::Solver::Options& solver_options, double* params, size_t num_params, size_t num_res) {

    // One parameter block per spline knot
    std::vector<double*> parameter_blocks;
    const size_t _stride = Sophus::SE3d::num_parameters;

	for (size_t i=0; i < num_params; ++i) {
        parameter_blocks.push_back(&params[i*_stride]);
		const std::vector<int>& xx = cost_func->parameter_block_sizes();
		std::vector<int>* pxx = (std::vector<int>*)&xx;
		size_t vv = pxx->size();
		pxx->push_back(_stride);
		vv = pxx->size();
        //cost_func->AddParameterBlock(_stride); 
    }

	//_var_dump1(401, &parameter_blocks);

    // Set residual count
    cost_func->SetNumResiduals(num_res);
    
    ceres::Problem problem;

    // Local parameterization
    for (size_t i=0; i < num_params; ++i) {
        problem.AddParameterBlock(&params[i*_stride],
                                  _stride,
                                  new Sophus::test::LocalParameterizationSE3);
    }

    problem.AddResidualBlock(cost_func, NULL, parameter_blocks);	// debug later, try other loss functions!!

    ceres::Solver::Summary summary;
    cout << "Solving..." << endl;
    ceres::Solve(solver_options, &problem, &summary);
    cout << summary.FullReport() << endl;
}

void spline_fusion(
	const double* ref,		// for constrain
	const double* ref_norm,
	const double* ref_ts,
	const double* src,
	const double* src_ts,
	const int* tiles,
	size_t num_tiles,

	double ts_offset,		// for spline
	double ts_step,
	double* params_data,	// output [num_params,7]
	size_t num_params			 
	) {
    //-- prepare spline
    //UniformSpline<double> spline = create_zero_spline(ts_offset, ts_step, params_data, num_params);

	//-- prepare constrain
    SplineConstraint* constraint = new SplineConstraint(ts_offset, ts_step, num_params);
    constraint->set_arr_ref(ref, ref_norm, ref_ts);
    constraint->set_arr_src(src, src_ts);
    constraint->set_arr_tiles(tiles, num_tiles);

    init_params(params_data, num_params);


    // Solver options
    ceres::Solver::Options solver_options;
    solver_options.max_solver_time_in_seconds = 60;
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    solver_options.minimizer_progress_to_stdout = true;
    solver_options.parameter_tolerance = 1e-4;


    // 1) Numeric differentiation
    cout << "\n\n------------------------- NUMERIC ------------------------------------" << endl;
    CostFunType* cost_func_numeric = new CostFunType(constraint);
    
	_var_dump5(300, ref, ref_norm, ref_ts, src, src_ts);
	_var_dump4(301, tiles, num_tiles, &ts_offset, &ts_step);
	_var_dump2(302, params_data, num_params);

    run_solver(cost_func_numeric, solver_options, params_data, num_params, num_tiles);

}

