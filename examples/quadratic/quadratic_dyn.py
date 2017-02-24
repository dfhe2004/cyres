from cyres import *
from cost_functions.wrappers import DynNumDiffCostF
from IPython import embed



def foo(x,rs,arg):
    rs[0] = 10 - x[0]


x = np.array([5.])
problem = Problem()


problem.add_residual_block(DynNumDiffCostF(foo, 1,1,  diff_type='numeric'), SquaredLoss(), [x]) # auto doesn't work. fix it later!!!

options = SolverOptions()
options.max_num_iterations = 50
options.linear_solver_type = LinearSolverType.DENSE_QR
#options.trust_region_strategy_type = TrustRegionStrategyType.DOGLEG
#options.dogleg_type = DoglegType.SUBSPACE_DOGLEG
options.minimizer_progress_to_stdout = True

summary = Summary()

solve(options, problem, summary)
print summary.briefReport()
print summary.fullReport()
print x
