from cyres import *
from cost_functions.wrappers import SimpleCostF_2f8
from IPython import embed
import numpy


other = {'k': []}

def f1(a,b,c,args):
    c[0] = a[0] + 10*b[0]
    print '... f1 id|%s'%(id(args),)
    
def f2(a,b,c,args):
    c[0] = numpy.sqrt(5.)*(a[0]-b[0])
    print '... f2 id|%s'%(id(args),)
    
def f3(a,b,c,args):
    c[0] = (a[0]-2*b[0])
    c[0] *= c[0]
    print '... f3 id|%s'%(id(args),)
    
def f4(a,b,c,args):
    c[0] = numpy.sqrt(10.0)*(a[0] - b[0])
    print '... f4 id|%s'%(id(args),)



x1 = np.array([3.])
x2 = np.array([-1.])
x3 = np.array([0.])
x4 = np.array([1.])

problem = Problem()


problem.add_residual_block(SimpleCostF_2f8(f1, {}), SquaredLoss(), [x1,x2])
problem.add_residual_block(SimpleCostF_2f8(f2, {}), SquaredLoss(), [x3,x4])
problem.add_residual_block(SimpleCostF_2f8(f3, {}), SquaredLoss(), [x2,x3])
problem.add_residual_block(SimpleCostF_2f8(f4, {}), SquaredLoss(), [x1,x4])

#problem.add_residual_block(SimpleCostFx(foo, other), SquaredLoss(), [x])
#problem.add_residual_block(SimpleCostFunction2(), SquaredLoss(), [x])


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

print [x1,x2,x3,x4]