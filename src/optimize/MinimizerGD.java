package optimize;

public class MinimizerGD {

	private int np = 0; 
	private FunctionValues fv = null;
	private int totalFunctionEvaluations = 0;	 
	
	int maxIterations = 10000;
	
	public Result run(DifferentiableFunction fun, double[] parameters) {
	
		//FunctionValues fvO = null;
		np = parameters.length;
		
		//fvO = new FunctionValues(np);
		
		double eta = 0.0001;
		double precision = 0.1;
		double alphak = 1;
		double c = 1e-4;
		double rho = 0.5;
		
		int iterCounter = 0;
		
		boolean maxIterationsFlag = false;

		do {
			// [fk,gk] = feval(f,x)
			fv = fun.getValues(parameters);
			totalFunctionEvaluations++;
		
			double temp_fval = 0;
			double[] temp_fgradient = new double[np];
			double[] temp_parameters = new double[np];
			
			// Copy in buffer
			temp_fval = fv.functionValue;
			copyParameters(temp_fgradient, fv.gradient);
			copyParameters(temp_parameters, parameters);
			
			//fun.setDisplayStat(false);
			//alphak = linesearch(fun, fvO.getG(), startingPoint, rho, c);
			//alphak = linesearch(fun, fv.functionValue, fv.gradient, parameters, rho, c);
			alphak = linesearch(fun, temp_fval, temp_fgradient, temp_parameters, rho, c);
			
			fv.functionValue = temp_fval;
			copyParameters(fv.gradient, temp_fgradient);
			copyParameters(parameters, temp_parameters);
			
			//fun.setDisplayStat(true);
			//System.out.println("-----"+alphak+"-----");
			
			System.out.print(fv.functionValue + ", ");
			
			/*
			 * Update the parameters with searched alpha
			 */
			for (int i = 0; i < np ; i++) {
				//startingPoint[i] = startingPoint[i] - (alphak * fvO.getG()[i]);
				parameters[i] = parameters[i] - (alphak * fv.gradient[i]);
			}
			iterCounter++;
			
			if (iterCounter >= maxIterations) {
				maxIterationsFlag = true;
				break;
			}
			
		} while (gradientNorm(fv.gradient) >= precision);
		
		//fvO.setNumIter(iterCounter);
		//fvO.setParams(parameters);

		IterationsInfo info = null;
		
		if (maxIterationsFlag) {
			info = new IterationsInfo(iterCounter, totalFunctionEvaluations, IterationsInfo.StopType.MAX_ITERATIONS, null);	
		} else {
			info = new IterationsInfo(iterCounter, totalFunctionEvaluations, IterationsInfo.StopType.OTHER_STOP_CONDITIONS, null);
		}
		
		Result result = new Result(parameters, fv.functionValue, fv.gradient, info);
		return result;
	}
	
	private double linesearch(DifferentiableFunction f, double fval, double[] g, double[] p, double rho, double c) {
		
		// Direction is the negative of the gradient
		double[] d = new double[np];
		for (int i = 0; i < np; i++) {
			d[i] = -g[i] ;
		}
		
		//fv = new FunctionValues(np);
		//fv1 = new FunctionValues(np);
		
		// Copy parameters in the x and xx array
		double[] x = new double[np];
		//double[] xx = new double[np];
		
		copyParameters(x, p);
		//copyParameters(xx, p);
		
		// alphak = 1
		double alphak = 1;
		
		// [fk,gk] = feval(f,x)
		fv = f.getValues(x);
		totalFunctionEvaluations++;
		
		// x = x + alphak*d
		for (int i = 0; i < np; i++) {
			//x[i] = x[i] + alphak * d[i];
			x[i] = p[i] + alphak * d[i];
		}
		
		// fk1 = feval(f,x)
		fv = f.getValues(x);
		totalFunctionEvaluations++;
		
		// while fk1 > fk + c * alphak * (gk' * d)
		//while (fv1.functionValue > fv.functionValue + c * alphak * dot(fv.gradient, d)) {
		while (fv.functionValue > fval + c * alphak * dot(g, d)) {
			
			double a = fv.functionValue;
			double b = fval + c * alphak * dot(g, d);
			
			//System.out.println(a + ", " + b + ", " + alphak);
			
			// alphak = alphak * rho
			alphak = alphak * rho;
			
			// x = xx + alphak*d
			for (int i = 0; i < np; i++) {
				//x[i] = xx[i] + alphak * d[i];
				x[i] = p[i] + alphak * d[i];
			}
			fv = f.getValues(x);
			totalFunctionEvaluations++;
		}
		
		//System.out.println("Returning " + alphak);
		return alphak;
	}

	private double dot(double[] g, double[] g2) {
		double dp = 0;
		for (int i = 0; i < g.length; i++) {
			dp += g[i] * g2[i];
		}
		return dp;
	}

	private void copyParameters(double[] newparameters, double[] parameters) {
		for (int i = 0; i < parameters.length; i++) {
			newparameters[i] = parameters[i];
		}
	}

	private double gradientNorm(double[] g) {
		double gnorm = 0;
		for (int i = 0; i < np; i++) {
			gnorm = gnorm + (g[i] * g[i]);
		}
		//gnorm = Math.sqrt(gnorm);
		return gnorm;
	}

	public void setMaxIterations(int m_MaxIterations) {
		maxIterations = 	m_MaxIterations;
	}

}
