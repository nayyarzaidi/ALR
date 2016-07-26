package optimize;

public class MinimizerCG {

	// Max iterations
	private int maxIterations = 10000;
	
	private FunctionValues fv = null;
	private int totalFunctionEvaluations = 0;	 

	// extrapolate maximum 3 times the current bracket.
	// this can be set higher for bigger extrapolations
	public static double EXT = 3.0;

	// a bunch of constants for line searches
	private static final double RHO = 0.01;
	// RHO and SIG are the constants in the Wolfe-Powell conditions
	private static final double SIG = 0.5;
	// don't reevaluate within 0.1 of the limit of the current bracket
	private static final double INT = 0.1;
	// max 20 function evaluations per line search
	private static final int MAX = 20;
	// maximum allowed slope ratio
	private static final int RATIO = 100;
	
	private boolean verbose = true;

	public Result run(DifferentiableFunction fun, double[] parameters) {

		DoubleVector theta = new DenseDoubleVector(parameters);

		DoubleVector input = theta;

		int M = 0;
		int i = 0; // zero the run length counter
		int red = 1; // starting point
		int ls_failed = 0; // no previous line search has failed

		/*
	    final CostGradientTuple evaluateCost = f.evaluateCost(input);
	    double f1 = evaluateCost.getCost();
	    DoubleVector df1 = evaluateCost.getGradient();
		 */

		fv = fun.getValues(input.toArray());
		double f1 = fv.functionValue;
		DoubleVector df1 = new DenseDoubleVector(fv.gradient); 

		i = i + (maxIterations < 0 ? 1 : 0);

		// search direction is steepest
		DoubleVector s = df1.multiply(-1.0d);

		double d1 = s.multiply(-1.0d).dot(s); // this is the slope
		double z1 = red / (1.0 - d1); // initial step is red/(|s|+1)

		while (i < Math.abs(maxIterations)) {	// while not finished
			i = i + (maxIterations > 0 ? 1 : 0);	// count iterations?!

			// make a copy of current values
			DoubleVector X0 = input.deepCopy();
			double f0 = f1;
			DoubleVector df0 = df1.deepCopy();

			// begin line search
			input = input.add(s.multiply(z1));

			/*
			final CostGradientTuple evaluateCost2 = f.evaluateCost(input);
			double f2 = evaluateCost2.getCost();
			DoubleVector df2 = evaluateCost2.getGradient();
			 */

			fv = fun.getValues(input.toArray());
			double f2 = fv.functionValue;
			DoubleVector df2 = new DenseDoubleVector(fv.gradient);

			i = i + (maxIterations < 0 ? 1 : 0); // count epochs
			double d2 = df2.dot(s);
			// initialize point 3 equal to point 1
			double f3 = f1;
			double d3 = d1;
			double z3 = -z1;
			if (maxIterations > 0) {
				M = MAX;
			} else {
				M = Math.min(MAX, -maxIterations - i);
			}
			// initialize quanteties
			int success = 0;
			double limit = -1;

			while (true) {
				while (((f2 > f1 + z1 * RHO * d1) | (d2 > -SIG * d1)) && (M > 0)) {
					// tighten the bracket
					limit = z1;
					double z2 = 0.0d;
					double A = 0.0d;
					double B = 0.0d;
					if (f2 > f1) {
						// quadratic fit
						z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3);
					} else {
						// cubic fit
						A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);
						B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
						// numerical error possible - ok!
						z2 = (Math.sqrt(B * B - A * d2 * z3 * z3) - B) / A;
					}
					if (Double.isNaN(z2) || Double.isInfinite(z2)) {
						// if we had a numerical problem then bisect
						z2 = z3 / 2.0d;
					}
					// don't accept too close to limits
					z2 = Math.max(Math.min(z2, INT * z3), (1 - INT) * z3);
					// update the step
					z1 = z1 + z2;
					input = input.add(s.multiply(z2));

					/*
					final CostGradientTuple evaluateCost3 = f.evaluateCost(input);
					f2 = evaluateCost3.getCost();
					df2 = evaluateCost3.getGradient();
					 */
					fv = fun.getValues(input.toArray());
					f2 = fv.functionValue;
					df2 = new DenseDoubleVector(fv.gradient);

					M = M - 1;
					i = i + (maxIterations < 0 ? 1 : 0); // count epochs
					d2 = df2.dot(s);
					// z3 is now relative to the location of z2
					z3 = z3 - z2;
				}

				if (f2 > f1 + z1 * RHO * d1 || d2 > -SIG * d1) {
					break; // this is a failure
				} else if (d2 > SIG * d1) {
					success = 1;
					break; // success
				} else if (M == 0) {
					break; // failure
				}

				// make cubic extrapolation
				double A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);
				double B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
				double z2 = -d2 * z3 * z3 / (B + Math.sqrt(B * B - A * d2 * z3 * z3));

				// num prob or wrong sign?
				if (Double.isNaN(z2) || Double.isInfinite(z2) || z2 < 0)
					// if we have no upper limit
					if (limit < -0.5) {
						// the extrapolate the maximum amount
						z2 = z1 * (EXT - 1);
					} else {
						// otherwise bisect
						z2 = (limit - z1) / 2;
					}
				else if ((limit > -0.5) && (z2 + z1 > limit)) {
					// extraplation beyond max?
					z2 = (limit - z1) / 2; // bisect
				} else if ((limit < -0.5) && (z2 + z1 > z1 * EXT)) {
					// extrapolationbeyond limit
					z2 = z1 * (EXT - 1.0); // set to extrapolation limit
				} else if (z2 < -z3 * INT) {
					z2 = -z3 * INT;
				} else if ((limit > -0.5) && (z2 < (limit - z1) * (1.0 - INT))) {
					// too close to the limit
					z2 = (limit - z1) * (1.0 - INT);
				}

				// set point 3 equal to point 2
				f3 = f2;
				d3 = d2;
				z3 = -z2;
				z1 = z1 + z2;

				// update current estimates
				input = input.add(s.multiply(z2));

				/*
				final CostGradientTuple evaluateCost3 = f.evaluateCost(input);
				f2 = evaluateCost3.getCost();
				df2 = evaluateCost3.getGradient();
				 */

				fv = fun.getValues(input.toArray());
				f2 = fv.functionValue;
				df2 = new DenseDoubleVector(fv.gradient);

				M = M - 1;
				i = i + (maxIterations < 0 ? 1 : 0); // count epochs?!
				d2 = df2.dot(s);
			} // end of line search

			DoubleVector tmp = null;

			if (success == 1) { // if line search succeeded
				f1 = f2;
				if (verbose) {
					//LOG.info("Iteration " + i + " | Cost: " + f1);
					//System.out.println("Iteration " + i + " | Cost: " + f1);
					System.out.print(f1 + ", ");
				}
				
				// Commenting this, because, I am not sure about this.
				//onIterationFinished(i, f1, input);

				// Polack-Ribiere direction: s =
				// (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;
				final double numerator = (df2.dot(df2) - df1.dot(df2)) / df1.dot(df1);
				s = s.multiply(numerator).subtract(df2);
				tmp = df1;
				df1 = df2;
				df2 = tmp; // swap derivatives
				d2 = df1.dot(s);
				if (d2 > 0) { // new slope must be negative
					s = df1.multiply(-1.0d); // otherwise use steepest direction
					d2 = s.multiply(-1.0d).dot(s);
				}
				// realmin in octave = 2.2251e-308
				// slope ratio but max RATIO
				z1 = z1 * Math.min(RATIO, d1 / (d2 - 2.2251e-308));
				d1 = d2;
				ls_failed = 0; // this line search did not fail
			} else {
				input = X0;
				f1 = f0;
				df1 = df0; // restore point from before failed line search
				// line search failed twice in a row?
				if (ls_failed == 1 || i > Math.abs(maxIterations)) {
					break; // or we ran out of time, so we give up
				}
				tmp = df1;
				df1 = df2;
				df2 = tmp; // swap derivatives
				s = df1.multiply(-1.0d); // try steepest
				d1 = s.multiply(-1.0d).dot(s);
				z1 = 1.0d / (1.0d - d1);
				ls_failed = 1; // this line search failed
			}

		}

		IterationsInfo info = null;
		info = new IterationsInfo(i, totalFunctionEvaluations, IterationsInfo.StopType.MAX_ITERATIONS, null);	
		
		Result result = new Result(input.toArray(), fv.functionValue, fv.gradient, info);
		return result;
	}
	
	public void setMaxIterations(int m_MaxIterations) {
		maxIterations = 	m_MaxIterations;
	}

}
