package objectiveFunction;

//import lbfgsb.DifferentiableFunction;
//import lbfgsb.FunctionValues;

import optimize.DifferentiableFunction;
import optimize.FunctionValues;

import ALR.wdAnJE;

public abstract class ObjectiveFunction implements DifferentiableFunction {

	protected final wdAnJE algorithm;
	
	public ObjectiveFunction(wdAnJE algorithm) {
		this.algorithm = algorithm;
	}

	@Override
	abstract public FunctionValues getValues(double params[]);	
	
	public void finish(){
		
	}
	
}
