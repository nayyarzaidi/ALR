package logDistributionComputation.W;

import DataStructure.wdAnJEParameters;
import logDistributionComputation.LogDistributionComputerAnJE;
import weka.core.Instance;

public class A1JELogDistributionComputerW extends LogDistributionComputerAnJE{

	public static LogDistributionComputerAnJE singleton = null;
	
	protected A1JELogDistributionComputerW(){}
	public static LogDistributionComputerAnJE getComputer() {
		if(singleton==null){
			singleton = new A1JELogDistributionComputerW();
		}
		return singleton;
	}

	@Override
	public void compute(double[] probs, wdAnJEParameters params,Instance inst) {
			
		for (int c = 0; c < probs.length; c++) {
			probs[c] = params.getProbAtFullIndex(c) * params.getParameterAtFullIndex(c);
			for (int att1 = 0; att1 < params.getNAttributes(); att1++) {
				int att1val = (int) inst.value(att1);

				long index = params.getAttributeIndex(att1, att1val, c);
				probs[c] += params.getProbAtFullIndex(index) * params.getParameterAtFullIndex(index);
			}
		}
	}

}
