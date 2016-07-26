package logDistributionComputation.W2;

import DataStructure.wdAnJEParameters;
import logDistributionComputation.LogDistributionComputerAnJE;
import weka.core.Instance;

public class A2JELogDistributionComputerW2 extends LogDistributionComputerAnJE{

	public static LogDistributionComputerAnJE singleton = null;
	
	protected A2JELogDistributionComputerW2(){}
	public static LogDistributionComputerAnJE getComputer() {
		if(singleton==null){
			singleton = new A2JELogDistributionComputerW2();
		}
		return singleton;
	}

	@Override
	public void compute(double[] probs, wdAnJEParameters params,Instance inst) {
		for (int c = 0; c < probs.length; c++) {
			probs[c] = params.getProbAtFullIndex(c) * params.getParameterAtFullIndex(c);
			double probsClass = 0;
			for (int att1 = 1; att1 < params.getNAttributes(); att1++) {
				int att1val = (int) inst.value(att1);

				for (int att2 = 0; att2 < att1; att2++) {
					int att2val = (int) inst.value(att2);

					long index = params.getAttributeIndex(att1, att1val, att2, att2val, c);
					probsClass += params.getProbAtFullIndex(index) * params.getParameterAtFullIndex(index);
				}
			}							
			probs[c] += probsClass;
		}
	}

}
