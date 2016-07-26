package logDistributionComputation.W2;

import DataStructure.wdAnJEParameters;
import logDistributionComputation.LogDistributionComputerAnJE;
import weka.core.Instance;

public class A4JELogDistributionComputerW2 extends LogDistributionComputerAnJE{

	public static LogDistributionComputerAnJE singleton = null;
	
	protected A4JELogDistributionComputerW2(){}
	public static LogDistributionComputerAnJE getComputer() {
		if(singleton==null){
			singleton = new A4JELogDistributionComputerW2();
		}
		return singleton;
	}

	@Override
	public void compute(double[] probs, wdAnJEParameters params,Instance inst) {
		for (int c = 0; c < probs.length; c++) {
			probs[c] = params.getProbAtFullIndex(c) * params.getParameterAtFullIndex(c);
			double probsClass = 0;
			for (int att1 = 3; att1 < params.getNAttributes(); att1++) {
				int att1val = (int) inst.value(att1);

				for (int att2 = 2; att2 < att1; att2++) {
					int att2val = (int) inst.value(att2);

					for (int att3 = 1; att3 < att2; att3++) {
						int att3val = (int) inst.value(att3);

						for (int att4 = 0; att4 < att3; att4++) {
							int att4val = (int) inst.value(att4);

							long index = params.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, att4, att4val, c);
							probsClass += params.getProbAtFullIndex(index) * params.getParameterAtFullIndex(index);
						}
					}
				}
			}
			probs[c] += probsClass;
		}
	}

}
