package logDistributionComputation.MAP2;

import DataStructure.wdAnJEParameters;
import logDistributionComputation.LogDistributionComputerAnJE;
import weka.core.Instance;

public class A1JELogDistributionComputerMAP2 extends LogDistributionComputerAnJE {

	public static LogDistributionComputerAnJE singleton = null;
	
	protected A1JELogDistributionComputerMAP2(){}
	
	public static LogDistributionComputerAnJE getComputer() {
		if(singleton == null){
			singleton = new A1JELogDistributionComputerMAP2();
		}
		return singleton;
	}

//	@Override
//	public void compute(double[] probs, wdAnJEParameters params, Instance inst) {
//		
//		for (int c = 0; c < probs.length; c++) {
//			//probs[c] = Math.log(params.getCountAtFullIndex(c));
//			
//			for (int att1 = 0; att1 < params.getNAttributes(); att1++) {
//				int att1val = (int) inst.value(att1);
//
//				long index = params.getAttributeIndex(att1, att1val, c);
//				probs[c] += Math.log(params.getCountAtFullIndex(index));				
//				
////				// Compute P(x)
////				int sumVal = 0;
////				for (int y = 0; y < probs.length; y++) {
////					long tempIndex = params.getAttributeIndex(att1, att1val, y);
////					sumVal += params.getCountAtFullIndex(tempIndex);
////				}
////				double prob = Math.log(Math.max(SUtils.MEsti(sumVal, 67557, 1), 1e-75));
////				
////				probs[c] += prob;
//			}
//			
//			probs[c] -=  ((params.getNAttributes() - 1) * Math.log(params.getCountAtFullIndex(c)));// - Math.log(67557);
//		}		
//	}
	
	@Override
	public void compute(double[] probs, wdAnJEParameters params, Instance inst) {
		
		for (int c = 0; c < probs.length; c++) {
			
			for (int att1 = 0; att1 < params.getNAttributes(); att1++) {
				int att1val = (int) inst.value(att1);

				long index = params.getAttributeIndex(att1, att1val, c);
				probs[c] += params.getProbAtFullIndex(index);				
			}
			
			probs[c] -=  ((params.getNAttributes() - 1) * params.getProbAtFullIndex(c));
		}		
	}

}
