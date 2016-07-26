package DataStructure;

import java.util.concurrent.Callable;

import Utils.SUtils;

import weka.core.Instance;
import weka.core.Instances;

public class CallableWdAnJEParametersFlat implements Callable<Double> {

	private Instances data;
	private int start;
	private int stop;

	private double Error;
	private int[] XYCount;	

	private int numTuples;
	private wdAnJEParametersFlat dParameters;
	
	private int threadID;

	public CallableWdAnJEParametersFlat(int start, int stop, Instances data, int numTuples, int[] XYCount, wdAnJEParametersFlat dParameters, int th) {
		this.data = data;
		this.start = start;
		this.stop = stop;

		this.XYCount = XYCount;
		this.numTuples = numTuples;
		this.dParameters = dParameters;
		
		this.threadID = th;
	}

	@Override
	public Double call() throws Exception {

		int n = dParameters.n;
		
		int numProcessed = 0;

		for (int j = start; j <= stop; j++) {
			Instance inst = data.instance(j);

			int x_C = (int) inst.classValue();
			dParameters.incCountAtFullIndex(XYCount, x_C);

			if (numTuples == 1) {

				for (int u1 = 0; u1 < n; u1++) {
					int x_u1 = (int) inst.value(u1);

					int index = (int) dParameters.getAttributeIndex(u1, x_u1, x_C);
					dParameters.incCountAtFullIndex(XYCount, index);
				}

			} else if (numTuples == 2) {

				for (int u1 = 1; u1 < n; u1++) {
					int x_u1 = (int) inst.value(u1);

					for (int u2 = 0; u2 < u1; u2++) {
						int x_u2 = (int) inst.value(u2);

						int index = (int) dParameters.getAttributeIndex(u1, x_u1, u2, x_u2, x_C);
						dParameters.incCountAtFullIndex(XYCount, index);
					}
				}

			} else if (numTuples == 3) {

				for (int u1 = 2; u1 < n; u1++) {
					int x_u1 = (int) inst.value(u1);

					for (int u2 = 1; u2 < u1; u2++) {
						int x_u2 = (int) inst.value(u2);

						for (int u3 = 0; u3 < u2; u3++) {
							int x_u3 = (int) inst.value(u3);

							int index = (int) dParameters.getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C);
							dParameters.incCountAtFullIndex(XYCount, index);
						}
					}
				}
			} else if (numTuples == 4) {

				for (int u1 = 3; u1 < n; u1++) {
					int x_u1 = (int) inst.value(u1);

					for (int u2 = 2; u2 < u1; u2++) {
						int x_u2 = (int) inst.value(u2);

						for (int u3 = 1; u3 < u2; u3++) {
							int x_u3 = (int) inst.value(u3);

							for (int u4 = 0; u4 < u3; u4++) {
								int x_u4 = (int) inst.value(u4);

								int index = (int) dParameters.getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, u4, x_u4, x_C);
								dParameters.incCountAtFullIndex(XYCount, index);
							}
						}
					}
				}
			} else if (numTuples == 5) {

				for (int u1 = 4; u1 < n; u1++) {
					int x_u1 = (int) inst.value(u1);

					for (int u2 = 3; u2 < u1; u2++) {
						int x_u2 = (int) inst.value(u2);

						for (int u3 = 2; u3 < u2; u3++) {
							int x_u3 = (int) inst.value(u3);

							for (int u4 = 1; u4 < u3; u4++) {
								int x_u4 = (int) inst.value(u4);

								for (int u5 = 0; u5 < u4; u5++) {
									int x_u5 = (int) inst.value(u5);

									int index = (int) dParameters.getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, u4, x_u4, u5, x_u5, x_C);
									dParameters.incCountAtFullIndex(XYCount, index);
								}
							}
						}
					}
				}
			}
			
			numProcessed++;
			if ((numProcessed % SUtils.displayPerfAfterInstances) == 0) {
				//System.out.print(perfOutput.charAt(threadID));
				System.out.print(SUtils.perfOutput.charAt(threadID));
			}

		} // ends j

		return Error;
	}

}
