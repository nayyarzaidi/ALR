package ALR;

import java.util.concurrent.Callable;

import weka.classifiers.Classifier;
import Utils.SUtils;
import weka.core.Instances;

public class CallableBVDCrossvalx implements Callable<Double> {

	private Instances data;
	private int start;
	private int stop;

	private double Error;
	private double[] threadResults;
	private double[][] threadInstanceProbs;

	private int threadID;

	int iindex[];
	int testStartIndex;

	Classifier current;
	
	int[][] threadConfusionMatrix;
	int[] threadInfoMatrix;

	public CallableBVDCrossvalx(int start, int stop, double[] threadResults, double[][] threadInstanceProbs, Instances data, int[] iindex, int testStartIndex, Classifier current, int threadID, int[][] threadConfusionMatrix, int[] threadInfoMatrix) {
		this.data = data;
		this.start = start;
		this.stop = stop;

		this.threadResults = threadResults;
		this.threadInstanceProbs = threadInstanceProbs;

		this.iindex = iindex;
		this.testStartIndex = testStartIndex;
		this.current = current;

		this.threadID = threadID;
		this.threadConfusionMatrix = threadConfusionMatrix;
		this.threadInfoMatrix = threadInfoMatrix;
	}

	@Override
	public Double call() throws Exception {

		double log2 = Math.log(2);
		double tempError = 0;
		int numProcessed = 0;

		try {
			// Evaluate the classifier on test, updating BVD stats
			//System.out.println("Thread Starting at: " + start + " and will be stopping at: " + stop);
			for (int j = start; j <= stop; j++) {
				int actualClass = (int)data.instance(iindex[testStartIndex+j]).classValue();
				double [] probs = current.distributionForInstance(data.instance(iindex[testStartIndex+j]));

				double RMSEy = 0;  // for calculating root mean squared error
				int pred = -1;
				double bestProb = Double.MIN_VALUE;

				for (int y = 0; y < data.numClasses(); y++) {
					if (!Double.isNaN(probs[y])) {
						if (probs[y] > bestProb) {
							pred = y;
							bestProb = probs[y];
						}
						RMSEy += Math.pow((probs[y]-((y==actualClass)?1:0)), 2);

					} else {
						System.out.println(probs[y]);
						System.err.println("probs[ " + y + "] is NaN! oh no :-( Whoopsy Daisy!");					
					}
				}
				
				//updateParams(actualClass, pred);
				threadConfusionMatrix[actualClass][pred]++;
				threadInfoMatrix[actualClass]++;
				
				threadResults[2] += Math.pow((1-probs[actualClass]), 2); 							
				threadResults[1] += -(Math.log(probs[actualClass]) / log2);
				if (pred != actualClass) {
					Error++;
					tempError++;
				}

				threadInstanceProbs[iindex[testStartIndex+j]][pred]++;

				threadResults[0] += (RMSEy / data.numClasses());

				numProcessed++;
				if ((numProcessed % SUtils.displayPerfAfterInstances) == 0) {
					//System.out.print(perfOutput.charAt(threadID));
					System.out.print(SUtils.perfOutput.charAt(threadID));
				}
			} // ends j
		} catch (Exception e) {
			System.out.println("Something nasty has happend in thread: " + threadID);
			e.printStackTrace();
		}

		return Error;
	}
	
	synchronized public void updateParams(int actualClass, int pred) {
		threadConfusionMatrix[actualClass][pred]++;
		threadInfoMatrix[actualClass]++;
	}

}
