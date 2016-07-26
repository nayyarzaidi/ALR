package DataStructure;

import java.util.Arrays;
import java.util.BitSet;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import Utils.plTechniques;

import weka.core.Instance;
import weka.core.Instances;

public class wdAnJEParametersIndexed extends wdAnJEParameters {

	protected final static int SENTINEL = -1;
	protected final static double PROBA_VALUE_WHEN_ZERO_COUNT = -25; //in Log Scale
	protected final static double GRADIENT_VALUE_WHEN_ZERO_COUNT = 0.0;

	int[] indexes;
	int actualNumberParameters;

	BitSet combinationRequired;

	/**
	 * Constructor called by wdAnJE
	 */	
	public wdAnJEParametersIndexed(int n, int nc, int N, int[] in_ParamsPerAtt, int m_P, int numTuples, String fs) {

		super(n, nc, N, in_ParamsPerAtt, m_P, numTuples,fs);

		System.out.print("In Constructor of wdAnJEParametersIndexed(), ");
		System.out.print("Total number of parameters are: " + getTotalNumberParameters() + ", ");
		System.out.println("Maximum TAB length is: " + MAX_TAB_LENGTH + ".");

		if (getTotalNumberParameters() > MAX_TAB_LENGTH) {
			System.err.println("CRITICAL ERROR: 'wdAnJEParametersIndexed' not implemented for such dimensionalities. Use 'wdAnJEParametersIndexedBig'");
			System.exit(-1);
		}

		combinationRequired = new BitSet((int) getTotalNumberParameters());		
	}


	@Override
	public void updateFirstPass(Instance inst) {

		int x_C = (int) inst.classValue();
		setCombinationRequired(x_C);

		if (numTuples == 1) {

			for (int u1 = 0; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				for (int c = 0; c < nc; c++) {
					long index = getAttributeIndex(u1, x_u1, c);
					setCombinationRequired(index);
				}
			}

		} else if (numTuples == 2) {

			for (int u1 = 1; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				for (int u2 = 0; u2 < u1; u2++) {
					int x_u2 = (int) inst.value(u2);

					for (int c = 0; c < nc; c++) {
						long index = getAttributeIndex(u1, x_u1, u2, x_u2, c);
						setCombinationRequired(index);
					}
				}
			}

		} else if (numTuples == 3) {

			for (int u1 = 2; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				for (int u2 = 1; u2 < u1; u2++) {
					int x_u2 = (int) inst.value(u2);

					for (int u3 = 0; u3 < u2; u3++) {
						int x_u3 = (int) inst.value(u3);

						for (int c = 0; c < nc; c++) {
							long index = getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, c);
							setCombinationRequired(index);
						}
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

							for (int c = 0; c < nc; c++) {
								long index = getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, u4, x_u4, c);
								setCombinationRequired(index);
							}
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

								for (int c = 0; c < nc; c++) {
									long index = getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, u4, x_u4, u5, x_u5, c);
									setCombinationRequired(index);
								}
							}
						}
					}
				}
			}
		}
	}

	@Override
	public void updateFirstPass_m(Instances m_Instances) {
		int nThreads;
		int minNPerThread = 10000;					
		int N = m_Instances.numInstances();

		ExecutorService executor;

		if (N < minNPerThread) {
			nThreads = 1;
		} else {
			nThreads = Runtime.getRuntime().availableProcessors();
			if (N/nThreads < minNPerThread) {
				nThreads = N/minNPerThread + 1;
			}
		}
		System.out.println("In wdAnJEParametersIndexed() - Pass1: Launching " + nThreads + " threads");

		executor = Executors.newFixedThreadPool(nThreads);					

		Future<Double>[] futures = new Future[nThreads];

		int assigned = 0;
		int remaining = N;

		for (int th = 0; th < nThreads; th++) {
			/*
			 * Compute the start and stop indexes for thread th
			 */
			int start = assigned;
			int nInstances4Thread = remaining / (nThreads - th);
			assigned += nInstances4Thread;
			int stop = assigned - 1;
			remaining -= nInstances4Thread;

			/*
			 * Calling thread
			 */
			Callable<Double> thread = new CallableWdAnJEParametersIndexed_Pass1(start, stop, m_Instances, numTuples, this);

			futures[th] = executor.submit(thread);
		}

		for (int th = 0; th < nThreads; th++) {

			try {
				double temp = futures[th].get();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (ExecutionException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}											
		}

		executor.shutdown();
		System.out.println("In wdAnJEParametersIndexed() - Pass1: All threads finished.");
	}

	@Override
	public void finishedFirstPass() {
		indexes = new int[(int) getTotalNumberParameters()];
		actualNumberParameters = combinationRequired.cardinality();

		int index = 0;
		for (int i = 0; i < indexes.length; i++) {
			if (combinationRequired.get(i)) {
				indexes[i] = index;
				index++;
			} else {
				indexes[i] = SENTINEL;
			}
		}

		if (!needFeatureSelection())
			combinationRequired = null;

		if (index != actualNumberParameters)
			System.err.println("Ouch " + index + " - " + actualNumberParameters);

		System.out.println("	Original number of parameters were: " + getTotalNumberParameters());
		System.out.println("	Compressed number of parameters are: " + actualNumberParameters);		

		// now compress count table
		initCount(actualNumberParameters);
		initProbs(actualNumberParameters);
	}
	
	@Override
	public void finishedFSPass() {
		actualNumberParameters = combinationRequired.cardinality();

		int index = 0;
		for (int i = 0; i < indexes.length; i++) {
			if (combinationRequired.get(i)) {
				indexes[i] = index;
				index++;
			} else {
				indexes[i] = SENTINEL;
			}
		}

		combinationRequired = null;

		if (index != actualNumberParameters)
			System.err.println("Ouch " + index + " - " + actualNumberParameters);

		System.out.println("	Number of parameters before Feature Selection: " + getTotalNumberParameters());
		System.out.println("	Number of parameters after Feature Selection: " + actualNumberParameters);

		// now compress count table
		initCount(actualNumberParameters);
		initProbs(actualNumberParameters);
	}

	@Override
	public boolean needSecondPass() {
		return true;
	}

	@Override
	public void updateAfterFirstPass(Instance inst) {
		int x_C = (int) inst.classValue();
		incCountAtFullIndex(x_C);

		if (numTuples == 1) {

			for (int u1 = 0; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				long index = getAttributeIndex(u1, x_u1, x_C);
				incCountAtFullIndex(index);
			}

		} else if (numTuples == 2) {

			for (int u1 = 1; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				for (int u2 = 0; u2 < u1; u2++) {
					int x_u2 = (int) inst.value(u2);

					long index = getAttributeIndex(u1, x_u1, u2, x_u2, x_C);
					incCountAtFullIndex(index);
				}
			}

		} else if (numTuples == 3) {

			for (int u1 = 2; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				for (int u2 = 1; u2 < u1; u2++) {
					int x_u2 = (int) inst.value(u2);

					for (int u3 = 0; u3 < u2; u3++) {
						int x_u3 = (int) inst.value(u3);

						long index = getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C);
						incCountAtFullIndex(index);
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

							long index = getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, u4, x_u4, x_C);
							incCountAtFullIndex(index);
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

								long index = getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, u4, x_u4, u5, x_u5, x_C);
								incCountAtFullIndex(index);
							}
						}
					}
				}
			}
		}
	}

	@Override
	public void updateAfterFirstPass_m(Instances m_Instances) {

		int nThreads;
		int minNPerThread = 10000;					
		int N = m_Instances.numInstances();

		int[][] threadXYCount;
		ExecutorService executor;

		if (N < minNPerThread) {
			nThreads = 1;
		} else {
			nThreads = Runtime.getRuntime().availableProcessors();
			if (N/nThreads < minNPerThread) {
				nThreads = N/minNPerThread + 1;
			}
		}
		System.out.println("In wdAnJEParametersIndexed() - Pass2: Launching " + nThreads + " threads");

		threadXYCount = new int[nThreads][actualNumberParameters];
		executor = Executors.newFixedThreadPool(nThreads);					

		Future<Double>[] futures = new Future[nThreads];

		int assigned = 0;
		int remaining = N;

		for (int th = 0; th < nThreads; th++) {
			/*
			 * Compute the start and stop indexes for thread th
			 */
			int start = assigned;
			int nInstances4Thread = remaining / (nThreads - th);
			assigned += nInstances4Thread;
			int stop = assigned - 1;
			remaining -= nInstances4Thread;

			/*
			 * Calling thread
			 */
			Callable<Double> thread = new CallableWdAnJEParametersIndexed_Pass2(start, stop, m_Instances, numTuples, threadXYCount[th], this);

			futures[th] = executor.submit(thread);
		}

		for (int th = 0; th < nThreads; th++) {

			try {
				double temp = futures[th].get();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (ExecutionException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}						

			for (int i = 0; i < xyCount.length; i++) {
				xyCount[i] += threadXYCount[th][i];
			}					
		}

		executor.shutdown();
		System.out.println("In wdAnJEParametersIndexed() - Pass2: All threads finished.");	
	}

	

	/* 
	 * --------------------------------------------------------------------------------
	 * Misc Functions
	 * -------------------------------------------------------------------------------- 
	 */

	/**
	 * Set the corresponding index as true (for seen the particular combination
	 * of values)
	 * 
	 * @param index
	 */
	public void setCombinationRequired(long index) {
		combinationRequired.set((int) index);
	}

	public void clearCombinationRequired(long index) {
		combinationRequired.clear((int) index);
	}

	@Override
	public void convertToProbs() {

		super.convertToProbs();

		switch(scheme){
		case plTechniques.dCCBN:
		case plTechniques.wCCBN:
		case plTechniques.dCCBNf:
		case plTechniques.wCCBNf:
		case plTechniques.eCCBN:
		case plTechniques.wCCBN2:
			initGradients(actualNumberParameters);
			initParameters(actualNumberParameters);
			break;
		default:
		}
	}		

	/* 
	 * --------------------------------------------------------------------------------
	 * Access Functions
	 * -------------------------------------------------------------------------------- 
	 */

	@Override
	public int getCountAtFullIndex(long index) {
		int indexCompact = indexes[(int)index];
		if (indexCompact == SENTINEL) {
			return 0;
		} else {
			return xyCount[indexCompact];
		}
	}

	@Override
	public void setProbAtFullIndex(long index, double p) {
		int indexCompact = indexes[(int)index];
		if (indexCompact != SENTINEL) {
			probs[indexCompact] = p;
		}
	}

	@Override
	public double getProbAtFullIndex(long index) {
		int indexCompact = indexes[(int)index];
		if (indexCompact == SENTINEL) {
			return PROBA_VALUE_WHEN_ZERO_COUNT;
		} else {
			return probs[indexCompact];
		}
	}

	@Override
	public double getGradientAtFullIndex(long index) {
		int indexCompact = indexes[(int)index];
		if (indexCompact == SENTINEL) {
			return GRADIENT_VALUE_WHEN_ZERO_COUNT;
		} else {
			return gradients[indexCompact];
		}
	}

	@Override
	public void setGradientAtFullIndex(long index, double g) {
		int indexCompact = indexes[(int)index];
		if (indexCompact != SENTINEL) {
			gradients[indexCompact] = g;
		}
	}

	@Override
	public void incGradientAtFullIndex(long index, double g) {
		int indexCompact = indexes[(int)index];
		if (indexCompact != SENTINEL) {
			gradients[indexCompact] += g;
		}
	}

	@Override
	public double getGradientAtFullIndex(double[]tab, long index) {
		int indexCompact = indexes[(int)index];
		if (indexCompact == SENTINEL) {
			return GRADIENT_VALUE_WHEN_ZERO_COUNT;
		} else {
			return tab[indexCompact];
		}
	}

	@Override
	public void setGradientAtFullIndex(double[]tab, long index, double g) {
		int indexCompact = indexes[(int)index];
		if (indexCompact != SENTINEL) {
			tab[indexCompact] = g;
		}
	}

	@Override
	public void incGradientAtFullIndex(double[]tab, long index, double g) {
		int indexCompact = indexes[(int)index];
		if (indexCompact != SENTINEL) {
			tab[indexCompact] += g;
		}
	}

	@Override
	public double getParameterAtFullIndex(long index) {
		int indexCompact = indexes[(int)index];
		if (indexCompact == SENTINEL) {
			return PARAMETER_VALUE_WHEN_ZERO_COUNT;
		} else {
			return parameters[indexCompact];
		}
	}

	@Override
	public void setParameterAtFullIndex(long index, double p) {
		int indexCompact = indexes[(int)index];
		if (indexCompact != SENTINEL) {
			parameters[indexCompact] = p;
		}
	}

	@Override
	public void setCountAtFullIndex(long index, int count) {
		int indexCompact = indexes[(int)index];
		if (indexCompact != SENTINEL) {
			xyCount[indexCompact] = count;
		}
	}

	@Override
	public void incCountAtFullIndex(long index, int value) {
		int indexCompact = indexes[(int)index];
		if (indexCompact != SENTINEL) {
			xyCount[indexCompact] += value;
		}
	}

	@Override
	public void incCountAtFullIndex(long index) {
		int indexCompact = indexes[(int)index];
		if (indexCompact != SENTINEL) {
			xyCount[indexCompact] ++;
		}
	}

	public void incCountAtFullIndex(int[] XYCount, long index) {
		int indexCompact = indexes[(int)index];
		if (indexCompact != SENTINEL) {
			XYCount[indexCompact] ++;
		}
	}

	@Override
	protected void initCount(long size) {
		xyCount = new int[(int)size];
	}

	@Override
	protected void initProbs(long size) {
		probs = new double[(int)size];
	}

	@Override
	protected void initParameters(long size) {
		parameters = new double[(int)size];
	}

	@Override
	protected void initGradients(long size) {
		gradients = new double[(int)size];
	}

	@Override
	public void resetGradients() {
		Arrays.fill(gradients, 0.0);
	}


	@Override
	public void disableFeatureAtIndex(long index) {
		clearCombinationRequired(index);
	}

} // ends class

