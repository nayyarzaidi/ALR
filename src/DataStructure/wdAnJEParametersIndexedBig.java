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

public class wdAnJEParametersIndexedBig extends wdAnJEParameters {

	protected final static int SENTINEL = -1;
	protected final static double PROBA_VALUE_WHEN_ZERO_COUNT = -25; //in Log Scale
	protected final static double GRADIENT_VALUE_WHEN_ZERO_COUNT = 0.0;
	
	int[][] indexes;
	int actualNumberParameters;
	
	BitSet[] combinationRequired;
	
	private int nLines;	
	
	/**
	 * Constructor called by wdAnJE
	 */
	public wdAnJEParametersIndexedBig(int n, int nc, int N, int[] in_ParamsPerAtt, int m_P, int numTuples, String fs) {
		
		super(n, nc, N, in_ParamsPerAtt, m_P, numTuples,fs);

		System.out.print("In Constructor of wdAnJEParametersIndexedBig(), ");
		System.out.print("Total number of parameters are: " + getTotalNumberParameters() + ", ");
		System.out.println("Maximum TAB length is: " + MAX_TAB_LENGTH + ".");

		if (getTotalNumberParameters() <= MAX_TAB_LENGTH) {
			System.out.println("WARNING: The number of parameters is not that big, it would be faster to use 'wdAnJEParametersIndexed'.");
			//System.exit(-1);
		}

		this.nLines = (int) (getTotalNumberParameters() / MAX_TAB_LENGTH) + 1;
		
		combinationRequired = new BitSet[nLines];
		for (int l = 0; l < combinationRequired.length - 1; l++) {
			combinationRequired[l] = new BitSet(MAX_TAB_LENGTH);
		}		
		combinationRequired[combinationRequired.length - 1] = new BitSet((int) (getTotalNumberParameters() % MAX_TAB_LENGTH));
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
		System.out.println("In wdAnJEParametersIndexedBig() - Pass1: Launching " + nThreads + " threads");
				
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
			Callable<Double> thread = new CallableWdAnJEParametersIndexedBig_Pass1(start, stop, m_Instances, numTuples, this);

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
		System.out.println("In wdAnJEParametersIndexedBig() - Pass1: All threads finished.");		
	}
	
	@Override
	public void finishedFirstPass() {
		
		indexes = new int[nLines][];
		for (int l = 0; l < indexes.length - 1; l++) {
			indexes[l] = new int[MAX_TAB_LENGTH];
		}
		indexes[indexes.length - 1] = new int[(int) (getTotalNumberParameters() % MAX_TAB_LENGTH)];

		actualNumberParameters = 0;

		for (int l = 0; l < indexes.length; l++) {
			for (int i = 0; i < indexes[l].length; i++) {
				if (combinationRequired[l].get(i)) {
					indexes[l][i] = actualNumberParameters;
					actualNumberParameters++;
				} else {
					indexes[l][i] = SENTINEL;
				}				
			}
			if(!needFeatureSelection()){
				combinationRequired[l] = null;
			}
		}

		System.out.println("	Original number of parameters: " + getTotalNumberParameters());
		System.out.println("	Compressed number of parameters: " + actualNumberParameters);

		// now compress count table
		initCount(actualNumberParameters);
		initProbs(actualNumberParameters);
	}
	
	@Override
	public void finishedFSPass() {

		actualNumberParameters = 0;

		for (int l = 0; l < indexes.length; l++) {
			for (int i = 0; i < indexes[l].length; i++) {
				if (combinationRequired[l].get(i)) {
					indexes[l][i] = actualNumberParameters;
					actualNumberParameters++;
				} else {
					indexes[l][i] = SENTINEL;
				}				
			}
			combinationRequired[l] = null;
		}

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

						long index = getAttributeIndex(u1, x_u1, u2, x_u2, u3,	x_u3, x_C);
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

							long index = getAttributeIndex(u1, x_u1, u2, x_u2,	u3, x_u3, u4, x_u4, x_C);
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
		System.out.println("In wdAnJEParametersIndexedBig() - Pass2: Launching " + nThreads + " threads");
				
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
			Callable<Double> thread = new CallableWdAnJEParametersIndexedBig_Pass2(start, stop, m_Instances, numTuples, threadXYCount[th], this);

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
		System.out.println("In wdAnJEParametersIndexedBig() - Pass2: All threads finished.");			
	}
	
	
	
	
	
	/* 
	 * --------------------------------------------------------------------------------
	 * Misc Functions
	 * -------------------------------------------------------------------------------- 
	 */
	
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
	
	/**
	 * Set the corresponding index as true (for seen the particular combination
	 * of values)
	 * 
	 * @param index
	 */
	public void setCombinationRequired(long index) {
		int indexL = (int) (index / MAX_TAB_LENGTH);
		int indexC = (int) (index % MAX_TAB_LENGTH);
		combinationRequired[indexL].set(indexC);
	}
	
	/**
	 * Set the corresponding index as false (for seen the particular combination
	 * of values)
	 * 
	 * @param index
	 */
	public void clearCombinationRequired(long index) {
		int indexL = (int) (index / MAX_TAB_LENGTH);
		int indexC = (int) (index % MAX_TAB_LENGTH);
		combinationRequired[indexL].clear(indexC);
	}
	
	/* 
	 * --------------------------------------------------------------------------------
	 * Access Functions
	 * -------------------------------------------------------------------------------- 
	 */

	public int getIndexCompact(long index) {
		int indexL = (int) (index / MAX_TAB_LENGTH);
		int indexC = (int) (index % MAX_TAB_LENGTH);
		return indexes[indexL][indexC];
	}

	@Override
	public int getCountAtFullIndex(long index) {
		int indexCompact = getIndexCompact(index);
		if (indexCompact == SENTINEL) {
			return 0;
		} else {
			return xyCount[indexCompact];
		}
	}

	@Override
	public void setProbAtFullIndex(long index, double p) {
		int indexCompact = getIndexCompact(index);
		if (indexCompact != SENTINEL) {
			probs[indexCompact] = p;
		}
	}

	@Override
	public double getProbAtFullIndex(long index) {
		int indexCompact = getIndexCompact(index);
		if (indexCompact == SENTINEL) {
			return PROBA_VALUE_WHEN_ZERO_COUNT;
		} else {
			return probs[indexCompact];
		}
	}

	@Override
	public double getGradientAtFullIndex(long index) {
		int indexCompact = getIndexCompact(index);
		if (indexCompact == SENTINEL) {
			return GRADIENT_VALUE_WHEN_ZERO_COUNT;
		} else {
			return gradients[indexCompact];
		}
	}

	@Override
	public void setGradientAtFullIndex(long index, double g) {
		int indexCompact = getIndexCompact(index);
		if (indexCompact != SENTINEL) {
			gradients[indexCompact] = g;
		}
	}

	@Override
	public void incGradientAtFullIndex(long index, double g) {
		int indexCompact = getIndexCompact(index);
		if (indexCompact != SENTINEL) {
			gradients[indexCompact] += g;
		}
	}

	@Override
	public double getGradientAtFullIndex(double[] tab, long index) {
		int indexCompact = getIndexCompact(index);
		if (indexCompact == SENTINEL) {
			return GRADIENT_VALUE_WHEN_ZERO_COUNT;
		} else {
			return tab[indexCompact];
		}
	}

	@Override
	public void setGradientAtFullIndex(double[] tab, long index, double g) {
		int indexCompact = getIndexCompact(index);
		if (indexCompact != SENTINEL) {
			tab[indexCompact] = g;
		}
	}

	@Override
	public void incGradientAtFullIndex(double[] tab, long index, double g) {
		int indexCompact = getIndexCompact(index);
		if (indexCompact != SENTINEL) {
			tab[indexCompact] += g;
		}
	}

	@Override
	public double getParameterAtFullIndex(long index) {
		int indexCompact = getIndexCompact(index);
		if (indexCompact == SENTINEL) {
			return PARAMETER_VALUE_WHEN_ZERO_COUNT;
		} else {
			return parameters[indexCompact];
		}
	}

	@Override
	public void setParameterAtFullIndex(long index, double p) {
		int indexCompact = getIndexCompact(index);
		if (indexCompact != SENTINEL) {
			parameters[indexCompact] = p;
		}
	}

	@Override
	public void setCountAtFullIndex(long index, int count) {
		int indexCompact = getIndexCompact(index);
		if (indexCompact != SENTINEL) {
			xyCount[indexCompact] = count;
		}
	}

	@Override
	public void incCountAtFullIndex(long index, int value) {
		int indexCompact = getIndexCompact(index);
		if (indexCompact != SENTINEL) {
			xyCount[indexCompact] += value;
		}
	}

	@Override
	public void incCountAtFullIndex(long index) {
		int indexCompact = getIndexCompact(index);
		if (indexCompact != SENTINEL) {
			xyCount[indexCompact]++;
		}
	}
	
	public void incCountAtFullIndex(int[] XYCount, long index) {
		int indexCompact = getIndexCompact(index);
		if (indexCompact != SENTINEL) {
			XYCount[indexCompact] ++;
		}
	}

	@Override
	protected void initCount(long size) {
		xyCount = new int[(int) size];
	}

	@Override
	protected void initProbs(long size) {
		probs = new double[(int) size];
	}

	@Override
	protected void initParameters(long size) {
		parameters = new double[(int) size];
	}

	@Override
	protected void initGradients(long size) {
		gradients = new double[(int) size];
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

