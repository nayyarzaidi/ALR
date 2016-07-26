package DataStructure;

import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import Utils.plTechniques;

import weka.core.Instance;
import weka.core.Instances;

public class wdAnJEParametersFlat extends wdAnJEParameters {

	/**
	 * Constructor called by wdAnJE
	 */
	public wdAnJEParametersFlat(int n, int nc, int N, int[] in_ParamsPerAtt, int m_P, int numTuples,String fs) {
		super(n, nc, N, in_ParamsPerAtt, m_P, numTuples,null);
		
		if (needFeatureSelection()){
			throw new RuntimeException("Feature Selection not available for the Flat version; use other Parameters datastructures (eg wcAnJEParametersIndexed)");
		}
		
		System.out.print("In Constructor of wdAnJEParametersFlat(), ");
		System.out.print("Total number of parameters are: " + getTotalNumberParameters() + ", ");
		System.out.println("Maximum TAB length is: " + MAX_TAB_LENGTH + ".");

		if (getTotalNumberParameters() > MAX_TAB_LENGTH) {
			System.err.println("CRITICAL ERROR: 'wdAnJEParametersFlat' not implemented for such dimensionalities. Use 'wdAnJEParametersIndexedBig' or 'wdAnJEParametersBitmap'");
			System.exit(-1);
		}

		if (scheme == plTechniques.MAP || scheme == plTechniques.MAP2) {
			
			initCount(getTotalNumberParameters());
			
		} else if (scheme == plTechniques.dCCBN || scheme == plTechniques.wCCBN 
				|| scheme == plTechniques.dCCBNf || scheme == plTechniques.wCCBNf 
				|| scheme == plTechniques.eCCBN 
				|| scheme == plTechniques.wCCBN2) {
			
			initCount(getTotalNumberParameters());
			initParameters(getTotalNumberParameters());
			initGradients(getTotalNumberParameters());			
		} 
	}


	@Override
	public void updateFirstPass(Instance inst) {

		int x_C = (int) inst.classValue();
		xyCount[x_C]++;

		if (numTuples == 1) {

			for (int u1 = 0; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				int index = (int) getAttributeIndex(u1, x_u1, x_C);
				incCountAtFullIndex(index);				
			}

		} else if (numTuples == 2) {

			for (int u1 = 1; u1 < n; u1++) {
				int x_u1 = (int) inst.value(u1);

				for (int u2 = 0; u2 < u1; u2++) {
					int x_u2 = (int) inst.value(u2);

					int index = (int) getAttributeIndex(u1, x_u1, u2, x_u2, x_C);
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

						int index = (int) getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, x_C);
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

							int index = (int) getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, u4, x_u4, x_C);
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

								int index = (int) getAttributeIndex(u1, x_u1, u2, x_u2, u3, x_u3, u4, x_u4, u5, x_u5, x_C);
								incCountAtFullIndex(index);
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
		int minNPerThread = 4000;					
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
		System.out.println("In wdAnJEParametersFlat() - Pass 1: Launching " + nThreads + " threads");
				
		threadXYCount = new int[nThreads][(int)getTotalNumberParameters()];
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
			Callable<Double> thread = new CallableWdAnJEParametersFlat(start, stop, m_Instances, numTuples, threadXYCount[th], this, th);

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
		System.out.println("In wdAnJEParametersFlat() - Pass 1: All threads finished.");		
	}
	
	@Override
	public void finishedFirstPass() {
		initProbs(getTotalNumberParameters());
	}
	
	@Override
	public boolean needSecondPass() {
		return false;
	}
	
	@Override
	public void updateAfterFirstPass(Instance inst) {
		// Nothing to do, needSecondPass() is false.
	}
	
	@Override
	public void updateAfterFirstPass_m(Instances m_Instances) {
		// Nothing to do, needSecondPass() is false.
	}	
	
	/* 
	 * --------------------------------------------------------------------------------
	 * Access Functions
	 * -------------------------------------------------------------------------------- 
	 */
	
	@Override
	public int getCountAtFullIndex(long index){
		return xyCount[(int)index];
	}	

	@Override
	public void setProbAtFullIndex(long index, double p) {
		probs[(int)index] = p;
	}

	@Override
	public double getProbAtFullIndex(long index) {
		return probs[(int)index];
	}

	@Override
	public double getGradientAtFullIndex(long index) {
		return gradients[(int)index];
	}

	@Override
	public void setGradientAtFullIndex(long index, double g) {
		gradients[(int)index] = g;
	}

	@Override
	public void incGradientAtFullIndex(long index, double g) {
		gradients[(int)index] += g;
	}

	@Override
	public double getGradientAtFullIndex(double[]tab, long index) {
		return tab[(int)index];
	}

	@Override
	public void setGradientAtFullIndex(double[]tab, long index, double g) {
		tab[(int)index] = g;
	}

	@Override
	public void incGradientAtFullIndex(double[]tab, long index, double g) {
		tab[(int)index] += g;
	}

	@Override
	public double getParameterAtFullIndex(long index) {
		return parameters[(int)index];
	}

	@Override
	public void setParameterAtFullIndex(long index, double p) {
		parameters[(int)index] = p;
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
		//Arrays.fill(parameters, 1.0);
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
	public void setCountAtFullIndex(long index, int count) {
		xyCount[(int)index] = count;
	}

	@Override
	public void incCountAtFullIndex(long index, int value) {
		xyCount[(int)index] += value;
	}

	@Override
	public void incCountAtFullIndex(long index) {
		xyCount[(int)index]++;
	}
	
	public void setCountAtFullIndex(int[]tab, long index, int val) {
		tab[(int)index] = val;
	}

	public void incCountAtFullIndex(int[]tab, long index) {
		tab[(int)index]++;
	}


	@Override
	public void disableFeatureAtIndex(long index) {
		throw new RuntimeException("Cannot disable a feature with the Flat version; use other Parameters datastructures (eg wdAnJEParametersIndexed)");
	}


	@Override
	public void finishedFSPass() {
		throw new RuntimeException("Feature Selection not available for the Flat version; use other Parameters datastructures (eg wdAnJEParametersIndexed)");
	}

} // ends class

