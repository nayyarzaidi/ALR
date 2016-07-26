package objectiveFunction.parallel;

import java.util.Arrays;
import java.util.concurrent.Callable;

import DataStructure.wdAnJEParameters;
import ALR.wdAnJE;
import Utils.SUtils;

import weka.core.Instance;
import weka.core.Instances;

public class CallableCLL_df implements Callable<Double> {

	private Instances instances;
	private int start;
	private int stop;

	private double[] myProbs;
	private wdAnJEParameters dParameters;

	private int nc;
	private double[] g;
	private double mLogNC;

	private int n;
	private String m_S;	

	private wdAnJE algorithm;
	
	private int threadID;

	public CallableCLL_df(Instances instances, int start, int stop, int nc, int n, String m_S, double[] myProbs, double[] g, wdAnJEParameters dParameters) {
		this.instances = instances;
		this.start = start;
		this.stop = stop;
		this.nc = nc;
		this.myProbs = myProbs;
		this.g = g;
		this.dParameters = dParameters;
		this.mLogNC = -Math.log(nc);
		this.m_S = m_S;
		this.n = n;
	}

	public CallableCLL_df(int start, int stop, double[] myProbs, double[] g, wdAnJE algorithm, int th) {
		this.algorithm = algorithm;
		this.instances = algorithm.getM_Instances();
		this.start = start;
		this.stop = stop;
		this.nc = algorithm.getNc();
		this.myProbs = myProbs;
		this.g = g;
		this.dParameters = algorithm.getdParameters_();
		this.mLogNC = -Math.log(nc);
		this.m_S = algorithm.getMS();
		this.n = algorithm.getnAttributes();
		this.threadID = th;
	}

	@Override
	public Double call() throws Exception {
		double negLogLikelihood = 0.0;
		Arrays.fill(g, 0.0);

		int numProcessed = 0;
		for (int i = start; i <= stop; i++) {
			Instance instance = instances.instance(i);
			int x_C = (int) instance.classValue();

			algorithm.logDistributionForInstance(myProbs,instance);
			negLogLikelihood += (mLogNC - myProbs[x_C]);
			SUtils.exp(myProbs);

			for (int c = 0; c < nc - 1; c++) {
//				g[c] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]);
				dParameters.incGradientAtFullIndex(g, c, (-1) * (SUtils.ind(c, x_C) - myProbs[c]));
			}

			if (m_S.equalsIgnoreCase("A1JE")) {
				// A1JE

				for (int att1 = 0; att1 < n; att1++) {
					int att1val = (int) instance.value(att1);

					for (int c = 0; c < nc - 1; c++) {
						long index = dParameters.getAttributeIndex(att1, att1val, c);
						
//						g[index] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]);
						dParameters.incGradientAtFullIndex(g, index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]));
					}
				}

			} else if (m_S.equalsIgnoreCase("A2JE")) {
				// A2JE

				for (int c = 0; c < nc - 1; c++) {
					for (int att1 = 1; att1 < n; att1++) {
						int att1val = (int) instance.value(att1);

						for (int att2 = 0; att2 < att1; att2++) {
							int att2val = (int) instance.value(att2);

							long index = dParameters.getAttributeIndex(att1, att1val, att2, att2val, c);
							dParameters.incGradientAtFullIndex(g, index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]));
						}
					}
				}

			} else if (m_S.equalsIgnoreCase("A3JE")) {
				// A3JE

				for (int c = 0; c < nc - 1; c++) {
					for (int att1 = 2; att1 < n; att1++) {
						int att1val = (int) instance.value(att1);

						for (int att2 = 1; att2 < att1; att2++) {
							int att2val = (int) instance.value(att2);

							for (int att3 = 0; att3 < att2; att3++) {
								int att3val = (int) instance.value(att3);

								long index = dParameters.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, c);
								dParameters.incGradientAtFullIndex(g, index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]));
							}
						}
					}
				}

			} else if (m_S.equalsIgnoreCase("A4JE")) {
				// A4JE

				for (int c = 0; c < nc - 1; c++) {
					for (int att1 = 3; att1 < n; att1++) {
						int att1val = (int) instance.value(att1);

						for (int att2 = 2; att2 < att1; att2++) {
							int att2val = (int) instance.value(att2);

							for (int att3 = 1; att3 < att2; att3++) {
								int att3val = (int) instance.value(att3);

								for (int att4 = 0; att4 < att3; att4++) {
									int att4val = (int) instance.value(att4);

									long index = dParameters.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, att4, att4val, c);
									dParameters.incGradientAtFullIndex(g, index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]));
								}
							}
						}
					}
				}

			} else if (m_S.equalsIgnoreCase("A5JE")) {
				// A5JE

				for (int c = 0; c < nc - 1; c++) {
					for (int att1 = 4; att1 < n; att1++) {
						int att1val = (int) instance.value(att1);

						for (int att2 = 3; att2 < att1; att2++) {
							int att2val = (int) instance.value(att2);

							for (int att3 = 2; att3 < att2; att3++) {
								int att3val = (int) instance.value(att3);

								for (int att4 = 1; att4 < att3; att4++) {
									int att4val = (int) instance.value(att4);

									for (int att5 = 0; att5 < att4; att5++) {
										int att5val = (int) instance.value(att5);

										long index = dParameters.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, att4, att4val, att5, att5val, c);
										dParameters.incGradientAtFullIndex(g, index, (-1) * (SUtils.ind(c, x_C) - myProbs[c]));
									}
								}
							}
						}
					}
				}

			} else {
				System.out.println("m_S value should be from set {A1JE, A2JE, A3JE, A4JE, A5JE}");
			}
			
			numProcessed++;
			if (algorithm.isM_MThreadVerb() && (numProcessed % SUtils.displayPerfAfterInstances) == 0) {
				//System.out.print(perfOutput.charAt(threadID));
				System.out.print(SUtils.perfOutput.charAt(threadID));
			}

		}

		return negLogLikelihood;
	}

}
