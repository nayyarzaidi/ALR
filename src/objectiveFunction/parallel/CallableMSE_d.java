package objectiveFunction.parallel;

import java.util.Arrays;
import java.util.concurrent.Callable;

import DataStructure.wdAnJEParameters;
import ALR.wdAnJE;
import Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;

public class CallableMSE_d implements Callable<Double> {

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

	public CallableMSE_d(Instances instances, int start, int stop, int nc, int n, String m_S, double[] myProbs, double[] g, wdAnJEParameters dParameters) {
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

	public CallableMSE_d(int start, int stop, double[] myProbs, double[] g, wdAnJE algorithm) {
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
	}

	@Override
	public Double call() throws Exception {
		double meanSquareError = 0.0;

		Arrays.fill(g, 0.0);

		for (int i = start; i <= stop; i++) {

			Instance instance = instances.instance(i);
			int x_C = (int) instance.classValue();
			algorithm.logDistributionForInstance(myProbs,instance);
			SUtils.exp(myProbs);

			double prod = 0;
			for (int y = 0; y < nc; y++) {
				prod += (SUtils.ind(y, x_C) - myProbs[y]) * (SUtils.ind(y, x_C) - myProbs[y]);
				//meanSquareError += (prod * prod);
			}
			//meanSquareError += prod/nc;
			meanSquareError += prod;


			//algorithm.logGradientForInstance_d(g, myProbs, instance);
			// -------------------------------------------------------			
			for (int c = 0; c < nc; c++) {
				for (int k = 0; k < nc; k++) {
//					g[c] += (SUtils.ind(k, x_C) - myProbs[k]) * (-1) * (SUtils.ind(c, k) - myProbs[k]) * myProbs[c];
					dParameters.incGradientAtFullIndex(g, c, (SUtils.ind(k, x_C) - myProbs[k]) * (-1) * (SUtils.ind(c, k) - myProbs[k]) * myProbs[c]);
				}
			}

			if (m_S.equalsIgnoreCase("A1JE")) {
				// A1JE

				for (int att1 = 0; att1 < n; att1++) {
					int att1val = (int) instance.value(att1);

					for (int c = 0; c < nc; c++) {
						long index = dParameters.getAttributeIndex(att1, att1val, c);
						for (int k = 0; k < nc; k++) {
//							g[index] += (SUtils.ind(k, x_C) - myProbs[k]) * (-1) * (SUtils.ind(c, k) - myProbs[k]) * myProbs[c];
							dParameters.incGradientAtFullIndex(g, index, (SUtils.ind(k, x_C) - myProbs[k]) * (-1) * (SUtils.ind(c, k) - myProbs[k]) * myProbs[c]);
						}
					}
				}

			} else if (m_S.equalsIgnoreCase("A2JE")) {
				// A2JE

				for (int c = 0; c < nc; c++) {
					for (int att1 = 1; att1 < n; att1++) {
						int att1val = (int) instance.value(att1);

						for (int att2 = 0; att2 < att1; att2++) {
							int att2val = (int) instance.value(att2);

							long index = dParameters.getAttributeIndex(att1, att1val, att2, att2val, c);
							for (int k = 0; k < nc; k++) {
								dParameters.incGradientAtFullIndex(g, index, (SUtils.ind(k, x_C) - myProbs[k]) * (-1) * (SUtils.ind(c, k) - myProbs[k]) * myProbs[c]);
							}
						}
					}
				}

			} else if (m_S.equalsIgnoreCase("A3JE")) {
				// A3JE

				for (int c = 0; c < nc; c++) {
					for (int att1 = 2; att1 < n; att1++) {
						int att1val = (int) instance.value(att1);

						for (int att2 = 1; att2 < att1; att2++) {
							int att2val = (int) instance.value(att2);

							for (int att3 = 0; att3 < att2; att3++) {
								int att3val = (int) instance.value(att3);

								long index = dParameters.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, c);
								for (int k = 0; k < nc; k++) {
									dParameters.incGradientAtFullIndex(g, index, (SUtils.ind(k, x_C) - myProbs[k]) * (-1) * (SUtils.ind(c, k) - myProbs[k]) * myProbs[c]);
								}
							}
						}
					}
				}

			} else if (m_S.equalsIgnoreCase("A4JE")) {
				// A4JE

				for (int c = 0; c < nc; c++) {
					for (int att1 = 3; att1 < n; att1++) {
						int att1val = (int) instance.value(att1);

						for (int att2 = 2; att2 < att1; att2++) {
							int att2val = (int) instance.value(att2);

							for (int att3 = 1; att3 < att2; att3++) {
								int att3val = (int) instance.value(att3);

								for (int att4 = 0; att4 < att3; att4++) {
									int att4val = (int) instance.value(att4);

									long index = dParameters.getAttributeIndex(att1, att1val, att2, att2val, att3, att3val, att4, att4val, c);
									for (int k = 0; k < nc; k++) {
										dParameters.incGradientAtFullIndex(g, index, (SUtils.ind(k, x_C) - myProbs[k]) * (-1) * (SUtils.ind(c, k) - myProbs[k]) * myProbs[c]);
									}
								}
							}
						}
					}
				}

			} else if (m_S.equalsIgnoreCase("A5JE")) {
				// A5JE

				for (int c = 0; c < nc; c++) {
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
										for (int k = 0; k < nc; k++) {
											dParameters.incGradientAtFullIndex(g, index, (SUtils.ind(k, x_C) - myProbs[k]) * (-1) * (SUtils.ind(c, k) - myProbs[k]) * myProbs[c]);
										}
									}
								}
							}
						}
					}
				}

			} else {
				System.out.println("m_S value should be from set {A1JE, A2JE, A3JE, A4JE, A5JE}");
			}

		}

		return meanSquareError;
	} // ends Call()

}
