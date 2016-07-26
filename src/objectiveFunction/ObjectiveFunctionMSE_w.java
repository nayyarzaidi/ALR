package objectiveFunction;

//import lbfgsb.FunctionValues;
import optimize.FunctionValues;

import ALR.wdAnJE;
import DataStructure.wdAnJEParameters;
import Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;

public class ObjectiveFunctionMSE_w extends ObjectiveFunction {

	public ObjectiveFunctionMSE_w(wdAnJE algorithm) {
		super(algorithm);
	}

	@Override
	public FunctionValues getValues(double params[]) {

		double meanSquareError = 0.0;
		String m_S = algorithm.getMS();

		algorithm.dParameters_.copyParameters(params);
		algorithm.dParameters_.resetGradients();
		
		int n = algorithm.getnAttributes();
		int nc = algorithm.getNc();

		double[] myProbs = new double[nc];

		wdAnJEParameters dParameters = algorithm.getdParameters_();
		Instances instances = algorithm.getM_Instances();
		int N = instances.numInstances();
		
		for (int i = 0; i < N; i++) {
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

			//algorithm.logGradientForInstance_w(g, myProbs, instance);
			// -------------------------------------------------------
			for (int c = 0; c < nc; c++) {
				for (int k = 0; k < nc; k++) {
					dParameters.incGradientAtFullIndex(c, (SUtils.ind(k, x_C) - myProbs[k]) * (-1) * (SUtils.ind(c, k) - myProbs[k]) * myProbs[c] * dParameters.getProbAtFullIndex(c));
				}
			}

			if (m_S.equalsIgnoreCase("A1JE")) {
				// A1JE

				for (int c = 0; c < nc; c++) {
					for (int att1 = 0; att1 < n; att1++) {
						int att1val = (int) instance.value(att1);

						long index = dParameters.getAttributeIndex(att1, att1val, c);
						for (int k = 0; k < nc; k++) {
							dParameters.incGradientAtFullIndex(index, (SUtils.ind(k, x_C) - myProbs[k]) * (-1) * (SUtils.ind(c, k) - myProbs[k]) * myProbs[c] * dParameters.getProbAtFullIndex(index));
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
								dParameters.incGradientAtFullIndex(index, (SUtils.ind(k, x_C) - myProbs[k]) * (-1) * (SUtils.ind(c, k) - myProbs[k]) * myProbs[c] * dParameters.getProbAtFullIndex(index));
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
									dParameters.incGradientAtFullIndex(index, (SUtils.ind(k, x_C) - myProbs[k]) * (-1) * (SUtils.ind(c, k) - myProbs[k]) * myProbs[c] * dParameters.getProbAtFullIndex(index));
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
										dParameters.incGradientAtFullIndex(index, (SUtils.ind(k, x_C) - myProbs[k]) * (-1) * (SUtils.ind(c, k) - myProbs[k]) * myProbs[c] * dParameters.getProbAtFullIndex(index));
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
											dParameters.incGradientAtFullIndex(index, (SUtils.ind(k, x_C) - myProbs[k]) * (-1) * (SUtils.ind(c, k) - myProbs[k]) * myProbs[c] * dParameters.getProbAtFullIndex(index));
										}
									}
								}
							}
						}
					}
				}

			} else {
				System.out.println("m_S value should be from set {A1JE, A2JE, A3JE, A4JE}");
			}
			// -------------------------------------------------------
		}

		if (algorithm.isM_MVerb()) {
			System.out.print(meanSquareError + ", ");
		}

		return new FunctionValues(meanSquareError, dParameters.getGradients());
	}

}
