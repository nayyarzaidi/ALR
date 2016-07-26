package DataStructure;

import java.util.Arrays;

import org.apache.commons.math3.distribution.ChiSquaredDistribution;

import logDistributionComputation.LogDistributionComputerAnJE;
import Utils.SUtils;

import weka.core.Instance;
import weka.core.Instances;

public abstract class wdAnJEParameters {

	protected long np;

	protected int n;
	protected int nc;
	protected int N;
	protected int scheme;

	protected int[] paramsPerAtt;	

	protected indexTrie[] indexTrie_;

	protected int [] xyCount;
	protected double [] probs;
	protected double [] parameters;
	protected double [] gradients;

	protected int numTuples;

	protected static int MAX_TAB_LENGTH = Integer.MAX_VALUE-8;
	//	protected static int MAX_TAB_LENGTH = 2009;
	protected double PARAMETER_VALUE_WHEN_ZERO_COUNT = 0.0;
	public static double CRITICAL_VALUE = 0.999999;
	public static double CRITICAL_VALUE_MI = 0.1;

	public enum FeatureSelectionType {NONE,CHI_SQUARE,G_SQUARE,FISHER,MI};
	protected FeatureSelectionType fs = FeatureSelectionType.NONE;

	/**
	 * Constructor called by wdAnJE
	 */
	public wdAnJEParameters(int n, int nc, int N, int[] in_ParamsPerAtt, int m_P, int numTuples,String fs) {
		this.n = n;
		this.nc = nc;
		this.N = N;
		scheme = m_P;
		this.numTuples = numTuples;

		if (fs != null) {

			if (fs.equalsIgnoreCase("ChiSqTest")){
				this.fs = FeatureSelectionType.CHI_SQUARE;
			} else if(fs.equalsIgnoreCase("GTest")){
				this.fs = FeatureSelectionType.G_SQUARE;
			} else if(fs.equalsIgnoreCase("FisherExactTest")){
				this.fs = FeatureSelectionType.FISHER;
			} else if(fs.equalsIgnoreCase("MI")){
				this.fs = FeatureSelectionType.MI;
			} else{
				this.fs = FeatureSelectionType.NONE;
			}

		} else{
			this.fs = FeatureSelectionType.NONE;
		}

		paramsPerAtt = new int[n];
		for (int u = 0; u < n; u++) {
			paramsPerAtt[u] = in_ParamsPerAtt[u];
		}

		indexTrie_ = new indexTrie[n];				

		if (numTuples == 1) {
			np = nc;
			for (int u1 = 0; u1 < n; u1++) {
				indexTrie_[u1] = new indexTrie();

				indexTrie_[u1].set(np);
				np += (paramsPerAtt[u1] * nc);
			}
		} else if (numTuples == 2) {
			np = nc;
			for (int u1 = 1; u1 < n; u1++) {
				indexTrie_[u1] = new indexTrie();
				indexTrie_[u1].children = new indexTrie[n];

				for (int u2 = 0; u2 < u1; u2++) {

					indexTrie_[u1].children[u2] = new indexTrie();
					indexTrie_[u1].children[u2].set(np);

					np += (paramsPerAtt[u1] * paramsPerAtt[u2] * nc);												
				}					
			}
		} else if (numTuples == 3) {
			np = nc;
			for (int u1 = 2; u1 < n; u1++) {
				indexTrie_[u1] = new indexTrie();
				indexTrie_[u1].children = new indexTrie[n];

				for (int u2 = 1; u2 < u1; u2++) {

					indexTrie_[u1].children[u2] = new indexTrie();
					indexTrie_[u1].children[u2].children = new indexTrie[n];

					for (int u3 = 0; u3 < u2; u3++) {
						indexTrie_[u1].children[u2].children[u3] = new indexTrie();
						indexTrie_[u1].children[u2].children[u3].set(np);		

						np += (paramsPerAtt[u1] * paramsPerAtt[u2] * paramsPerAtt[u3] * nc);												
					}					
				}
			}
		} else if (numTuples == 4) {
			np = nc;
			for (int u1 = 3; u1 < n; u1++) {
				indexTrie_[u1] = new indexTrie();
				indexTrie_[u1].children = new indexTrie[n];

				for (int u2 = 2; u2 < u1; u2++) {

					indexTrie_[u1].children[u2] = new indexTrie();
					indexTrie_[u1].children[u2].children = new indexTrie[n];

					for (int u3 = 1; u3 < u2; u3++) {
						indexTrie_[u1].children[u2].children[u3] = new indexTrie();
						indexTrie_[u1].children[u2].children[u3].children = new indexTrie[n];

						for (int u4 = 0; u4 < u3; u4++) {
							indexTrie_[u1].children[u2].children[u3].children[u4] = new indexTrie();
							indexTrie_[u1].children[u2].children[u3].children[u4].set(np);

							np += (paramsPerAtt[u1] * paramsPerAtt[u2] * paramsPerAtt[u3] * paramsPerAtt[u4] * nc);												
						}				
					}
				}
			}
		} else if (numTuples == 5) {
			np = nc;			
			for (int u1 = 4; u1 < n; u1++) {
				indexTrie_[u1] = new indexTrie();
				indexTrie_[u1].children = new indexTrie[n];

				for (int u2 = 3; u2 < u1; u2++) {

					indexTrie_[u1].children[u2] = new indexTrie();
					indexTrie_[u1].children[u2].children = new indexTrie[n];

					for (int u3 = 2; u3 < u2; u3++) {
						indexTrie_[u1].children[u2].children[u3] = new indexTrie();
						indexTrie_[u1].children[u2].children[u3].children = new indexTrie[n];

						for (int u4 = 1; u4 < u3; u4++) {
							indexTrie_[u1].children[u2].children[u3].children[u4] = new indexTrie();
							indexTrie_[u1].children[u2].children[u3].children[u4].children = new indexTrie[n];

							for (int u5 = 0; u5 < u4; u5++) {
								indexTrie_[u1].children[u2].children[u3].children[u4].children[u5] = new indexTrie();
								indexTrie_[u1].children[u2].children[u3].children[u4].children[u5].set(np);

								np += (paramsPerAtt[u1] * paramsPerAtt[u2] * paramsPerAtt[u3] * paramsPerAtt[u4] * paramsPerAtt[u5] * nc);
							}
						}				
					}
				}
			}
		}

	}

	/**
	 * Function called in the first pass to look at the combinations that have been seen or not. 
	 * Then the function finishedUpdatingSeenObservations should be called, and then the update_MAP function. 
	 * @param inst
	 */
	public abstract void updateFirstPass(Instance inst);

	/**
	 * Multi-threaded version of updateFirstPass 
	 * @param inst
	 */
	public abstract void updateFirstPass_m(Instances m_Instances);

	/**
	 * Function called when the first pass is finished
	 */
	public abstract void finishedFirstPass();

	public abstract boolean needSecondPass();

	/**
	 * Function called to initialize the counts, if needed, in the second pass after having called 'update_seen_observations' on every instance first.
	 * Needs to be overriden, or will just do nothing 
	 * @param inst
	 */
	public void updateAfterFirstPass(Instance inst) {

	}

	/**
	 * Multi-threaded version of update_MAP 
	 * @param inst
	 */
	public void updateAfterFirstPass_m(Instances m_Instances) {

	}

	public boolean needFeatureSelection(){
		return this.fs != FeatureSelectionType.NONE;
	}

	protected boolean shouldRemoveFeature(long[][]counts){
		boolean likelyCorrelatedWithClass;
		switch (this.fs) {
		case CHI_SQUARE:
			double chi = SUtils.chisquare(counts);
			double df = 1.0* (counts.length -1)*(counts[0].length-1);
			ChiSquaredDistribution distribution = new ChiSquaredDistribution(df,1e-100);
			double pValue = 1 - distribution.cumulativeProbability(chi);
			System.out.println(pValue);
			likelyCorrelatedWithClass = (pValue < CRITICAL_VALUE);
			break;
		case FISHER:
			throw new RuntimeException("Fisher test not implemented");
		case G_SQUARE:
			double gs = SUtils.gsquare(counts);
			df = 1.0* (counts.length -1)*(counts[0].length-1);
			distribution = new ChiSquaredDistribution(df,1e-100);
			pValue = 1 - distribution.cumulativeProbability(gs);
			likelyCorrelatedWithClass = (pValue < CRITICAL_VALUE);
			break;
		case MI:
			double mi = SUtils.MI(counts);
			likelyCorrelatedWithClass = (mi >= CRITICAL_VALUE_MI);
			break;
		default:
			likelyCorrelatedWithClass = true;
			break;
		}
		return !likelyCorrelatedWithClass;

	}

	public abstract void disableFeatureAtIndex(long index);

	public void updateVectorsBasedOnFS() {
		if(this.fs == FeatureSelectionType.NONE)return;

		if (numTuples == 1) {

			for (int u1 = 0; u1 < n; u1++) {
				long[][] contingencyMatrix = new long[paramsPerAtt[u1]][nc];

				for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) {
					for (int c = 0; c < nc; c++) {
						long index = getAttributeIndex(u1, u1val, c);
						contingencyMatrix[u1val][c] = getCountAtFullIndex(index);
					}
				}

				// Pass contingency table to FS-statistics Test to determine if
				// feature or feature combination is to be retained.
				if(shouldRemoveFeature(contingencyMatrix)){
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) {

						for (int c = 0; c < nc; c++) {
							long index = getAttributeIndex(u1, u1val, c);
							disableFeatureAtIndex(index);
						}					
					}
				}

			}
		} else if (numTuples == 2) {

			for (int u1 = 1; u1 < n; u1++) {

				for (int u2 = 0; u2 < u1; u2++) {					
					long[][] contingencyMatrix = new long[paramsPerAtt[u1] * paramsPerAtt[u2]][nc];

					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) {
						for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {
							for (int c = 0; c < nc; c++) {
								long index = getAttributeIndex(u1, u1val, u2, u2val, c);
								contingencyMatrix[u1val][c] = getCountAtFullIndex(index);
							}
						}
					}

					// Pass contingency table to FS-statistics Test to determine if
					// feature or feature combination is to be retained.
					if (shouldRemoveFeature(contingencyMatrix)) {
						for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {
								for (int c = 0; c < nc; c++) {
									long index = getAttributeIndex(u1, u1val, u2, u2val, c);
									disableFeatureAtIndex(index);
								}
							}
						}
					}

				}
			}

		} else if (numTuples == 3) {

			for (int u1 = 2; u1 < n; u1++) {

				for (int u2 = 1; u2 < u1; u2++) {

					for (int u3 = 0; u3 < u2; u3++) {
						long[][] contingencyMatrix = new long[paramsPerAtt[u1] * paramsPerAtt[u2] * paramsPerAtt[u3]][nc];

						for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {
								for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {
									for (int c = 0; c < nc; c++) {
										long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, c);
										contingencyMatrix[u1val][c] = getCountAtFullIndex(index);
									}
								}
							}
						}

						// Pass contingency table to FS-statistics Test to determine if
						// feature or feature combination is to be retained.
						if (shouldRemoveFeature(contingencyMatrix)) {
							for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) {
								for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {
									for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {
										for (int c = 0; c < nc; c++) {
											long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, c);
											disableFeatureAtIndex(index);
										}
									}
								}
							}
						}

					}
				}
			}

		} else if (numTuples == 4) {

			for (int u1 = 3; u1 < n; u1++) {

				for (int u2 = 2; u2 < u1; u2++) {

					for (int u3 = 1; u3 < u2; u3++) {

						for (int u4 = 0; u4 < u3; u4++) {
							long[][] contingencyMatrix = new long[paramsPerAtt[u1] * paramsPerAtt[u2] * paramsPerAtt[u3] * paramsPerAtt[u4]][nc];

							for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) {
								for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {
									for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {
										for (int u4val = 0; u4val < paramsPerAtt[u4]; u4val++) {
											for (int c = 0; c < nc; c++) {
												long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, c);
												contingencyMatrix[u1val][c] = getCountAtFullIndex(index);
											}
										}
									}
								}
							}

							// Pass contingency table to FS-statistics Test to determine if
							// feature or feature combination is to be retained.
							if (shouldRemoveFeature(contingencyMatrix)) {
								for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) {
									for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {
										for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {
											for (int u4val = 0; u4val < paramsPerAtt[u4]; u4val++) {
												for (int c = 0; c < nc; c++) {
													long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, c);
													disableFeatureAtIndex(index);
												}
											}
										}
									}
								}
							}

						}
					}
				}
			}

		} else if (numTuples == 5) {

			for (int u1 = 4; u1 < n; u1++) {

				for (int u2 = 3; u2 < u1; u2++) {

					for (int u3 = 2; u3 < u2; u3++) {

						for (int u4 = 1; u4 < u3; u4++) {

							for (int u5 = 0; u5 < u4; u5++) {
								long[][] contingencyMatrix = new long[paramsPerAtt[u1] * paramsPerAtt[u2] * paramsPerAtt[u3] * paramsPerAtt[u4]][nc];

								for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) {
									for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {
										for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {
											for (int u4val = 0; u4val < paramsPerAtt[u4]; u4val++) {
												for (int u5val = 0; u5val < paramsPerAtt[u5]; u5val++) {
													for (int c = 0; c < nc; c++) {
														long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, u5, u5val, c);
														contingencyMatrix[u1val][c] = getCountAtFullIndex(index);
													}
												}
											}
										}
									}
								}

								// Pass contingency table to FS-statistics Test to determine if
								// feature or feature combination is to be retained.
								if (shouldRemoveFeature(contingencyMatrix) ) {
									for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) {
										for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {
											for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {
												for (int u4val = 0; u4val < paramsPerAtt[u4]; u4val++) {
													for (int u5val = 0; u5val < paramsPerAtt[u5]; u5val++) {
														for (int c = 0; c < nc; c++) {
															long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, u5, u5val, c);
															disableFeatureAtIndex(index);
														}
													}
												}
											}
										}
									}
								}

							}
						}
					}
				}
			}

		}

		finishedFSPass();		
	}

	/**
	 * Function called when the feature selection pass is finished
	 */
	public abstract void finishedFSPass();



	public abstract int getCountAtFullIndex(long index);
	public abstract void setCountAtFullIndex(long index,int count);
	public void incCountAtFullIndex(long index){
		incCountAtFullIndex(index,1);
	}
	public abstract void incCountAtFullIndex(long index,int value);

	public abstract void setProbAtFullIndex(long index,double p);
	public abstract double getProbAtFullIndex(long index);

	public abstract void setGradientAtFullIndex(long index,double g);
	public abstract double getGradientAtFullIndex(long index);
	public abstract void incGradientAtFullIndex(long index, double g);

	/**
	 * Set the value of one dimension of a given gradient
	 * @param gradient the gradient (array) to which the value has to be set
	 * @param index the index given as a coordinate in the full 1-d array
	 * @param value the value to which the specific dimension of the gradient has to be set to
	 */
	public abstract void setGradientAtFullIndex(double[]gradient,long index,double value);

	/**
	 * Get the value of one dimension of a given gradient
	 * @param gradient the gradient (array) from which the value has to be gotten
	 * @param index the index given as a coordinate in the full 1-d array
	 */
	public abstract double getGradientAtFullIndex(double[]gradient,long index);

	/**
	 * Increase the value of one dimension of a given gradient
	 * @param gradient the gradient (array) to which the value has to be set
	 * @param index the index given as a coordinate in the full 1-d array
	 * @param value the value by which the specific dimension of the gradient has to be increased 
	 */
	public abstract void incGradientAtFullIndex(double []gradient,long index, double value);


	public void convertToProbs() {

		for (int c = 0; c < nc; c++) {
			setProbAtFullIndex(c,  Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(c), N, nc), 1e-75)));
		}


		if (numTuples == 1) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 0; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						long index = getAttributeIndex(u1, u1val, c);
						setProbAtFullIndex(index, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(index), getCountAtFullIndex(c), paramsPerAtt[u1]), 1e-75)));
					}
				}				
			}

		} else if (numTuples == 2) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 1; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 0; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {					

								long index = getAttributeIndex(u1, u1val, u2, u2val, c);
								setProbAtFullIndex(index, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(index), getCountAtFullIndex(c), paramsPerAtt[u1] * paramsPerAtt[u2]), 1e-75)));
							}
						}
					}
				}
			}	

		} else if (numTuples == 3) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 2; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 1; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

								for (int u3 = 0; u3 < u2; u3++) {
									for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {	

										long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, c);
										setProbAtFullIndex(index, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(index), getCountAtFullIndex(c), paramsPerAtt[u1] * paramsPerAtt[u2] * paramsPerAtt[u3]), 1e-75)));
									}
								}
							}
						}
					}
				}
			}

		} else if (numTuples == 4) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 3; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 2; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

								for (int u3 = 1; u3 < u2; u3++) {
									for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {

										for (int u4 = 0; u4 < u3; u4++) {
											for (int u4val = 0; u4val < paramsPerAtt[u4]; u4val++) {

												long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, c);
												setProbAtFullIndex(index, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(index), getCountAtFullIndex(c), paramsPerAtt[u1] * paramsPerAtt[u2] * paramsPerAtt[u3] * paramsPerAtt[u4]), 1e-75)));
											}
										}
									}
								}
							}
						}
					}
				}
			}
		} else if (numTuples == 5) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 4; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 3; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

								for (int u3 = 2; u3 < u2; u3++) {
									for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {

										for (int u4 = 1; u4 < u3; u4++) {
											for (int u4val = 0; u4val < paramsPerAtt[u4]; u4val++) {

												for (int u5 = 0; u5 < u4; u5++) {
													for (int u5val = 0; u5val < paramsPerAtt[u5]; u5val++) {

														long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, u5, u5val, c);
														setProbAtFullIndex(index, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(index), getCountAtFullIndex(c), paramsPerAtt[u1] * paramsPerAtt[u2] * paramsPerAtt[u3] * paramsPerAtt[u4] * paramsPerAtt[u5]), 1e-75)));
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}

		xyCount = null;
		System.gc();
	}

	public void convertToProbs_Cond() {

		for (int c = 0; c < nc; c++) {
			setProbAtFullIndex(c, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(c), N, nc), 1e-75)));
		}


		if (numTuples == 1) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 0; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						long index = getAttributeIndex(u1, u1val, c);
						//setProbAtFullIndex(index, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(index), getCountAtFullIndex(c), paramsPerAtt[u1]), 1e-75)));

						int sumVal = 0;
						for (int y = 0; y < nc; y++) {
							long tempIndex = getAttributeIndex(u1, u1val, y);
							sumVal += getCountAtFullIndex(tempIndex);
						}

						// P(y|x)
						setProbAtFullIndex(index, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(index), sumVal, nc), 1e-75)));
						//setProbAtFullIndex(index, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(index), sumVal), 1e-75)));

						// P(y,x)
						//setProbAtFullIndex(index, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(index), N, paramsPerAtt[u1] * nc), 1e-75)));
						//setProbAtFullIndex(index, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(index), N), 1e-75)));
					}
				}				
			}

		} else if (numTuples == 2) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 1; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 0; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {					

								long index = getAttributeIndex(u1, u1val, u2, u2val, c);
								//setProbAtFullIndex(index, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(index), getCountAtFullIndex(c), paramsPerAtt[u1] * paramsPerAtt[u2]), 1e-75)));

								int sumVal = 0;
								for (int y = 0; y < nc; y++) {
									long tempIndex = getAttributeIndex(u1, u1val, u2, u2val, y);
									sumVal += getCountAtFullIndex(tempIndex);
								}

								setProbAtFullIndex(index, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(index), sumVal, nc), 1e-75)));		
							}
						}
					}
				}
			}	

		} else if (numTuples == 3) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 2; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 1; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

								for (int u3 = 0; u3 < u2; u3++) {
									for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {	

										long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, c);
										//setProbAtFullIndex(index, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(index), getCountAtFullIndex(c), paramsPerAtt[u1] * paramsPerAtt[u2] * paramsPerAtt[u3]), 1e-75)));

										int sumVal = 0;
										for (int y = 0; y < nc; y++) {
											long tempIndex = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, y);
											sumVal += getCountAtFullIndex(tempIndex);
										}

										setProbAtFullIndex(index, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(index), sumVal, nc), 1e-75)));		
									}
								}
							}
						}
					}
				}
			}

		} else if (numTuples == 4) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 3; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 2; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

								for (int u3 = 1; u3 < u2; u3++) {
									for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {

										for (int u4 = 0; u4 < u3; u4++) {
											for (int u4val = 0; u4val < paramsPerAtt[u4]; u4val++) {

												long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, c);
												//setProbAtFullIndex(index, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(index), getCountAtFullIndex(c), paramsPerAtt[u1] * paramsPerAtt[u2] * paramsPerAtt[u3] * paramsPerAtt[u4]), 1e-75)));

												int sumVal = 0;
												for (int y = 0; y < nc; y++) {
													long tempIndex = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, y);
													sumVal += getCountAtFullIndex(tempIndex);
												}

												setProbAtFullIndex(index, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(index), sumVal, nc), 1e-75)));		
											}
										}
									}
								}
							}
						}
					}
				}
			}
		} else if (numTuples == 5) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 4; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 3; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

								for (int u3 = 2; u3 < u2; u3++) {
									for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {

										for (int u4 = 1; u4 < u3; u4++) {
											for (int u4val = 0; u4val < paramsPerAtt[u4]; u4val++) {

												for (int u5 = 0; u5 < u4; u5++) {
													for (int u5val = 0; u5val < paramsPerAtt[u5]; u5val++) {

														long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, u5, u5val, c);
														//setProbAtFullIndex(index, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(index), getCountAtFullIndex(c), paramsPerAtt[u1] * paramsPerAtt[u2] * paramsPerAtt[u3] * paramsPerAtt[u4] * paramsPerAtt[u5]), 1e-75)));

														int sumVal = 0;
														for (int y = 0; y < nc; y++) {
															long tempIndex = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, u5, u5val, y);
															sumVal += getCountAtFullIndex(tempIndex);
														}

														setProbAtFullIndex(index, Math.log(Math.max(SUtils.MEsti(getCountAtFullIndex(index), sumVal, nc), 1e-75)));		
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}

		xyCount = null;
		System.gc();
	}

	public void convertToProbs_F() {
		convertToProbs();

		for (int c = 0; c < nc; c++) {
			setProbAtFullIndex(c,  getProbAtFullIndex(c) - getProbAtFullIndex(nc - 1));
		}


		if (numTuples == 1) {

			for (int c = 0; c < nc; c++) {				

				for (int u1 = 0; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						long index = getAttributeIndex(u1, u1val, c);
						long indexMax = getAttributeIndex(u1, u1val, nc - 1);

						setProbAtFullIndex(index, getProbAtFullIndex(index) - getProbAtFullIndex(indexMax));
					}
				}				
			}

		} else if (numTuples == 2) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 1; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 0; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {					

								long index = getAttributeIndex(u1, u1val, u2, u2val, c);
								long indexMax = getAttributeIndex(u1, u1val, u2, u2val, nc - 1);

								setProbAtFullIndex(index, getProbAtFullIndex(index) - getProbAtFullIndex(indexMax));
							}
						}
					}
				}
			}	

		} else if (numTuples == 3) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 2; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 1; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

								for (int u3 = 0; u3 < u2; u3++) {
									for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {	

										long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, c);
										long indexMax = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, nc - 1);

										setProbAtFullIndex(index, getProbAtFullIndex(index) - getProbAtFullIndex(indexMax));
									}
								}
							}
						}
					}
				}
			}

		} else if (numTuples == 4) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 3; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 2; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

								for (int u3 = 1; u3 < u2; u3++) {
									for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {

										for (int u4 = 0; u4 < u3; u4++) {
											for (int u4val = 0; u4val < paramsPerAtt[u4]; u4val++) {

												long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, c);
												long indexMax = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, nc - 1);

												setProbAtFullIndex(index, getProbAtFullIndex(index) - getProbAtFullIndex(indexMax));
											}
										}
									}
								}
							}
						}
					}
				}
			}
		} else if (numTuples == 5) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 4; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 3; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

								for (int u3 = 2; u3 < u2; u3++) {
									for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {

										for (int u4 = 1; u4 < u3; u4++) {
											for (int u4val = 0; u4val < paramsPerAtt[u4]; u4val++) {

												for (int u5 = 0; u5 < u4; u5++) {
													for (int u5val = 0; u5val < paramsPerAtt[u5]; u5val++) {

														long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, u5, u5val, c);
														long indexMax = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, u5, u5val, c);

														setProbAtFullIndex(index, getProbAtFullIndex(index) - getProbAtFullIndex(indexMax));
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}

	}

	private void setParametersOfOneClassToZero() {

		int mClass = nc - 1;
		parameters[mClass] = 0;

		if (numTuples == 1) {

			for (int u1 = 0; u1 < n; u1++) {
				for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

					long index = getAttributeIndex(u1, u1val, mClass);
					parameters[(int) index] = 0;						
				}
			}				


		} else if (numTuples == 2) {


			for (int u1 = 1; u1 < n; u1++) {
				for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

					for (int u2 = 0; u2 < u1; u2++) {
						for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {					

							long index = getAttributeIndex(u1, u1val, u2, u2val, mClass);
							parameters[(int) index] = 0;
						}
					}
				}
			}


		} else if (numTuples == 3) {


			for (int u1 = 2; u1 < n; u1++) {
				for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

					for (int u2 = 1; u2 < u1; u2++) {
						for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

							for (int u3 = 0; u3 < u2; u3++) {
								for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {	

									long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, mClass);
									parameters[(int) index] = 0;
								}
							}
						}
					}
				}
			}


		} else if (numTuples == 4) {


			for (int u1 = 3; u1 < n; u1++) {
				for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

					for (int u2 = 2; u2 < u1; u2++) {
						for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

							for (int u3 = 1; u3 < u2; u3++) {
								for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {

									for (int u4 = 0; u4 < u3; u4++) {
										for (int u4val = 0; u4val < paramsPerAtt[u4]; u4val++) {

											long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, mClass);
											parameters[(int) index] = 0;	
										}
									}
								}
							}
						}
					}
				}

			}
		} else if (numTuples == 5) {


			for (int u1 = 4; u1 < n; u1++) {
				for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

					for (int u2 = 3; u2 < u1; u2++) {
						for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

							for (int u3 = 2; u3 < u2; u3++) {
								for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {

									for (int u4 = 1; u4 < u3; u4++) {
										for (int u4val = 0; u4val < paramsPerAtt[u4]; u4val++) {

											for (int u5 = 0; u5 < u4; u5++) {
												for (int u5val = 0; u5val < paramsPerAtt[u5]; u5val++) {

													long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, u5, u5val, mClass);
													parameters[(int) index] = 0;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}

	}
	
	public void multiplyProbsWithAnJEWeightOpt() {

		double w = 0.0;

		if (numTuples == 1) {
			w = 1;
		} else if (numTuples == 2) {
			//w = n/2.0 * 1.0/SUtils.NC2(n); // (also equal to 1/(n-1))	
			w = 1 / ((double)n-1);
		} else if (numTuples == 3) {
			w = n/3.0 * 1.0/SUtils.NC3(n);		
		} else if (numTuples == 4) {
			w = n/4.0 * 1.0/SUtils.NC4(n);			
		} else if (numTuples == 5) {
			w = n/5.0 * 1.0/SUtils.NC5(n);;		
		}

		for (int c = 0; c < nc; c++) {
			double prob = getProbAtFullIndex(c);
			setProbAtFullIndex(c,  prob * 1/w);
		}

	}

	public void multiplyProbsWithAnJEWeight() {

		double w = 0.0;

		if (numTuples == 1) {
			w = 1;
		} else if (numTuples == 2) {
			//w = n/2.0 * 1.0/SUtils.NC2(n); // (also equal to 1/(n-1))	
			w = 1 / ((double)n-1);
		} else if (numTuples == 3) {
			w = n/3.0 * 1.0/SUtils.NC3(n);		
		} else if (numTuples == 4) {
			w = n/4.0 * 1.0/SUtils.NC4(n);			
		} else if (numTuples == 5) {
			w = n/5.0 * 1.0/SUtils.NC5(n);;		
		}

		for (int c = 0; c < nc; c++) {
			double prob = getProbAtFullIndex(c);
			setProbAtFullIndex(c,  prob);
		}

		if (numTuples == 1) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 0; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						long index = getAttributeIndex(u1, u1val, c);
						double prob = getProbAtFullIndex(index);
						setProbAtFullIndex(index, prob * w);
					}
				}				
			}

		} else if (numTuples == 2) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 1; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 0; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {					

								long index = getAttributeIndex(u1, u1val, u2, u2val, c);
								double prob = getProbAtFullIndex(index);
								setProbAtFullIndex(index, prob * w);
							}
						}
					}
				}
			}	

		} else if (numTuples == 3) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 2; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 1; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

								for (int u3 = 0; u3 < u2; u3++) {
									for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {	

										long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, c);
										double prob = getProbAtFullIndex(index);
										setProbAtFullIndex(index, prob * w);
									}
								}
							}
						}
					}
				}
			}

		} else if (numTuples == 4) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 3; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 2; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

								for (int u3 = 1; u3 < u2; u3++) {
									for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {

										for (int u4 = 0; u4 < u3; u4++) {
											for (int u4val = 0; u4val < paramsPerAtt[u4]; u4val++) {

												long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, c);
												double prob = getProbAtFullIndex(index);
												setProbAtFullIndex(index, prob * w);
											}
										}
									}
								}
							}
						}
					}
				}
			}
			
		} else if (numTuples == 5) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 4; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 3; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

								for (int u3 = 2; u3 < u2; u3++) {
									for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {

										for (int u4 = 1; u4 < u3; u4++) {
											for (int u4val = 0; u4val < paramsPerAtt[u4]; u4val++) {

												for (int u5 = 0; u5 < u4; u5++) {
													for (int u5val = 0; u5val < paramsPerAtt[u5]; u5val++) {

														long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, u5, u5val, c);
														double prob = getProbAtFullIndex(index);
														setProbAtFullIndex(index, prob * w);
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}

	}

	/* 
	 * -----------------------------------------------------------------------------------------
	 * Common to dCCBN, wCCBN, eCCBN
	 * -----------------------------------------------------------------------------------------
	 */	


	// ----------------------------------------------------------------------------------
	// Access Functions
	// ----------------------------------------------------------------------------------

	public abstract double getParameterAtFullIndex(long index);
	public abstract void setParameterAtFullIndex(long index,double p);

	public double[]getParameters(){
		return parameters;
	}

	public double[]getGradients(){
		return gradients;
	}

	public int getClassIndex(int k) {
		return k;
	}

	public long getAttributeIndex(int att1, int att1val, int c) {
		long offset = indexTrie_[att1].offset;		
		return offset + c * (paramsPerAtt[att1]) + att1val;
	}

	public long getAttributeIndex(int att1, int att1val, int att2, int att2val, int c) {
		long offset = indexTrie_[att1].children[att2].offset;		
		return offset + c * (paramsPerAtt[att1] * paramsPerAtt[att2]) + 
				att2val * (paramsPerAtt[att1]) + 
				att1val;
	}

	public long getAttributeIndex(int att1, int att1val, int att2, int att2val, int att3, int att3val, int c) {
		long offset = indexTrie_[att1].children[att2].children[att3].offset;
		return offset + c * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3]) + 
				att3val * (paramsPerAtt[att1] * paramsPerAtt[att2]) + 
				att2val * (paramsPerAtt[att1]) + 
				att1val;
	}

	public long getAttributeIndex(int att1, int att1val, int att2, int att2val, int att3, int att3val, int att4, int att4val, int c) {
		long offset = indexTrie_[att1].children[att2].children[att3].children[att4].offset;
		return offset + c * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3] * paramsPerAtt[att4]) +
				att4val * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3]) +
				att3val * (paramsPerAtt[att1] * paramsPerAtt[att2]) + 
				att2val * (paramsPerAtt[att1]) + 
				att1val;
	}

	public long getAttributeIndex(int att1, int att1val, int att2, int att2val, int att3, int att3val, int att4, int att4val, int att5, int att5val, int c) {
		long offset = indexTrie_[att1].children[att2].children[att3].children[att4].children[att5].offset;		
		return offset + c * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3] * paramsPerAtt[att4] * paramsPerAtt[att5]) +
				att5val * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3] * paramsPerAtt[att4]) +
				att4val * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3]) +
				att3val * (paramsPerAtt[att1] * paramsPerAtt[att2]) + 
				att2val * (paramsPerAtt[att1]) + 
				att1val;
	}



	//	public double getParameters(int att1, int att1val, int c) {
	//		int offset = indexTrie_[att1].offset;		
	//		return parameters[offset + c * (paramsPerAtt[att1]) + att1val];
	//	}
	//
	//	public double getParameters(int att1, int att1val, int att2, int att2val, int c) {
	//		int offset = indexTrie_[att1].children[att2].offset;
	//		return parameters[offset + c * (paramsPerAtt[att1] * paramsPerAtt[att2]) + att2val * (paramsPerAtt[att1]) + att1val];
	//	}	
	//
	//	public double getParameters(int att1, int att1val, int att2, int att2val, int att3, int att3val, int c) {
	//		int offset = indexTrie_[att1].children[att2].children[att3].offset;
	//		return parameters[offset + c * (paramsPerAtt[att1] * paramsPerAtt[att2] * paramsPerAtt[att3]) + att3val * (paramsPerAtt[att1] * paramsPerAtt[att2]) + att2val * (paramsPerAtt[att1]) + att1val];
	//	}


	public double getParamsPetAtt(int att) {
		return paramsPerAtt[att];
	}

	public int getNumberParametersAllocated(){
		return parameters.length;
	}

	public long getTotalNumberParameters() {
		return np;
	}

	public int getTotalNumberParametersLevel1() {
		int numParams = nc;

		if (numTuples == 1) {
			numParams += n;
		} else if (numTuples == 2) {
			numParams += SUtils.NC2(n);
		} else if (numTuples == 3) {
			numParams += SUtils.NC3(n);
		} else if (numTuples == 4) {
			numParams += SUtils.NC4(n);
		} else if (numTuples == 5) {
			numParams += SUtils.NC5(n);
		}

		return numParams;
	}

	public int getTotalNumberParametersLevel2() {
		int numParams = nc;

		if (numTuples == 1) {
			numParams += (n * nc);
		} else if (numTuples == 2) {
			numParams += (SUtils.NC2(n) * nc);
		} else if (numTuples == 3) {
			numParams += (SUtils.NC3(n) * nc);
		} else if (numTuples == 4) {
			numParams += (SUtils.NC4(n) * nc);
		} else if (numTuples == 5) {
			numParams += (SUtils.NC5(n) * nc);
		}

		return numParams;
	}



	// ----------------------------------------------------------------------------------
	// Flat structure into array
	// ----------------------------------------------------------------------------------

	//public void copyParameters(double[] inParameters) {
	//	System.arraycopy(inParameters, 0, parameters, 0, np);
	//}

	public void copyParameters(double[] params) {
		System.arraycopy(params, 0, parameters, 0, params.length);
	}

	public void initializeParameters_W(int val, boolean isFeelders) {

		if (val == -1) {

			if (numTuples == 1) {
				double w = 1;
				for (int i = 0; i < parameters.length; i++) {
					parameters[i] = w;
				}		
			} else if (numTuples == 2) {
				double w = n/2.0 * 1.0/SUtils.NC2(n); // (also equal to 1/(n-1))
				for (int i = 0; i < parameters.length; i++) {
					parameters[i] = w;
				}		
			} else if (numTuples == 3) {
				double w = n/3.0 * 1.0/SUtils.NC3(n);
				for (int i = 0; i < parameters.length; i++) {
					parameters[i] = w;
				}		
			} else if (numTuples == 4) {
				double w = n/4.0 * 1.0/SUtils.NC4(n);
				for (int i = 0; i < parameters.length; i++) {
					parameters[i] = w;
				}		
			} else if (numTuples == 5) {
				double w = n/5.0 * 1.0/SUtils.NC5(n);;
				for (int i = 0; i < parameters.length; i++) {
					parameters[i] = w;
				}		
			}

			if (isFeelders) {
				setParametersOfOneClassToZero();
			}


		} else {

			Arrays.fill(parameters, val);

			if (isFeelders) {
				setParametersOfOneClassToZero();
			}		

		}

		//		if (scheme == 3) { // doing only for wCCBN
		//			if (numTuples == 1) {
		//				PARAMETER_VALUE_WHEN_ZERO_COUNT = 0;
		//			} else if (numTuples == 2) {
		//				PARAMETER_VALUE_WHEN_ZERO_COUNT = n / 2.0 * 1.0 / SUtils.NC2(n); 
		//			} else if (numTuples == 3) {
		//				PARAMETER_VALUE_WHEN_ZERO_COUNT = n / 3.0 * 1.0 / SUtils.NC3(n);
		//			} else if (numTuples == 4) {
		//				PARAMETER_VALUE_WHEN_ZERO_COUNT = n / 4.0 * 1.0 / SUtils.NC4(n);
		//			} else if (numTuples == 5) {
		//				PARAMETER_VALUE_WHEN_ZERO_COUNT = n / 5.0 * 1.0 / SUtils.NC5(n);
		//			}	
		//		}

	}

	public void initializeParameters_D(int val, boolean isFeelders) {

		if (val == -1) {

			if (numTuples == 1) {
				double w = 1;
				for (int i = 0; i < parameters.length; i++) {
					parameters[i] = w * probs[i];
				}		
			} else if (numTuples == 2) {
				double w = n/2.0 * 1.0/SUtils.NC2(n); // (also equal to 1/(n-1))
				for (int i = 0; i < parameters.length; i++) {
					parameters[i] = w * probs[i];
				}		
			} else if (numTuples == 3) {
				double w = n/3.0 * 1.0/SUtils.NC3(n);
				for (int i = 0; i < parameters.length; i++) {
					parameters[i] = w * probs[i];
				}		
			} else if (numTuples == 4) {
				double w = n/4.0 * 1.0/SUtils.NC4(n);
				for (int i = 0; i < parameters.length; i++) {
					parameters[i] = w * probs[i];
				}		
			} else if (numTuples == 5) {
				double w = n/5.0 * 1.0/SUtils.NC5(n);;
				for (int i = 0; i < parameters.length; i++) {
					parameters[i] = w * probs[i];
				}		
			}

			if (isFeelders) {
				setParametersOfOneClassToZero();
			}


		} else {

			Arrays.fill(parameters, val);

			if (isFeelders) {
				setParametersOfOneClassToZero();
			}		

		}

	}

	public void printStatistics() {

		int[] countVector = new int[7];

		if (numTuples == 1) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 0; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						long index = getAttributeIndex(u1, u1val, c);
						int count = getCountAtFullIndex(index);
						if (count == 0) {
							countVector[0]++;
						} else if (count == 1) {
							countVector[1]++;
						} else if (count > 1 && count <= 5) {
							countVector[2]++;
						} else if (count > 5 && count <= 10) {
							countVector[3]++;
						} else if (count > 10 && count <= 15) {
							countVector[4]++;
						} else if (count > 15 && count <= 20) {
							countVector[5]++;
						} else if (count > 20) {
							countVector[6]++;
						}
					}
				}				
			}

		} else if (numTuples == 2) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 1; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 0; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {					

								long index = getAttributeIndex(u1, u1val, u2, u2val, c);
								int count = getCountAtFullIndex(index);
								if (count == 0) {
									countVector[0]++;
								} else if (count == 1) {
									countVector[1]++;
								} else if (count > 1 && count <= 5) {
									countVector[2]++;
								} else if (count > 5 && count <= 10) {
									countVector[3]++;
								} else if (count > 10 && count <= 15) {
									countVector[4]++;
								} else if (count > 15 && count <= 20) {
									countVector[5]++;
								} else if (count > 20) {
									countVector[6]++;
								}
							}
						}
					}
				}
			}	

		} else if (numTuples == 3) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 2; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 1; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

								for (int u3 = 0; u3 < u2; u3++) {
									for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {	

										long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, c);
										int count = getCountAtFullIndex(index);
										if (count == 0) {
											countVector[0]++;
										} else if (count == 1) {
											countVector[1]++;
										} else if (count > 1 && count <= 5) {
											countVector[2]++;
										} else if (count > 5 && count <= 10) {
											countVector[3]++;
										} else if (count > 10 && count <= 15) {
											countVector[4]++;
										} else if (count > 15 && count <= 20) {
											countVector[5]++;
										} else if (count > 20) {
											countVector[6]++;
										}
									}
								}
							}
						}
					}
				}
			}

		} else if (numTuples == 4) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 3; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 2; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

								for (int u3 = 1; u3 < u2; u3++) {
									for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {

										for (int u4 = 0; u4 < u3; u4++) {
											for (int u4val = 0; u4val < paramsPerAtt[u4]; u4val++) {

												long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, c);
												int count = getCountAtFullIndex(index);
												if (count == 0) {
													countVector[0]++;
												} else if (count == 1) {
													countVector[1]++;
												} else if (count > 1 && count <= 5) {
													countVector[2]++;
												} else if (count > 5 && count <= 10) {
													countVector[3]++;
												} else if (count > 10 && count <= 15) {
													countVector[4]++;
												} else if (count > 15 && count <= 20) {
													countVector[5]++;
												} else if (count > 20) {
													countVector[6]++;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		} else if (numTuples == 5) {

			for (int c = 0; c < nc; c++) {

				for (int u1 = 4; u1 < n; u1++) {
					for (int u1val = 0; u1val < paramsPerAtt[u1]; u1val++) { 

						for (int u2 = 3; u2 < u1; u2++) {
							for (int u2val = 0; u2val < paramsPerAtt[u2]; u2val++) {

								for (int u3 = 2; u3 < u2; u3++) {
									for (int u3val = 0; u3val < paramsPerAtt[u3]; u3val++) {

										for (int u4 = 1; u4 < u3; u4++) {
											for (int u4val = 0; u4val < paramsPerAtt[u4]; u4val++) {

												for (int u5 = 0; u5 < u4; u5++) {
													for (int u5val = 0; u5val < paramsPerAtt[u5]; u5val++) {

														long index = getAttributeIndex(u1, u1val, u2, u2val, u3, u3val, u4, u4val, u5, u5val, c);
														int count = getCountAtFullIndex(index);
														if (count == 0) {
															countVector[0]++;
														} else if (count == 1) {
															countVector[1]++;
														} else if (count > 1 && count <= 5) {
															countVector[2]++;
														} else if (count > 5 && count <= 10) {
															countVector[3]++;
														} else if (count > 10 && count <= 15) {
															countVector[4]++;
														} else if (count > 15 && count <= 20) {
															countVector[5]++;
														} else if (count > 20) {
															countVector[6]++;
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}

		double totalCount = 0;
		for (int i = 0; i < 7; i++) {
			totalCount += countVector[i];
		}

		System.out.println("Priniting Statistics");
		System.out.println(" = 0           :" + countVector[0] + " : " + countVector[0]/totalCount);
		System.out.println(" = 1           :" + countVector[1] + " : " + countVector[1]/totalCount);
		System.out.println(" > 1 && <= 5   :" + countVector[2] + " : " + countVector[2]/totalCount);
		System.out.println(" > 6 && <= 10  :" + countVector[3] + " : " + countVector[3]/totalCount);
		System.out.println(" > 11 && <= 15 :" + countVector[4] + " : " + countVector[4]/totalCount);
		System.out.println(" > 15 && <= 20 :" + countVector[5] + " : " + countVector[5]/totalCount);
		System.out.println(" > 20          :" + countVector[6] + " : " + countVector[6]/totalCount);		
	}

	protected abstract void initCount(long size);

	protected abstract void initProbs(long size);

	protected abstract void initParameters(long size);

	protected abstract void initGradients(long size);

	public abstract void resetGradients();

	public int getNAttributes(){
		return n;
	}

	public String getNLL(Instances instances, LogDistributionComputerAnJE logDComputer) {

		double nll = 0;

		int N = instances.numInstances();
		double mLogNC = -Math.log(nc); 
		double[] myProbs = new double[nc];

		for (int i = 0; i < N; i++) {
			Instance instance = instances.instance(i);
			//logDComputer.compute(myProbs, this, instance);

			int x_C = (int) instance.classValue();

			SUtils.normalizeInLogDomain(myProbs);
			nll += (mLogNC - myProbs[x_C]);
		}

		return nll + "";
	}

} // ends class
