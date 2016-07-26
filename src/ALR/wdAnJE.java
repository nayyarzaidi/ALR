/*
 * MMaLL: An open source system for learning from very large data
 * Copyright (C) 2014 Nayyar A Zaidi, Francois Petitjean and Geoffrey I Webb
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * Please report any bugs to Nayyar Zaidi <nayyar.zaidi@monash.edu>
 */

/*
 * wdAnJE Classifier
 * 
 * wdAnJE.java     
 * Code written by: Nayyar Zaidi, Francois Petitjean
 * 
 * Options:
 * -------
 * 
 * -D   Discretize numeric attributes
 * -V 	Verbosity
 * -M   Multi-threaded
 * -W   Initialize weights to AnJE weights
 * 
 * -S	Structure learning (A1JE, A2JE, A3JE, A4JE, A5JE)
 * -P	Parameter learning (MAP, dCCBN, wCCBN, eCCBN, MAP2, wCCBN2)
 * -I   Structure to use (Flat, Indexed, IndexedBig, BitMap) 
 * -E   Objective function to optimize (CLL, MSE)
 * 
 */
package ALR;

import DataStructure.wdAnJEParameters;
import DataStructure.wdAnJEParametersBitmap;
import DataStructure.wdAnJEParametersFlat;
import DataStructure.wdAnJEParametersIndexed;
import DataStructure.wdAnJEParametersIndexedBig;

import Utils.SUtils;
import Utils.plTechniques;

import logDistributionComputation.LogDistributionComputerAnJE;

import objectiveFunction.ObjectiveFunction;
import objectiveFunction.ObjectiveFunctionCLL_d;
import objectiveFunction.ObjectiveFunctionCLL_df;
import objectiveFunction.ObjectiveFunctionCLL_w;
import objectiveFunction.ObjectiveFunctionCLL_w2;
import objectiveFunction.ObjectiveFunctionCLL_wf;
import objectiveFunction.ObjectiveFunctionMSE_d;
import objectiveFunction.ObjectiveFunctionMSE_w;

import objectiveFunction.parallel.ParallelObjectiveFunctionCLL_d;
import objectiveFunction.parallel.ParallelObjectiveFunctionCLL_df;
import objectiveFunction.parallel.ParallelObjectiveFunctionCLL_w;
import objectiveFunction.parallel.ParallelObjectiveFunctionCLL_wf;
import objectiveFunction.parallel.ParallelObjectiveFunctionMSE_d;
import objectiveFunction.parallel.ParallelObjectiveFunctionMSE_w;

import optimize.Minimizer;
import optimize.MinimizerCG;
import optimize.MinimizerGD;
import optimize.Result;
import optimize.StopConditions;

import weka.core.Instance;
import weka.core.Instances;

import weka.classifiers.AbstractClassifier;

import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.OptionHandler;
import weka.core.Utils;

public class wdAnJE extends AbstractClassifier implements OptionHandler {

	private static final long serialVersionUID = 4823531716976859217L;

	private Instances m_Instances;

	int nInstances;
	int nAttributes;
	int nc;
	int[] paramsPerAtt;

	private String m_S = "A1JE"; 						// -S (A1JE, A2JE, A3JE, A4JE, A5JE)
	private String m_P = "MAP";  						// -P (MAP, dCCBN, wCCBN, eCCBN, MAP2, wCCBN2)
	private String m_E = "CLL";  						// -E (CLL, MSE)
	private String m_I = "Flat"; 							// -I (Flat, Indexed, IndexedBig, BitMap)

	private String m_O = "QN"; 						// -O (QN, GD, CG)

	private String m_X = "None"; 						// -X (None, ChiSqTest, GTest, FisherExactTest, MI)


	private boolean m_Discretization	 	= false; 			// -D
	private boolean m_MVerb 					= false; 			// -V		
	private boolean m_MultiThreaded		= false; 			// -M
	private int m_WeightingInitialization 	= 0; 	        			// -W 0

	private boolean m_ReConditioning 	= false; 			// -G
	private boolean m_PreScaling            = false;			// - J
	private boolean m_Regularization 		= false; 			// -R
	private double m_Lambda 					= 0.01; 			// -L 

	private int m_MaxIterations = 10000;						// -C 

	private boolean m_MThreadVerb = false;					// -T

	private ObjectiveFunction function_to_optimize;

	private double maxGradientNorm = 0.000000000000000000000000000000001;

	private double[] probs;	
	private int numTuples;

	private weka.filters.supervised.attribute.Discretize m_Disc = null;

	public wdAnJEParameters dParameters_;
	private LogDistributionComputerAnJE logDComputer;

	private boolean isFeelders = false;

	@Override
	public void buildClassifier(Instances instances) throws Exception {

		Instances  m_DiscreteInstances = null;

		// can classifier handle the data?
		getCapabilities().testWithFail(instances);

		// Discretize instances if required
		if (m_Discretization) {
			m_Disc = new weka.filters.supervised.attribute.Discretize();
			m_Disc.setUseBinNumbers(true);
			m_Disc.setInputFormat(instances);
			System.out.println("Applying Discretization Filter - dodo");
			m_DiscreteInstances = weka.filters.Filter.useFilter(instances, m_Disc);
			System.out.println("Done");

			m_Instances = new Instances(m_DiscreteInstances);
			m_DiscreteInstances = new Instances(m_DiscreteInstances, 0);
		} else {
			m_Instances = new Instances(instances);
			instances = new Instances(instances, 0);
		}

		// Remove instances with missing class
		m_Instances.deleteWithMissingClass();

		// All done, gather statistics
		nInstances = m_Instances.numInstances();
		nAttributes = m_Instances.numAttributes() - 1;		
		nc = m_Instances.numClasses();

		probs = new double[nc];		

		paramsPerAtt = new int[nAttributes];
		for (int u = 0; u < nAttributes; u++) {
			paramsPerAtt[u] = m_Instances.attribute(u).numValues();
		}		

		/*
		 * Initialize structure array based on m_S
		 */
		if (m_S.equalsIgnoreCase("A1JE")) {
			numTuples = 1; // NB
		} else if (m_S.equalsIgnoreCase("A2JE")) {		
			numTuples = 2;
		} else if (m_S.equalsIgnoreCase("A3JE")) {
			numTuples = 3;	
		} else if (m_S.equalsIgnoreCase("A4JE")) {
			numTuples = 4;	
		} else if (m_S.equalsIgnoreCase("A5JE")) {
			numTuples = 5;	
		} else {
			System.out.println("m_S value should be from set {A1JE, A2JE, A3JE, A4JE, A5JE}");
		}		

		/* 
		 * ----------------------------------------------------------------------------------------
		 * Start Parameter Learning Process
		 * ----------------------------------------------------------------------------------------
		 */

		int scheme = 1;

		/*
		 * ---------------------------------------------------------------------------------------------
		 * Intitialize data structure
		 * ---------------------------------------------------------------------------------------------
		 */

		if (m_P.equalsIgnoreCase("MAP") || m_P.equalsIgnoreCase("MAP2")) {	
			/*
			 * MAP - Maximum Likelihood Estimates of the Parameters characterzing P(x_i|y)
			 * MAP2 - MLE of parameters characterizing P(y|x_i)
			 */
			scheme = plTechniques.MAP;			

		} else if (m_P.equalsIgnoreCase("dCCBN")) {

			scheme = plTechniques.dCCBN;

		} else if (m_P.equalsIgnoreCase("dCCBNf")) {

			scheme = plTechniques.dCCBNf;

		} else if (m_P.equalsIgnoreCase("wCCBN")) {

			scheme = plTechniques.wCCBN;

		} else if (m_P.equalsIgnoreCase("wCCBNf")) {

			scheme = plTechniques.wCCBNf;

		} else if (m_P.equalsIgnoreCase("wCCBN2")) {

			scheme = plTechniques.wCCBN2;

		} else if (m_P.equalsIgnoreCase("eCCBN")) {
			//TODO						
		} else {
			//System.out.println("m_P value should be from set {MAP, dCCBN, wCCBN, dCCBNf, wCCBNf, eCCBN, MAP2, wCCBN2}");
			System.out.println("m_P value should be from set {MAP, dCCBN (LR), wCCBN (ALR), dCCBNf (LR Feelders), wCCBNf (ALR Feelders), eCCBN (ELR), MAP2 (discriminative MAP), wCCBN2 (ALR discriminative MAP)}");
		}

		logDComputer = LogDistributionComputerAnJE.getDistributionComputer(numTuples, scheme);

		if (m_I.equalsIgnoreCase("Flat")) {
			dParameters_ = new wdAnJEParametersFlat(nAttributes, nc, nInstances, paramsPerAtt, scheme, numTuples, m_X);				
		} else if (m_I.equalsIgnoreCase("Indexed")) {
			dParameters_ = new wdAnJEParametersIndexed(nAttributes, nc, nInstances, paramsPerAtt, scheme, numTuples, m_X);				
		} else if (m_I.equalsIgnoreCase("IndexedBig")) {
			dParameters_ = new wdAnJEParametersIndexedBig(nAttributes, nc, nInstances, paramsPerAtt, scheme, numTuples, m_X);				
		} else if (m_I.equalsIgnoreCase("BitMap")) {
			dParameters_ = new wdAnJEParametersBitmap(nAttributes, nc, nInstances, paramsPerAtt, scheme, numTuples, m_X);				
		} else {
			System.out.println("m_I value should be from set {Flat, Indexed, IndexedBig, BitMap}");
		}

		/*
		 * ---------------------------------------------------------------------------------------------
		 * Create Data Structure by leveraging ONE or TWO pass through the data
		 * (These routines are common to all parameter estimation methods)
		 * ---------------------------------------------------------------------------------------------
		 */		
		if (m_MultiThreaded) {

			dParameters_.updateFirstPass_m(m_Instances);
			System.out.println("Finished first pass.");

			dParameters_.finishedFirstPass();

			if (dParameters_.needSecondPass() ){
				dParameters_.updateAfterFirstPass_m(m_Instances);				
				System.out.println("Finished second pass.");
			}

		} else {

			for (int i = 0; i < nInstances; i++) {
				Instance instance = m_Instances.instance(i);
				dParameters_.updateFirstPass(instance);				
			}
			System.out.println("Finished first pass.");

			dParameters_.finishedFirstPass();

			if (dParameters_.needSecondPass() ){
				for (int i = 0; i < nInstances; i++) {
					Instance instance = m_Instances.instance(i);
					dParameters_.updateAfterFirstPass(instance);				
				}
				System.out.println("Finished second pass.");
			}
		}

		/*
		 * Check if Feature Selection needs to be done.
		 */				
		if (dParameters_.needFeatureSelection()) {
			System.out.println("Feature Selection Flag is On.");
			System.out.println("Reallocating Counts, Probs and Gradient vectors based on FS Results");
			dParameters_.updateVectorsBasedOnFS();
		}

		/*
		 * Routine specific operations.
		 */

		System.out.println("All data structures are initialized. Starting to estimate parameters.");

		if (m_P.equalsIgnoreCase("MAP") || m_P.equalsIgnoreCase("MAP2")) {

			/* 
			 * ------------------------------------------------------------------------------
			 * MAP - Maximum Likelihood Estimates of the Parameters characterzing P(x_i|y)
			 * MAP2 - MLE of parameters characterizing P(y|x_i)
			 * ------------------------------------------------------------------------------
			 */

			if (m_P.equalsIgnoreCase("MAP2"))
				dParameters_.convertToProbs_Cond();
			else
				dParameters_.convertToProbs();			

		} else if (m_P.equalsIgnoreCase("dCCBN")) {

			/*
			 * ------------------------------------------------------------------------------
			 * Classic high-order Logistic Regression
			 * ------------------------------------------------------------------------------			 
			 */

			dParameters_.convertToProbs();		

			dParameters_.initializeParameters_D(m_WeightingInitialization, isFeelders);

			if (m_MultiThreaded) {
				if (m_E.equalsIgnoreCase("CLL")) {
					function_to_optimize = new ParallelObjectiveFunctionCLL_d(this);					
				} else if (m_E.equalsIgnoreCase("MSE")) {
					function_to_optimize = new ParallelObjectiveFunctionMSE_d(this);
				}
			} else {
				if (m_E.equalsIgnoreCase("CLL")) {
					function_to_optimize = new ObjectiveFunctionCLL_d(this);					
				} else if (m_E.equalsIgnoreCase("MSE")) {
					function_to_optimize = new ObjectiveFunctionMSE_d(this);
				}
			}

		} else if (m_P.equalsIgnoreCase("dCCBNf")) {

			/*
			 * ------------------------------------------------------------------------------
			 * Classic high-order Logistic Regression (Feelders implementation)
			 * ------------------------------------------------------------------------------
			 */

			dParameters_.convertToProbs();			

			dParameters_.initializeParameters_D(m_WeightingInitialization, isFeelders);

			if (m_MultiThreaded) {
				if (m_E.equalsIgnoreCase("CLL")) {
					function_to_optimize = new ParallelObjectiveFunctionCLL_df(this);
				} else if (m_E.equalsIgnoreCase("MSE")) {
					// TODO
				}
			} else {
				if (m_E.equalsIgnoreCase("CLL")) {
					function_to_optimize = new ObjectiveFunctionCLL_df(this);
				} else if (m_E.equalsIgnoreCase("MSE")) {
					// TODO
				}
			}

		} else if (m_P.equalsIgnoreCase("wCCBN")) {

			/*
			 * ------------------------------------------------------------------------------
			 * ALR
			 * ------------------------------------------------------------------------------
			 */

			dParameters_.convertToProbs();			

			//double scale = 1e5;
			//dParameters_.multiplyProbsWithAnJEWeight(scale);

			if (isM_PreScaling()) {
				//dParameters_.multiplyProbsWithAnJEWeight();
				dParameters_.multiplyProbsWithAnJEWeightOpt();
			}

			dParameters_.initializeParameters_W(m_WeightingInitialization, isFeelders);						

			if (m_MultiThreaded) {
				if (m_E.equalsIgnoreCase("CLL")) {
					function_to_optimize = new ParallelObjectiveFunctionCLL_w(this);
				} else if (m_E.equalsIgnoreCase("MSE")) {
					function_to_optimize = new ParallelObjectiveFunctionMSE_w(this);
				}
			} else {
				if (m_E.equalsIgnoreCase("CLL")) {
					function_to_optimize = new ObjectiveFunctionCLL_w(this);
				} else if (m_E.equalsIgnoreCase("MSE")) {
					function_to_optimize = new ObjectiveFunctionMSE_w(this);
				}
			}

		} else if (m_P.equalsIgnoreCase("wCCBNf")) {

			/*
			 * ------------------------------------------------------------------------------
			 * ALR (Feelders implementation)
			 * ------------------------------------------------------------------------------
			 */

			dParameters_.convertToProbs_F();
			//dParameters_.convertToProbs();

			isFeelders = true;			
			dParameters_.initializeParameters_W(m_WeightingInitialization, isFeelders);

			if (m_MultiThreaded) {
				if (m_E.equalsIgnoreCase("CLL")) {
					function_to_optimize = new ParallelObjectiveFunctionCLL_wf(this);
				} else if (m_E.equalsIgnoreCase("MSE")) {
					// TODO
				}
			} else {
				if (m_E.equalsIgnoreCase("CLL")) {
					function_to_optimize = new ObjectiveFunctionCLL_wf(this);
				} else if (m_E.equalsIgnoreCase("MSE")) {
					// TODO
				}
			}

		} else if (m_P.equalsIgnoreCase("wCCBN2")) {

			/*
			 * ------------------------------------------------------------------------------
			 * DBL (discriminative optimization - Bin Liu's idea)
			 * ------------------------------------------------------------------------------
			 */

			dParameters_.convertToProbs_Cond();

			dParameters_.initializeParameters_W(m_WeightingInitialization, isFeelders);

			if (m_E.equalsIgnoreCase("CLL")) {
				function_to_optimize = new ObjectiveFunctionCLL_w2(this);
			} 

		} else if (m_P.equalsIgnoreCase("eCCBN")) {
			//TODO				
			/* 
			 * Implement ELR here
			 */			
		} else {
			System.out.println("m_P value should be from set {MAP, dCCBN (LR), wCCBN (ALR), dCCBNf (LR Feelders), wCCBNf (ALR Feelders), eCCBN (ELR), MAP2 (discriminative MAP), wCCBN2 (ALR discriminative MAP)}");
		}

		/*
		 * Train the classifier on initialized data structure.
		 */

		if (m_P.equalsIgnoreCase("MAP") || m_P.equalsIgnoreCase("MAP2")) {

			// Do nothing
			System.out.print("NLL (MAP) = " + dParameters_.getNLL(m_Instances, logDComputer) + "\n");

		} else if (m_MaxIterations != 0) {

			if (m_O.equalsIgnoreCase("QN")) {

				Minimizer alg = new Minimizer();
				StopConditions sc = alg.getStopConditions();
				//sc.setFunctionReductionFactor(1e1);
				//sc.setFunctionReductionFactorInactive();
				sc.setMaxGradientNorm(maxGradientNorm);	
				sc.setMaxIterations(m_MaxIterations);

				Result result;	

				// Call the lbfgs optimizer
				if (isM_MVerb()) {
					System.out.println();
					System.out.print("fx_QN = [");

					System.out.print(dParameters_.getNLL(m_Instances, logDComputer) + ", ");

					alg.setIterationFinishedListener((p,nll,g)->{System.out.print(nll+", "); return true;});
					result = alg.run(function_to_optimize, dParameters_.getParameters());
					System.out.println("];");
					System.out.println(result);

					System.out.println("NoIter = " + result.iterationsInfo.iterations);
					System.out.println();

				} else {
					result = alg.run(function_to_optimize, dParameters_.getParameters());
					System.out.println("NoIter = " + result.iterationsInfo.iterations);
				}

				function_to_optimize.finish();

			} else if (m_O.equalsIgnoreCase("GD")) {

				MinimizerGD alg = new MinimizerGD();
				alg.setMaxIterations(m_MaxIterations);

				Result result;	

				if (isM_ReConditioning()) {

					System.out.println("---->");
					System.out.print("fx_GD = [");

					for (int i = 0; i < 200; i++) {
						alg.setMaxIterations(1);
						result = alg.run(function_to_optimize, dParameters_.getParameters());
					}

					System.out.println("];");

				} else {

					if (isM_MVerb()) {
						System.out.println("---->");
						System.out.print("fx_GD = [");

						result = alg.run(function_to_optimize, dParameters_.getParameters());
						System.out.println("];");

						System.out.println("NoIter = " + result.iterationsInfo.iterations);
						System.out.println();

					} else {
						result = alg.run(function_to_optimize, dParameters_.getParameters());
						System.out.println("NoIter = " + result.iterationsInfo.iterations);
					}

					function_to_optimize.finish();
				}

			} else if (m_O.equalsIgnoreCase("CG")) {

				MinimizerCG alg = new MinimizerCG();
				alg.setMaxIterations(m_MaxIterations);

				Result result;

				if (isM_MVerb()) {
					System.out.println("---->");
					System.out.print("fx_CG = [");

					System.out.print(dParameters_.getNLL(m_Instances, logDComputer) + ", ");

					result = alg.run(function_to_optimize, dParameters_.getParameters());
					System.out.println("];");

					System.out.println("NoIter = " + result.iterationsInfo.iterations);
					System.out.println();

				} else {
					result = alg.run(function_to_optimize, dParameters_.getParameters());
					System.out.println("NoIter = " + result.iterationsInfo.iterations);
				}

				function_to_optimize.finish();

			}

		}

		// free up some space
		m_Instances = new Instances(m_Instances, 0);
	}

	@Override
	public double[] distributionForInstance(Instance instance) {

		if (m_Discretization) {
			synchronized(m_Disc) {	
				m_Disc.input(instance);
				instance = m_Disc.output();
			}
		}

		double[] probs = logDistributionForInstance(instance);
		SUtils.exp(probs);
		return probs;
	}	

	public double[] logDistributionForInstance(Instance inst) {
		double[] probs = new double[nc];
		logDistributionForInstance(probs,inst) ;
		return probs;
	}

	public void logDistributionForInstance(double [] probs,Instance inst) {
		logDComputer.compute(probs, dParameters_, inst);
		SUtils.normalizeInLogDomain(probs);
	}

	// ----------------------------------------------------------------------------------
	// Weka Functions
	// ----------------------------------------------------------------------------------

	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		// class
		result.enable(Capability.NOMINAL_CLASS);
		// instances
		result.setMinimumNumberInstances(0);
		return result;
	}

	@Override
	public void setOptions(String[] options) throws Exception {

		m_Discretization = Utils.getFlag('D', options);
		m_MVerb = Utils.getFlag('V', options);

		//m_WeightingInitialization = Utils.getFlag('W', options);
		String SW = Utils.getOption('W', options);
		if (SW.length() != 0) {
			// m_S = Integer.parseInt(SK);
			m_WeightingInitialization = Integer.parseInt(SW);
		}

		m_MultiThreaded = Utils.getFlag('M', options);
		m_MThreadVerb = Utils.getFlag('T', options);

		String SK = Utils.getOption('S', options);
		if (SK.length() != 0) {
			// m_S = Integer.parseInt(SK);
			m_S = SK;
		}

		String MP = Utils.getOption('P', options);
		if (MP.length() != 0) {
			// m_P = Integer.parseInt(MP);
			m_P = MP;
		}

		String ME = Utils.getOption('E', options);
		if (ME.length() != 0) {
			m_E = ME;
		}

		String MI = Utils.getOption('I', options);
		if (MI.length() != 0) {
			m_I = MI;
		}

		String MX = Utils.getOption('X', options);
		if (MX.length() != 0) {
			m_X = MX;
		}

		String MO = Utils.getOption('O', options);
		if (MO.length() != 0) {
			m_O = MO;
		}

		String CK = Utils.getOption('C', options);
		if (CK.length() != 0) {
			m_MaxIterations = Integer.parseInt(CK);			
		}

		m_Regularization = Utils.getFlag('R', options);

		m_ReConditioning = Utils.getFlag('G', options); 

		m_PreScaling = Utils.getFlag('J', options);
		
		String strL = Utils.getOption('L', options);
		if (strL.length() != 0) {
			m_Lambda = (Double.valueOf(strL));
		}

		Utils.checkForRemainingOptions(options);
	}

	@Override
	public String[] getOptions() {
		String[] options = new String[3];
		int current = 0;
		while (current < options.length) {
			options[current++] = "";
		}
		return options;
	}

	public static void main(String[] argv) {
		runClassifier(new wdAnJE(), argv);
	}

	public boolean getRegularization() {
		return m_Regularization;
	}

	public double getLambda() {
		return m_Lambda;
	}

	public String getMS() {
		return m_S;
	}

	public int getNInstances() {
		return nInstances;
	}

	public int getNc() {
		return nc;
	}

	public int getnAttributes() {
		return nAttributes;
	}

	public wdAnJEParameters getdParameters_() {
		return dParameters_;
	}

	public Instances getM_Instances() {
		return m_Instances;
	}

	public boolean isM_MVerb() {
		return m_MVerb;
	}

	public boolean isM_MThreadVerb() {
		return m_MThreadVerb;
	}

	public boolean isM_ReConditioning() {
		return m_ReConditioning;
	}

	public void setM_ReConditioning(boolean m_ReConditioning) {
		this.m_ReConditioning = m_ReConditioning;
	}
	
	public boolean isM_PreScaling() {
		return m_PreScaling;
	}

	public void setM_PreScaling(boolean m_PreScaling) {
		this.m_PreScaling = m_PreScaling;
	}

}
