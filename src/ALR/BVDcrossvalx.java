package ALR;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Reader;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;

/**
 * Class for performing a Bias-Variance decomposition on any classifier 
 * using the method specified in:<p>
 * 
 * R. Kohavi & D. Wolpert (1996), <i>Bias plus variance decomposition for 
 * zero-one loss functions</i>, in Proc. of the Thirteenth International 
 * Machine Learning Conference (ICML96) 
 * <a href="http://robotics.stanford.edu/~ronnyk/biasVar.ps">
 * download postscript</a>.<p>
 *
 * but modified to use x-fold cross validation.
 * and outputs the root mean squared error (friday, 13th oct, 2006)
 *
 * Valid options are:<p>
 *
 * -D <br>
 * Turn on debugging output.<p>
 *
 * -W classname <br>
 * Specify the full class name of a learner to perform the 
 * decomposition on (required).<p>
 *
 * -t filename <br>
 * Set the arff file to use for the decomposition (required).<p>
 *
 * -c num <br>
 * Specify the index of the class attribute (default last).<p>
 *
 * -i num <br>
 * Set the number of train iterations (default 10). <p>
 *
 * -x num <br>
 * Set the number of folds for cross validation (default 3). <p>
 *
 * -s num <br>
 * Set the seed for the dataset randomisation (default 1). <p>
 *
 * -V <br>
 * Set flag for calculating variance separately from bias<br>
 * (default is variance = error - bias). <p>
 *
 * Options after -- are passed to the designated sub-learner. <p>
 *
 * @author Janice Boughton
 * @version $Revision: 1.0.0.0 $
 */
public class BVDcrossvalx implements OptionHandler {
	
	/** Debugging mode, gives extra output if true */
	protected boolean m_Debug;

	/** An instantiated base classifier used for getting and testing options. */
	protected Classifier m_Classifier = new weka.classifiers.rules.ZeroR();

	/** The options to be passed to the base classifier. */
	protected String [] m_ClassifierOptions;

	/** The number of train iterations */
	protected int m_TrainIterations = 10;

	/** The number of folds for cross-validation */
	protected int m_NumFolds = 3;

	/** The name of the data file used for the decomposition */
	protected String m_DataFileName;

	/** The index of the class attribute */
	protected int m_ClassIndex = -1;

	/** The random number seed */
	protected int m_Seed = 1;

	/** The calculated bias (squared) */
	protected double m_Bias;

	/** The calculated variance */
	protected double m_Variance;

	/** The calculated sigma (squared) */
	protected double m_Sigma;

	/** The error rate */
	protected double m_Error;

	/** The root mean squared error */
	protected double m_RootMeanSquaredError = 0.0;

	/** The root mean squared error (V2) */
	protected double m_RootMeanSquaredErrorV2 = 0.0;

	protected double m_TotalIncorrect = 0;

	protected double m_Total = 0;

	/** The whatever the hell this is */
	protected double m_InfoLossFunc = 0.0;

	protected double m_Train = 0;

	protected double m_Test = 0;

	/** The number of instances used in the training pool */
	// protected int m_TrainPoolSize = 100;

	/**
	 * Flag for calculating variance (true = separete calculation; 
	 * false = variance is error minus bias)
	 */
	protected boolean m_CalcVarianceSeparately = false;

	/**
	 * Multi-thread the classification process
	 */
	private boolean m_MultiThreaded = false; 			// -M
	
	private int[][] confusionMatrix;
	private int[] infoMatrix;

	/**
	 * Carry out the bias-variance decomposition
	 *
	 * @exception Exception if the decomposition couldn't be carried out
	 */
	public void decompose() throws Exception {

		Reader dataReader = new BufferedReader(new FileReader(m_DataFileName));
		Instances data = new Instances(dataReader);

		// Variables for calculating the train and test times
		double trainTime = 0, testTime = 0;
		double trainStart = 0, testStart = 0;

		if (m_ClassIndex < 0) {
			data.setClassIndex(data.numAttributes() - 1);
		} else {
			data.setClassIndex(m_ClassIndex);
		}
		if (data.classAttribute().type() != Attribute.NOMINAL) {
			throw new Exception("Class attribute must be nominal");
		}
		int numClasses = data.numClasses();

		// Random number generator used to shuffle data for each iteration
		Random random = new Random(m_Seed);

		/*
		 *  Array of indices, so we can shuffle data without losing track of which 
		 *  instance is which. Initialize to current order of data
		 */
		int [] iindex = new int[data.numInstances()];
		for (int x = 0; x < data.numInstances(); x++)
			iindex[x] = x;

		// Tracks how many times each class was predicted for each instance
		double [][] instanceProbs = new double [data.numInstances()][numClasses];

		// Determine number of instances per fold (+ leftovers if not divided evenly)
		int numInstancesInFold = data.numInstances() / m_NumFolds;
		int leftOver = data.numInstances() % m_NumFolds;

		// Math.log(2) -- divide Math.log(x) by this to get the base 2 log
		// (this is used when calculating informational loss function.
		double log2 = Math.log(2);

		// Get the train set (all instances except those in test set)
		Instances train = new Instances(data, data.numInstances() - numInstancesInFold);

		int testSetSize = 0;		
		m_Error = 0; m_RootMeanSquaredError = 0; m_InfoLossFunc = 0;
		
		confusionMatrix = new int[data.numClasses()][data.numClasses()];
		infoMatrix = new int[data.numClasses()];		

		for (int i = 0; i < m_TrainIterations; i++) {
			
			System.out.println(" ----------------------------------------------------------------------- ");
			System.out.println(" ---------------------- Iteration " + i + " ------------------------------------ ");
			System.out.println(" ----------------------------------------------------------------------- \n");

			System.out.println("Shuffling Data.");
			randomize(iindex, random);
			System.out.println("Shuffling Data Finished. \n");
			
			// For the first fold, the testdata begins at the start of the dataset.
			// All instances after testSetSize are in the trainset
			int testStartIndex = 0;

			for (int fold = 0; fold < m_NumFolds; fold++) {
				
				System.out.println("Building Classifier. Fold " + fold);

				// reset the train set to zero instances
				train.removeAll(train);

				// size of test set
				testSetSize = (fold < leftOver) ? numInstancesInFold + 1 : numInstancesInFold;

				for (int x = 0; x < data.numInstances(); x++) {
					if (x == testStartIndex) {
						x += testSetSize;  
						// skipping instances in the test set break out of loop if 
						// there aren't anymore instances to add
						if (x >= data.numInstances())
							break;
					}

					train.add(data.instance(iindex[x]));
				}

				trainStart = System.currentTimeMillis();

				Classifier current = AbstractClassifier.makeCopy(m_Classifier);
				current.buildClassifier(train);

				trainTime += (System.currentTimeMillis() - trainStart);				
				
				System.out.println("Training Fininshed.");
				System.out.println();
				
				// ------------------------------------------------------------------------------------------
				// start the timer for testing
				// ------------------------------------------------------------------------------------------
				testStart = System.currentTimeMillis();

				if (m_MultiThreaded && testSetSize > 10000) {
					
					System.out.println("Testing Classifier.");
					//String perfOutput = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-+*`~!@#$%^&_|:;'?";

					int nThreads;
					int minNPerThread = 4000;					
					int N = testSetSize;

					double[][] threadResults;
					int[][][] threadConfusionMatrix;
					int[][] threadInfoMatrix;
					
					ExecutorService executor;

					if (N < minNPerThread) {
						nThreads = 1;
					} else {
						nThreads = Runtime.getRuntime().availableProcessors();
						if (N/nThreads < minNPerThread) {
							nThreads = N/minNPerThread + 1;
						}
					}					
					System.out.println("In BVDCrossvalx(): Launching " + nThreads + " threads");
					
					threadConfusionMatrix = new int[nThreads][data.numClasses()][data.numClasses()];
					threadInfoMatrix = new int[nThreads][data.numClasses()];
					
					threadResults = new double[nThreads][3];
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
						Callable<Double> thread = new CallableBVDCrossvalx(start, stop, threadResults[th], instanceProbs, data, iindex, testStartIndex, current, th, threadConfusionMatrix[th], threadInfoMatrix[th]);

						futures[th] = executor.submit(thread);
					}

					for (int th = 0; th < nThreads; th++) {
						
						try {
							m_Error += futures[th].get();
						} catch (InterruptedException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						} catch (ExecutionException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}						
						
						m_RootMeanSquaredError += threadResults[th][0];
						m_InfoLossFunc += threadResults[th][1];
						m_RootMeanSquaredErrorV2 += threadResults[th][2];
						
						for (int ca = 0; ca < data.numClasses(); ca++) {
							infoMatrix[ca] += threadInfoMatrix[th][ca];
							for (int cp = 0; cp < data.numClasses(); cp++) {				
								confusionMatrix[ca][cp] += threadConfusionMatrix[th][ca][cp];
							}							
						}
					}
					
					executor.shutdown();
					System.out.println();
					System.out.println("In BVDCrossvalx(): All threads finished.");
					System.out.println("Testing Finished.");
					System.out.println();

				} else {

					System.out.println("Testing Classifier.");
					
					double tempError = 0;
					// Evaluate the classifier on test, updating BVD stats
					for (int j = 0; j < testSetSize; j++) {
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
								System.err.println("probs[ " + y + "] is NaN! oh no!");
							}							
						}
						confusionMatrix[actualClass][pred]++;
						infoMatrix[actualClass]++;

						m_RootMeanSquaredErrorV2 += Math.pow((1-probs[actualClass]), 2); 							
						m_InfoLossFunc += -(Math.log(probs[actualClass]) / log2);
						if (pred != actualClass) {
							m_Error++;
							tempError++;
						}
						instanceProbs[iindex[testStartIndex+j]][pred]++;
						m_RootMeanSquaredError += (RMSEy / data.numClasses());
					} // ends j
					
					System.out.println("Testing Finished.");
					System.out.println();
				}

				testStartIndex += testSetSize;
			} // ends folds

			testTime += (System.currentTimeMillis() - testStart);  
			System.out.println("After this Iteration - Error is: " + m_Error/(data.numInstances() * (i + 1)));
			System.out.println("After this Iteration - RMSE is: " + Math.sqrt(m_RootMeanSquaredError/(data.numInstances() * (i + 1))));
			System.out.println();
		} // ends Iterations

		System.out.println("Accumulating Bias-Variance Statistics from the results.");
		
		//System.out.println("Incorreclty Classifierd / Total: " + m_Error + "/" + (data.numInstances() * m_TrainIterations));

		// Average the BV over each instance in test.
		m_RootMeanSquaredError = Math.sqrt(m_RootMeanSquaredError / (data.numInstances() * m_TrainIterations)); 
		m_RootMeanSquaredErrorV2 = Math.sqrt(m_RootMeanSquaredErrorV2 / (data.numInstances() * m_TrainIterations));
		m_InfoLossFunc /= (data.numInstances() * m_TrainIterations);

		m_Bias = 0;
		m_Sigma = 0;
		m_Variance = 0;

		for (int i = 0; i < data.numInstances(); i++) {
			Instance current = data.instance(i);
			double [] predProbs = instanceProbs[i];
			double pActual, pPred;
			double bsum = 0, vsum = 0, ssum = 0;
			for (int j = 0; j < numClasses; j++) {
				pActual = (current.classValue() == j) ? 1 : 0;
				pPred = predProbs[j] / m_TrainIterations;
				bsum += (pActual - pPred) * (pActual - pPred) - pPred * (1 - pPred) / (m_TrainIterations - 1);
				vsum += (pPred * pPred);
				ssum += pActual * pActual;
			}
			m_Bias += bsum;
			m_Variance += (1 - vsum);
			m_Sigma += (1 - ssum);
		}

		m_TotalIncorrect = m_Error;
		m_Total = data.numInstances() * m_TrainIterations;
		m_Error /= (data.numInstances() * m_TrainIterations);
		m_Bias /= (2.0 * data.numInstances());
		m_Sigma /= (2.0 * data.numInstances());
		if (m_CalcVarianceSeparately)
			m_Variance /= (2 * data.numInstances());
		else 
			m_Variance = m_Error - m_Bias;

		m_Train = trainTime;
		m_Test = testTime;
		
		System.out.println("Finished accumulating Bias-variance Statistics. All Done.\n");
		System.out.println(" ----------------------------------------------------------------------- \n\n");
		
		System.out.println("Class Distribution");
		
		System.out.format("%10s","");
		for (int ca = 0; ca < data.numClasses(); ca++) {
			System.out.format("%10d,", ca);
		}
		System.out.println();
		
		System.out.format("%10s","");
		for (int ca = 0; ca < data.numClasses(); ca++) {
			System.out.format("%10d,",infoMatrix[ca]/(m_TrainIterations));
		}
		
		System.out.println("\n\n Confusion Matrix (actual down, pred ->)");
		
		System.out.format("%10s","");
		for (int ca = 0; ca < data.numClasses(); ca++) {
			System.out.format("%10d,", ca);
		}
		System.out.println();
		
		for (int ca = 0; ca < data.numClasses(); ca++) {
			System.out.format("%10d", ca);
			for (int cp = 0; cp < data.numClasses(); cp++) {				
				System.out.format("%10d,",confusionMatrix[ca][cp]/(m_TrainIterations));
			}
			System.out.println();
		}
		
		
	}
	
	/**
	 * Accepts an array of ints and randomises the values in the 
	 * array, using the random seed.
	 *
	 *@param index is the array of integers
	 *@param random is the Random seed.
	 */
	public final void randomize(int[] index, Random random) {
		for (int j = index.length - 1; j > 0; j-- ){
			int k = random.nextInt( j + 1 );
			int temp = index[j];
			index[j] = index[k];
			index[k] = temp;
		}
	}


	/**
	 * Returns description of the bias-variance decomposition results.
	 *
	 * @return the bias-variance decomposition results as a string
	 */
	@Override
	public String toString() {

		boolean m_XMLOutput = false;
		boolean m_StraightOutput = false;

		if (m_XMLOutput) {

			if (getClassifier() == null) {
				return "Invalid setup";
			}

			String result = "";
			String options = "";
			String classifier = extractClassifierName(getClassifier().getClass().getName());

			if (getClassifier() instanceof OptionHandler) {
				options = Utils.joinOptions(((OptionHandler)m_Classifier).getOptions());
			}
			String dataFile = extractDataFileName(getDataFileName());
			String index = "";
			if (getClassIndex() == 0) {
				index = "last";
			} else {
				index = getClassIndex() + "";
			}

			result += "<" + dataFile + ">";
			result += "\n<Classifier>" + classifier + "</Classifier>";			
			result += "\n<Options>" + options + "</Options>";
			result += "\n<DataFile>" + dataFile + "</DataFile>";
			result += "\n<ClassIndex>" + index + "</ClassIndex>";
			result += "\n<CalcVariance>" + (m_CalcVarianceSeparately ? "separate" : "error - bias") + "</CalcVariance>";			
			result += "\n<Iterations>" + getTrainIterations() + "</Iterations>";			
			result += "\n<Folds>" + getNumFolds() + "</Folds>";			
			result += "\n<Seed>" + getSeed() + "</Seed>";
			result += "\n<Error>" + Utils.doubleToString(getError(), 6, 4) + "</Error>";
			result += "\n<Sigma2>" + Utils.doubleToString(getSigma(), 6, 4) + "</Sigma2>";
			result += "\n<Bias2>" + Utils.doubleToString(getBias(), 6, 4) + "</Bias2>";
			result += "\n<Variance>" + Utils.doubleToString(getVariance(), 6, 4) + "</Variance>";
			result += "\n<RMSE>" + Utils.doubleToString(getRMSE(), 6, 4) + "</RMSE>";
			result += "\n<InfoLoss>" + Utils.doubleToString(getInfoLoss(), 6, 4) + "</InfoLoss>";
			result += "\n<TrainingTime>" + Utils.doubleToString(m_Train / 1000, 6, 4) + "</TrainingTime>";
			result += "\n<TestTime>" + Utils.doubleToString(m_Test / 1000, 6, 4) + "</TestTime>";

			result += "\n</" + dataFile + ">";
			result += "\n";

			return result;

		} else if (m_StraightOutput) {
			String classifierName = getClassifier().getClass().getName();
			String classifierOptions;
			if (getClassifier() instanceof OptionHandler) 
				classifierOptions = Utils.joinOptions(((OptionHandler)m_Classifier).getOptions());
			else
				classifierOptions = "";		
			/*
			String result = classifierName + "," + classifierOptions + "," + getTrainIterations() + "," + getNumFolds() + "," + getDataFileName() + "," +
					Utils.doubleToString(getError(), 6, 4) + "," + Utils.doubleToString(getBias(), 6, 4) + "," + Utils.doubleToString(getVariance(), 6, 4) + 
					Utils.doubleToString(getRMSE(), 6, 4) + "," + Utils.doubleToString(m_Train/1000, 6, 4) + "," + Utils.doubleToString(m_Test/1000, 6, 4) + "\n";
			 */
			String result = Utils.doubleToString(getError(), 6, 4) + "," + Utils.doubleToString(getBias(), 6, 4) + "," + Utils.doubleToString(getVariance(), 6, 4) + "," +
					Utils.doubleToString(getRMSE(), 6, 4) + "," + Utils.doubleToString(m_Train/1000, 6, 4) + "," + Utils.doubleToString(m_Test/1000, 6, 4) + "\n";
			return result;
		} else {

			String result = "\nBias-Variance Decomposition\n";

			if (getClassifier() == null) {
				return "Invalid setup";
			}

			result += "\nClassifier   : " + getClassifier().getClass().getName();
			if (getClassifier() instanceof OptionHandler) {
				result += Utils.joinOptions(((OptionHandler)m_Classifier).getOptions());
			}
			result += "\nData File    : " + getDataFileName();
			result += "\nClass Index  : ";
			if (getClassIndex() == 0) {
				result += "last";
			} else {
				result += getClassIndex();
			}

			result += "\nCalcVariance : " + (m_CalcVarianceSeparately 
					? "separate" : "error - bias");
			result += "\nIterations       : " + getTrainIterations();
			result += "\nFolds            : " + getNumFolds();
			result += "\nSeed             : " + getSeed();
			result += "\nError            : " + Utils.doubleToString(getError(), 6, 4);
			result += "\nSigma^2          : " + Utils.doubleToString(getSigma(), 6, 4);
			result += "\nBias^2           : " + Utils.doubleToString(getBias(), 6, 4);
			result += "\nVariance         : " + Utils.doubleToString(getVariance(), 6, 4);
			result += "\nRMSE             : " + Utils.doubleToString(getRMSE(), 6, 4);
			result += "\nRMSEV2           : " + Utils.doubleToString(getRMSEV2(), 6, 4);
			result += "\nInfo Loss        : " + Utils.doubleToString(getInfoLoss(), 6, 4);
			result += "\nTotal Incorrect  : " + Utils.doubleToString(getTotalIncorrect(), 6, 4);
			result += "\nTotal            : " + Utils.doubleToString(getTotal(), 6, 4);
			result += "\nTraining Time    : " + Utils.doubleToString(m_Train / 1000, 6, 4) + "seconds";
			result += "\nTest Time        : " + Utils.doubleToString(m_Test / 1000, 6, 4) + "seconds";		

			return result + "\n";
		}

	}

	/**
	 *  Gets String like weka.classifiers.trees.RF and returns RF
	 *  to be later used for XML
	 */
	public String extractClassifierName(String name) {
		String classifier = "";
		classifier = name.substring((name.lastIndexOf('.'))+1, name.length());				
		return classifier;
	}

	/**
	 *  Gets DataFile name as: /home/nayyar/workspaces/segment.arff
	 *  Returns: segment
	 */
	public String extractDataFileName(String name) {
		String dataFile = "";
		int beginIndex = name.lastIndexOf('/') + 1;
		int endIndex = name.lastIndexOf('.');
		dataFile = name.substring(beginIndex, endIndex);
		return dataFile;
	}

	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration listOptions() {

		Vector newVector = new Vector(7);

		newVector.addElement(new Option(
				"\tThe index of the class attribute.\n"+
						"\t(default last)",
						"c", 1, "-c <class index>"));
		newVector.addElement(new Option(
				"\tThe name of the arff file used for the decomposition.",
				"t", 1, "-t <name of arff file>"));
		newVector.addElement(new Option(
				"\tThe random number seed used.",
				"s", 1, "-s <seed>"));
		newVector.addElement(new Option(
				"\tThe number of training repetitions used.\n"
						+"\t(default 10)",
						"i", 1, "-i <num>"));
		newVector.addElement(new Option(
				"\tThe number of folds used in cross-validation.\n"
						+"\t(default 3)",
						"x", 1, "-x <num>"));
		newVector.addElement(new Option(
				"\tTurn on debugging output.",
				"D", 0, "-D"));
		newVector.addElement(new Option(
				"\tFull class name of the learner used in the decomposition.\n"
						+"\teg: weka.classifiers.bayes.NaiveBayes",
						"W", 1, "-W <classifier class name>"));
		newVector.addElement(new Option(
				"\tSet flag for separate variance calculation.",
				"V", 0, "-V"));

		if ((m_Classifier != null) &&
				(m_Classifier instanceof OptionHandler)) {
			newVector.addElement(new Option(
					"",
					"", 0, "\nOptions specific to learner "
							+ m_Classifier.getClass().getName()
							+ ":"));
			Enumeration enu = ((OptionHandler)m_Classifier).listOptions();
			while (enu.hasMoreElements()) {
				newVector.addElement(enu.nextElement());
			}
		}
		return newVector.elements();
	}

	/**
	 * Parses a given list of options. Valid options are:<p>
	 *
	 * -D <br>
	 * Turn on debugging output.<p>
	 *
	 * -W classname <br>
	 * Specify the full class name of a learner to perform the 
	 * decomposition on (required).<p>
	 *
	 * -t filename <br>
	 * Set the arff file to use for the decomposition (required).<p>
	 *
	 * -T num <br>
	 * Specify the number of instances in the training pool (default 100).<p>
	 *
	 * -c num <br>
	 * Specify the index of the class attribute (default last).<p>
	 *
	 * -i num <br>
	 * Set the number of train iterations (default 50). <p>
	 *
	 * -x num <br>
	 * Set the number of folds for cross validation (default 3). <p>
	 *
	 * -s num <br>
	 * Set the seed for the dataset randomisation (default 1). <p>
	 *
	 * -V <br>
	 * Set flag for calculating variance separately from bias<br>
	 * (default is variance = error - bias). <p>
	 *
	 * Options after -- are passed to the designated sub-learner. <p>
	 *
	 * @param options the list of options as an array of strings
	 * @exception Exception if an option is not supported
	 */
	@Override
	public void setOptions(String[] options) throws Exception {

		setDebug(Utils.getFlag('D', options));

		m_MultiThreaded = Utils.getFlag('M', options);

		String classIndex = Utils.getOption('c', options);
		if (classIndex.length() != 0) {
			if (classIndex.toLowerCase().equals("last")) {
				setClassIndex(0);
			} else if (classIndex.toLowerCase().equals("first")) {
				setClassIndex(1);
			} else {
				setClassIndex(Integer.parseInt(classIndex));
			}
		} else {
			setClassIndex(0);
		}

		String trainIterations = Utils.getOption('i', options);
		if (trainIterations.length() != 0) {
			setTrainIterations(Integer.parseInt(trainIterations));
		} else {
			setTrainIterations(10);
		}

		String numFolds = Utils.getOption('x', options);
		if (numFolds.length() != 0) {
			setNumFolds(Integer.parseInt(numFolds));
		} else {
			setNumFolds(3);
		}

		String seedString = Utils.getOption('s', options);
		if (seedString.length() != 0) {
			setSeed(Integer.parseInt(seedString));
		} else {
			setSeed(1);
		}

		String dataFile = Utils.getOption('t', options);
		if (dataFile.length() == 0) {
			throw new Exception("An arff file must be specified"
					+ " with the -t option.");
		}
		setDataFileName(dataFile);

		setCalcVarianceSeparately(Utils.getFlag('V', options));

		String classifierName = Utils.getOption('W', options);
		if (classifierName.length() == 0) {
			throw new Exception("A learner must be specified with the -W option.");
		}
		setClassifier(AbstractClassifier.forName(classifierName, Utils.partitionOptions(options)));
	}

	/**
	 * Gets the current settings of the CheckClassifier.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	@Override
	public String [] getOptions() {

		String [] classifierOptions = new String [0];
		if ((m_Classifier != null) && 
				(m_Classifier instanceof OptionHandler)) {
			classifierOptions = ((OptionHandler)m_Classifier).getOptions();
		}
		String [] options = new String [classifierOptions.length + 14];
		int current = 0;
		if (getDebug()) {
			options[current++] = "-D";
		}
		options[current++] = "-c"; options[current++] = "" + getClassIndex();
		options[current++] = "-i"; options[current++] = "" + getTrainIterations();
		options[current++] = "-x"; options[current++] = "" + getNumFolds();
		options[current++] = "-s"; options[current++] = "" + getSeed();
		if (getCalcVarianceSeparately()) {
			options[current++] = "-V";
		}
		if (getDataFileName() != null) {
			options[current++] = "-t"; options[current++] = "" + getDataFileName();
		}
		if (getClassifier() != null) {
			options[current++] = "-W";
			options[current++] = getClassifier().getClass().getName();
		}
		options[current++] = "--";
		System.arraycopy(classifierOptions, 0, options, current, 
				classifierOptions.length);
		current += classifierOptions.length;
		while (current < options.length) {
			options[current++] = "";
		}
		return options;
	}

	/**
	 * Get the number of folds used in cross-validation.
	 *
	 * @return number of folds in cross-validation.
	 */
	public int getNumFolds() {
		return m_NumFolds;
	}

	/**
	 * Set the number of folds used in cross-validation.
	 *
	 * @param numFolds number of folds used in cross-validation.
	 */
	public void setNumFolds(int numFolds) {
		m_NumFolds = numFolds;
	}

	/**
	 * set flag for calculating the variance separately (true) 
	 * or as error minus bias (false)
	 *
	 * @param val true or false
	 */
	public void setCalcVarianceSeparately(boolean val) {
		m_CalcVarianceSeparately = val;
	}

	/**
	 * return whether or not the variance should be calculated 
	 * separately or as error minus bias
	 *
	 * @return true if separate calculation, false otherwise
	 */
	public boolean getCalcVarianceSeparately() {
		return m_CalcVarianceSeparately;
	}

	/**
	 * Set the classifiers being analysed
	 *
	 * @param newClassifier the Classifier to use.
	 */
	public void setClassifier(Classifier newClassifier) {
		m_Classifier = newClassifier;
	}

	/**
	 * Gets the name of the classifier being analysed
	 *
	 * @return the classifier being analysed.
	 */
	public Classifier getClassifier() {
		return m_Classifier;
	}

	/**
	 * Sets debugging mode
	 *
	 * @param debug true if debug output should be printed
	 */
	public void setDebug(boolean debug) {
		m_Debug = debug;
	}

	/**
	 * Gets whether debugging is turned on
	 *
	 * @return true if debugging output is on
	 */
	public boolean getDebug() {
		return m_Debug;
	}

	/**
	 * Sets the random number seed
	 */
	public void setSeed(int seed) {
		m_Seed = seed;
	}

	/**
	 * Gets the random number seed
	 *
	 * @return the random number seed
	 */
	public int getSeed() {
		return m_Seed;
	}

	/**
	 * Sets the maximum number of boost iterations
	 */
	public void setTrainIterations(int trainIterations) {
		m_TrainIterations = trainIterations;
	}

	/**
	 * Gets the maximum number of boost iterations
	 *
	 * @return the maximum number of boost iterations
	 */
	public int getTrainIterations() {
		return m_TrainIterations;
	}

	/**
	 * Sets the maximum number of boost iterations
	 */
	public void setDataFileName(String dataFileName) {
		m_DataFileName = dataFileName;
	}

	/**
	 * Get the name of the data file used for the decomposition
	 *
	 * @return the name of the data file
	 */
	public String getDataFileName() {
		return m_DataFileName;
	}

	/**
	 * Get the index (starting from 1) of the attribute used as the class.
	 *
	 * @return the index of the class attribute
	 */
	public int getClassIndex() {
		return m_ClassIndex + 1;
	}

	/**
	 * Sets index of attribute to discretize on
	 *
	 * @param index the index (starting from 1) of the class attribute
	 */
	public void setClassIndex(int classIndex) {
		m_ClassIndex = classIndex - 1;
	}

	/**
	 * Get the calculated bias squared
	 *
	 * @return the bias squared
	 */
	public double getBias() {
		return m_Bias;
	} 

	/**
	 * Get the calculated variance
	 *
	 * @return the variance
	 */
	public double getVariance() {
		return m_Variance;
	}

	/**
	 * Get the calculated sigma squared
	 *
	 * @return the sigma squared
	 */
	public double getSigma() {
		return m_Sigma;
	}

	/**
	 * Get the calculated error rate
	 *
	 * @return the error rate
	 */
	public double getError() {
		return m_Error;
	}

	/**
	 * Get the calculated root mean squared error
	 *
	 * @return the root mean squared error
	 */
	public double getRMSE() {
		return m_RootMeanSquaredError;
	}

	/**
	 * Get the calculated informational loss function
	 *
	 * @return the informational loss function
	 */
	public double getInfoLoss() {
		return m_InfoLossFunc;
	}

	public double getRMSEV2() {
		return m_RootMeanSquaredErrorV2;
	}

	public double getTotalIncorrect() {
		return m_TotalIncorrect;
	}

	public double getTotal() {
		return m_Total;
	}

	/**
	 * Test method for this class
	 *
	 * @param args the command line arguments
	 */
	public static void main(String [] args) {

		try {
			BVDcrossvalx bvd = new BVDcrossvalx();

			try {
				bvd.setOptions(args);
				Utils.checkForRemainingOptions(args);
			} catch (Exception ex) {
				String result = ex.getMessage() + "\nBVDcrossvalx Options:\n\n";
				Enumeration enu = bvd.listOptions();
				while (enu.hasMoreElements()) {
					Option option = (Option) enu.nextElement();
					result += option.synopsis() + "\n" + option.description() + "\n";
				}
				throw new Exception(result);
			}

			bvd.decompose();
			System.out.println(bvd.toString());
		} catch (Exception ex) {
			System.err.println(ex.getMessage());
		}
	}
}


