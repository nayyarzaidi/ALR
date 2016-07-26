package logDistributionComputation;

import DataStructure.wdAnJEParameters;
import Utils.plTechniques;

import logDistributionComputation.D.A1JELogDistributionComputerD;
import logDistributionComputation.D.A2JELogDistributionComputerD;
import logDistributionComputation.D.A3JELogDistributionComputerD;
import logDistributionComputation.D.A4JELogDistributionComputerD;
import logDistributionComputation.D.A5JELogDistributionComputerD;

import logDistributionComputation.MAP.A1JELogDistributionComputerMAP;
import logDistributionComputation.MAP.A2JELogDistributionComputerMAP;
import logDistributionComputation.MAP.A3JELogDistributionComputerMAP;
import logDistributionComputation.MAP.A4JELogDistributionComputerMAP;
import logDistributionComputation.MAP.A5JELogDistributionComputerMAP;

import logDistributionComputation.MAP2.A1JELogDistributionComputerMAP2;
import logDistributionComputation.MAP2.A2JELogDistributionComputerMAP2;
import logDistributionComputation.MAP2.A3JELogDistributionComputerMAP2;
import logDistributionComputation.MAP2.A4JELogDistributionComputerMAP2;
import logDistributionComputation.MAP2.A5JELogDistributionComputerMAP2;

import logDistributionComputation.W.A1JELogDistributionComputerW;
import logDistributionComputation.W.A2JELogDistributionComputerW;
import logDistributionComputation.W.A3JELogDistributionComputerW;
import logDistributionComputation.W.A4JELogDistributionComputerW;
import logDistributionComputation.W.A5JELogDistributionComputerW;

import logDistributionComputation.W2.A1JELogDistributionComputerW2;
import logDistributionComputation.W2.A2JELogDistributionComputerW2;
import logDistributionComputation.W2.A3JELogDistributionComputerW2;
import logDistributionComputation.W2.A4JELogDistributionComputerW2;
import logDistributionComputation.W2.A5JELogDistributionComputerW2;

import logDistributionComputation.WF.A1JELogDistributionComputerWF;
import logDistributionComputation.WF.A2JELogDistributionComputerWF;
import logDistributionComputation.WF.A3JELogDistributionComputerWF;
import logDistributionComputation.WF.A4JELogDistributionComputerWF;
import logDistributionComputation.WF.A5JELogDistributionComputerWF;

import logDistributionComputation.DF.A1JELogDistributionComputerDF;
import logDistributionComputation.DF.A2JELogDistributionComputerDF;
import logDistributionComputation.DF.A3JELogDistributionComputerDF;
import logDistributionComputation.DF.A4JELogDistributionComputerDF;
import logDistributionComputation.DF.A5JELogDistributionComputerDF;

import weka.core.Instance;


public abstract class LogDistributionComputerAnJE {
	
	public abstract void compute(double[] probs, wdAnJEParameters params, Instance inst);
	
	public static LogDistributionComputerAnJE getDistributionComputer(int n, int scheme) {
		switch (n) {
		case 1: return getComputerA1JE(scheme);
		case 2: return getComputerA2JE(scheme);
		case 3: return getComputerA3JE(scheme);
		case 4: return getComputerA4JE(scheme);
		case 5: return getComputerA5JE(scheme);
		default:
			System.err.println("A"+n+"JE not implemented, choosing A1JE");
			return getComputerA1JE(scheme); 
		}
	}
	
	public static LogDistributionComputerAnJE getComputerA1JE(int scheme) {
		switch (scheme) {
		case plTechniques.MAP: 		return A1JELogDistributionComputerMAP.getComputer();
		case plTechniques.dCCBN: 	return A1JELogDistributionComputerD.getComputer();
		case plTechniques.wCCBN: 	return A1JELogDistributionComputerW.getComputer();
		case plTechniques.dCCBNf: 	return A1JELogDistributionComputerDF.getComputer();
		case plTechniques.wCCBNf: 	return A1JELogDistributionComputerWF.getComputer();
		case plTechniques.wCCBN2: 	return A1JELogDistributionComputerW2.getComputer();
		case plTechniques.MAP2: 	return A1JELogDistributionComputerMAP2.getComputer();
		default: 
			System.err.println("Scheme not implemented, resorting to MAP");
			return A1JELogDistributionComputerMAP.getComputer();
		}
	}
	
	public static LogDistributionComputerAnJE getComputerA2JE(int scheme) {
		switch (scheme) {
		case plTechniques.MAP: 		return A2JELogDistributionComputerMAP.getComputer();
		case plTechniques.dCCBN: 	return A2JELogDistributionComputerD.getComputer();
		case plTechniques.wCCBN: 	return A2JELogDistributionComputerW.getComputer();
		case plTechniques.dCCBNf:	return A2JELogDistributionComputerDF.getComputer();
		case plTechniques.wCCBNf:	return A2JELogDistributionComputerWF.getComputer();
		case plTechniques.wCCBN2: 	return A2JELogDistributionComputerW2.getComputer();
		case plTechniques.MAP2: 	return A2JELogDistributionComputerMAP2.getComputer();
		default: 
			System.err.println("Scheme not implemented, resorting to MAP");
			return A2JELogDistributionComputerMAP.getComputer();
		}
	}
	
	public static LogDistributionComputerAnJE getComputerA3JE(int scheme) {
		switch (scheme) {
		case plTechniques.MAP: 		return A3JELogDistributionComputerMAP.getComputer();
		case plTechniques.dCCBN: 	return A3JELogDistributionComputerD.getComputer();
		case plTechniques.wCCBN: 	return A3JELogDistributionComputerW.getComputer();
		case plTechniques.dCCBNf: 	return A3JELogDistributionComputerDF.getComputer();
		case plTechniques.wCCBNf: 	return A3JELogDistributionComputerWF.getComputer();
		case plTechniques.wCCBN2: 	return A3JELogDistributionComputerW2.getComputer();
		case plTechniques.MAP2: 	return A3JELogDistributionComputerMAP2.getComputer();
		default: 
			System.err.println("Scheme not implemented, resorting to MAP");
			return A3JELogDistributionComputerMAP.getComputer();
		}
	}
	
	public static LogDistributionComputerAnJE getComputerA4JE(int scheme) {
		switch (scheme) {
		case plTechniques.MAP: 		return A4JELogDistributionComputerMAP.getComputer();
		case plTechniques.dCCBN: 	return A4JELogDistributionComputerD.getComputer();
		case plTechniques.wCCBN: 	return A4JELogDistributionComputerW.getComputer();
		case plTechniques.dCCBNf: 	return A4JELogDistributionComputerDF.getComputer();
		case plTechniques.wCCBNf: 	return A4JELogDistributionComputerWF.getComputer();
		case plTechniques.wCCBN2: 	return A4JELogDistributionComputerW2.getComputer();
		case plTechniques.MAP2: 	return A4JELogDistributionComputerMAP2.getComputer();
		default: 
			System.err.println("Scheme not implemented, resorting to MAP");
			return A4JELogDistributionComputerMAP.getComputer();
		}
	}
	
	public static LogDistributionComputerAnJE getComputerA5JE(int scheme) {
		switch (scheme) {
		case plTechniques.MAP: 		return A5JELogDistributionComputerMAP.getComputer();
		case plTechniques.dCCBN: 	return A5JELogDistributionComputerD.getComputer();
		case plTechniques.wCCBN: 	return A5JELogDistributionComputerW.getComputer();
		case plTechniques.dCCBNf: 	return A5JELogDistributionComputerDF.getComputer();
		case plTechniques.wCCBNf: 	return A5JELogDistributionComputerWF.getComputer();
		case plTechniques.wCCBN2: 	return A5JELogDistributionComputerW2.getComputer();
		case plTechniques.MAP2: 	return A5JELogDistributionComputerMAP2.getComputer();
		default: 
			System.err.println("Scheme not implemented, resorting to MAP");
			return A5JELogDistributionComputerMAP.getComputer();
		}
	}
}
