import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import mulan.classifier.transformation.BR;
import mulan.classifier.transformation.ECC;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.AveragePrecision;
import mulan.evaluation.measure.Coverage;
import mulan.evaluation.measure.ErrorSetSize;
import mulan.evaluation.measure.ExampleBasedAccuracy;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.ExampleBasedPrecision;
import mulan.evaluation.measure.ExampleBasedRecall;
import mulan.evaluation.measure.ExampleBasedSpecificity;
import mulan.evaluation.measure.GeometricMeanAveragePrecision;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.IsError;
import mulan.evaluation.measure.MacroFMeasure;
import mulan.evaluation.measure.MacroPrecision;
import mulan.evaluation.measure.MacroRecall;
import mulan.evaluation.measure.MacroSpecificity;
import mulan.evaluation.measure.MeanAveragePrecision;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.MicroFMeasure;
import mulan.evaluation.measure.MicroPrecision;
import mulan.evaluation.measure.MicroRecall;
import mulan.evaluation.measure.MicroSpecificity;
import mulan.evaluation.measure.OneError;
import mulan.evaluation.measure.RankingLoss;
import mulan.evaluation.measure.SubsetAccuracy;
import parallelCC.NewCC;
import parallelCC.ParallelCC;
import parallelCC.ParallelECC;
import weka.classifiers.trees.J48;
import weka.core.Utils;

public class main {
	
	/**
	 * Arguments to main method are:
	 * 	1) -d Path file including paths to datasets
	 *  2) -t Number of threads
	 *  3) -s Number of different seeds for random numbers
	 *  4) -o Report filename
	 *  5) -a Algorithm to execute (BR, CC, PCC)
	 *  
	 * @param args
	 */
	public static void main(String [] args) {

		PrintWriter pw = null;			
		ArrayList<String> trainFilenames = new ArrayList<String>();
		ArrayList<String> testFilenames = new ArrayList<String>();
		ArrayList<String> xmlFilenames = new ArrayList<String>();
		
		try {
			String dataFilenames = Utils.getOption("d", args);
			
			int numThreads = Integer.parseInt(Utils.getOption("t", args));
			if(numThreads < 1) {
				numThreads = Runtime.getRuntime().availableProcessors();
			}
			
			String reportFilename = Utils.getOption("o", args);
			int numSeeds = Integer.parseInt(Utils.getOption("s", args));
			String algorithm = Utils.getOption("a", args);
			
			//Read filenames
			BufferedReader b = null;		
			b = new BufferedReader(new FileReader(new File(dataFilenames)));
			String readLine = "";
			String [] words;			
			while ((readLine = b.readLine()) != null) {
				words = readLine.split(" ");
                trainFilenames.add(words[0]);
                testFilenames.add(words[1]);
                xmlFilenames.add(words[2]);
            }
			b.close();
			int nFiles = trainFilenames.size();
			
			MultiLabelInstances trainData = null;
			MultiLabelInstances testData = null;
			List<Measure> measures = null;
			Evaluator eval = new Evaluator();
			Evaluation results;
			long init_time, end_time;
			
			pw = new PrintWriter(new FileWriter(reportFilename, true));
			
			//For each dataset
			for(int f=0; f<nFiles; f++) {
				/*
				 * Read the dataset
				 */
				trainData = new MultiLabelInstances(trainFilenames.get(f), xmlFilenames.get(f));
				testData = new MultiLabelInstances(testFilenames.get(f), xmlFilenames.get(f));
				
				measures = prepareMeasuresClassification(trainData);			
				
				main.printHeader(pw, measures, trainData);
				
				if(algorithm.equalsIgnoreCase("BR")) {
					for(int i=0; i<numSeeds; i++) {
						init_time = System.currentTimeMillis();
						BR br = new BR(new J48());
						br.build(trainData);
						results = eval.evaluate(br, testData, measures);
						end_time = System.currentTimeMillis();
						main.printResults(pw, results, trainFilenames.get(f), "BR", (end_time - init_time), br.getBuildingTime());
					}
				}
				else if(algorithm.equalsIgnoreCase("CC")) {
					for(int i=0; i<numSeeds; i++) {
						init_time = System.currentTimeMillis();
						NewCC cc = new NewCC(new J48());
						cc.setSeed((i+1)*10);
						cc.build(trainData);
						results = eval.evaluate(cc, testData, measures);
						end_time = System.currentTimeMillis();
						main.printResults(pw, results, trainFilenames.get(f), "CC", (end_time - init_time), cc.getBuildingTime());
					}
				}
				else if(algorithm.equalsIgnoreCase("PCC")) {
					for(int i=0; i<numSeeds; i++) {
						init_time = System.currentTimeMillis();
						ParallelCC pcc = new ParallelCC(new J48());
						pcc.setNumThreads(numThreads);
						pcc.setSeed((i+1)*10);
						pcc.build(trainData);
						results = eval.evaluate(pcc, testData, measures);
						end_time = System.currentTimeMillis();
						main.printResults(pw, results, trainFilenames.get(f), "pCC_" + numThreads, (end_time - init_time), pcc.getBuildingTime());
					}
				}
				else if(algorithm.equalsIgnoreCase("ECC")) {
					for(int i=0; i<numSeeds; i++) {
						init_time = System.currentTimeMillis();
						ECC ecc = new ECC();
						ecc.setSeed((i+1)*10);
						ecc.build(trainData);
						results = eval.evaluate(ecc, testData, measures);
						end_time = System.currentTimeMillis();
						main.printResults(pw, results, trainFilenames.get(f), "ECC", (end_time - init_time), ecc.getBuildingTime());
					}
				}
				else if(algorithm.equalsIgnoreCase("PECC")) {
					for(int i=0; i<numSeeds; i++) {
						init_time = System.currentTimeMillis();
						ParallelECC pecc = new ParallelECC();
						pecc.setNumThreads(numThreads);
						pecc.setSeed((i+1)*10);
						pecc.build(trainData);
						results = eval.evaluate(pecc, testData, measures);
						end_time = System.currentTimeMillis();
						main.printResults(pw, results, trainFilenames.get(f), "PECC", (end_time - init_time), pecc.getBuildingTime());
					}
				}
				else {
					System.out.println("Algorithm not defined.");
				}
			}		
			
		} catch (InvalidDataFormatException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
		finally{
    		if(pw != null)
    		{
    			pw.close();
    		}
    	}   
	}
	
	public static void printHeader(PrintWriter pw, List<Measure> measures, MultiLabelInstances mlData) throws Exception{
		//Print header
	    pw.print("Dataset" + ";");
        for(Measure m : measures)
        {
        	pw.print(m.getName() + ";");
        }
        pw.print("Execution time (ms)" + ";" + "Building time (ms)");
        pw.println();
	}
	
	public static void printResults(PrintWriter pw, Evaluation results, String dataname, String algorithm, long runtime, long buildTime) throws Exception {
		String [] p = dataname.split("\\/");
		String datasetName = p[p.length-1].split("\\.")[0];                   
       
		pw.print(algorithm + "_" + datasetName + ";");
    	for(Measure m : results.getMeasures())
        {
        	pw.print(m.getValue() + ";");
        }
        pw.print(runtime + ";" + buildTime);
        pw.println();  
	}
	
	protected static List<Measure> prepareMeasuresClassification(MultiLabelInstances mlTrainData) {
        List<Measure> measures = new ArrayList<Measure>();

        int numOfLabels = mlTrainData.getNumLabels();
        
        // add example-based measures
        measures.add(new HammingLoss());
        measures.add(new SubsetAccuracy());
        measures.add(new ExampleBasedPrecision());
        measures.add(new ExampleBasedRecall());
        measures.add(new ExampleBasedFMeasure());
        measures.add(new ExampleBasedAccuracy());
        measures.add(new ExampleBasedSpecificity());
        
        // add label-based measures
        measures.add(new MicroPrecision(numOfLabels));
        measures.add(new MicroRecall(numOfLabels));
        measures.add(new MicroFMeasure(numOfLabels));
        measures.add(new MicroSpecificity(numOfLabels));
        measures.add(new MacroPrecision(numOfLabels));
        measures.add(new MacroRecall(numOfLabels));
        measures.add(new MacroFMeasure(numOfLabels));
        measures.add(new MacroSpecificity(numOfLabels));
        
        // add ranking based measures
        measures.add(new AveragePrecision());
        measures.add(new Coverage());
        measures.add(new OneError());
        measures.add(new IsError());
        measures.add(new ErrorSetSize());
        measures.add(new RankingLoss());
        
        // add confidence measures if applicable
        measures.add(new MeanAveragePrecision(numOfLabels));
        measures.add(new GeometricMeanAveragePrecision(numOfLabels));
//        measures.add(new MicroAUC(numOfLabels));
//        measures.add(new MacroAUC(numOfLabels));

        return measures;
    }
}
