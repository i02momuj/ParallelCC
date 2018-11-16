/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
package parallelCC;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import mulan.data.MultiLabelInstances;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 * Parallel implementation of Ensemble of Parallel Classifier Chains (PEPCC).
 * Not only the members of the ensemble are PCCs, but also each member of the ensemble is able to be built in parallel.
 * For mor information, see <em></em>
 *
 * @author Jose M. Moyano
 * @version 2018.11.16
 */
public class PEPCC extends EPCC {

    /**
	 * 
	 */
	private static final long serialVersionUID = 6713525906907893794L;

    /**
     * Default constructor
     */
    public PEPCC() {
        this(new J48(), 10, true, true);
    }

    /**
     * Creates a new object
     *
     * @param classifier the base classifier for each ClassifierChain model
     * @param aNumOfModels the number of models
     * @param doUseConfidences whether to use confidences or not
     * @param doUseSamplingWithReplacement whether to use sampling with replacement or not 
     */
    public PEPCC(Classifier classifier, int aNumOfModels,
            boolean doUseConfidences, boolean doUseSamplingWithReplacement) {
        super(classifier, aNumOfModels, doUseConfidences, doUseSamplingWithReplacement);
    }

    
    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
    	long time_init = System.currentTimeMillis();
    	
        Instances dataSet = new Instances(trainingSet.getDataSet());

        //Set number of threads
        ExecutorService executorService = Executors.newFixedThreadPool(numThreads);
        
        //Build each member in parallel
        for (int i = 0; i < numOfModels; i++) {
            executorService.execute(new BuildEnsembleParallel(numOfModels, dataSet, rand, useSamplingWithReplacement, 
            		BagSizePercent,  samplingPercentage, numLabels, ensemble, trainingSet, baseClassifier, i, 
            		numThreads));
        }
        
        executorService.shutdown();
        
        try {
			//Wait until all threads finish
			executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch (Exception e) {
			e.printStackTrace();
		}

        timeBuild = System.currentTimeMillis() - time_init;
    }
    
    /**
     * Class that extends Thread, for code that is executed in parallel
     * 
     * @author Jose M. Moyano
     */
    public static class BuildEnsembleParallel extends Thread {

    	int numOfModels;
    	
    	Instances dataSet;
    	
    	Random rand;
    	
    	boolean useSamplingWithReplacement;
    	
    	int BagSizePercent;
    	
    	double samplingPercentage;
    	
    	int numLabels;
    	
    	protected ParallelCC[] ensemble;
    	
    	MultiLabelInstances trainingSet;
    	
    	Classifier baseClassifier;
    	
    	int i;
    	
    	int numThreads;
    	
		/**
		 * Constructor
		 */
		BuildEnsembleParallel(int numOfModels, Instances dataSet, Random rand, boolean useSamplingWithReplacement, 
				int BagSizePercent, double samplingPercentage, int numLabels, ParallelCC[] ensemble, 
				MultiLabelInstances trainingSet, Classifier baseClassifier, int i, int numThreads){
			this.numOfModels = numOfModels;
			this.dataSet = dataSet;
			this.rand = rand;
			this.useSamplingWithReplacement = useSamplingWithReplacement;
			this.BagSizePercent = BagSizePercent;
			this.samplingPercentage = samplingPercentage;
			this.numLabels = numLabels;
			this.ensemble = ensemble;
			this.trainingSet = trainingSet;
			this.baseClassifier = baseClassifier;
			this.i = i;
			this.numThreads = numThreads;
		}
		
		/**
		 * Override run method for parallel execution.
		 * It is in charge of building each different CC in parallel.
		 * It has not critical code.
		 */
		public void run() {
			try {
	            Instances sampledDataSet;
	            dataSet.randomize(rand);
	            if (useSamplingWithReplacement) {
	                int bagSize = dataSet.numInstances() * BagSizePercent / 100;
	                // create the in-bag dataset
	                sampledDataSet = dataSet.resampleWithWeights(rand);
	                if (bagSize < dataSet.numInstances()) {
	                    sampledDataSet = new Instances(sampledDataSet, 0, bagSize);
	                }
	            } else {
	                RemovePercentage rmvp = new RemovePercentage();
	                rmvp.setInvertSelection(true);
	                rmvp.setPercentage(samplingPercentage);
	                rmvp.setInputFormat(dataSet);
	                sampledDataSet = Filter.useFilter(dataSet, rmvp);
	            }
	            MultiLabelInstances train = new MultiLabelInstances(sampledDataSet, trainingSet.getLabelsMetaData());

	            int[] chain = new int[numLabels];
	            for (int j = 0; j < numLabels; j++) {
	                chain[j] = j;
	            }
	            for (int j = 0; j < chain.length; j++) {
	                int randomPosition = rand.nextInt(chain.length);
	                int temp = chain[j];
	                chain[j] = chain[randomPosition];
	                chain[randomPosition] = temp;
	            }

	            //Further, each member of the ensemble is a PCC -> built in parallel
	            ensemble[i] = new ParallelCC(baseClassifier, chain);
	            ensemble[i].setNumThreads(numThreads);
	            ensemble[i].build(train);
			}catch(Exception e) {
			e.printStackTrace();	
			}
		}
	}
}