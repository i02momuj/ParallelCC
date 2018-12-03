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

import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import mulan.classifier.MultiLabelOutput;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Implementation of the Parallel Classifier Chain (PCC) algorithm. 
 * It is able to build CC by using predictions on training; and not only ground truth (as Mulan).
 * Many binary classifiers are built in parallel, using predictions of labels that have been previously built.
 * For more information, see <em>https://github.com/i02momuj/ParallelCC</em>
 *
 * @author Jose M. Moyano
 * @version 2018.12.03
 */
public class ParallelCC extends NewCC {	
    
    /**
	 * 
	 */
	private static final long serialVersionUID = 3738617818728999467L;

	/**
     * Number of threads to execute PCC in parallel
     * By default, it obtains all available processors
     */
    int numThreads = Runtime.getRuntime().availableProcessors();
    
    /**
     * Variable to lock critical code
     */
    Lock lock = new ReentrantLock();

    /**
     * Creates a new instance using J48 as the underlying classifier
     */
    public ParallelCC() {
        super(new J48());
    }

    /**
     * Creates a new instance given underlying classifier and chain
     * 
     * @param classifier Single-label classifier
     * @param aChain Chain of labels
     */
    public ParallelCC(Classifier classifier, int[] aChain) {
        super(classifier, aChain);
    }

    /**
     * Creates a new instance given underlying classifier
     * 
     * @param classifier Single-label classifier
     */
    public ParallelCC(Classifier classifier) {
        super(classifier);
    }
    
    public void setNumThreads(int numThreads) {
    	this.numThreads = numThreads;
    }    

    protected void buildInternal(MultiLabelInstances train) throws Exception {
        long time_init = System.currentTimeMillis();
    	
    	//Create chain if it does not exists
    	//Create RANDOM chain if it does not exists
    	if (chain == null) {
            chain = randomChain(seed);
        }
        
    	//At the beginning, all bytes from 'trained' are zeros
        trained = new byte[chain.length];

        //Get training dataset
        Instances trainDataset;
        numLabels = train.getNumLabels();
        ensemble = new FilteredClassifier[numLabels];
        trainDataset = train.getDataSet();
        
        //Set number of threads
        ExecutorService executorService = Executors.newFixedThreadPool(numThreads);
        
        //Loop for building classifier for each label (in parallel)
        for (int i = 0; i < numLabels; i++) {        	
        	executorService.execute(new BuildClassifierParallel(i, trainDataset, labelIndices, chain,
    				ensemble, baseClassifier, trained, usePredictions, lock));
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
    public static class BuildClassifierParallel extends Thread {
		
		/**
		 * Index of label to build the classifier
		 */
		int labelIndex;
		
		/**
		 * Training dataset
		 */
		Instances trainDataset;
		
		/**
		 * Indices of labels in the dataset
		 */
		int [] labelIndices;
		
		/**
		 * Chain ordering
		 */
		int [] chain;
		
		/**
		 * Set of filtered classifiers
		 */
		FilteredClassifier[] ensemble;
		
		/**
		 * Variable to lock critical code
		 */
	    private Lock lock;
	    
	    /**
	     * Number of labels in the dataset
	     */
		int numLabels;
		
		/**
		 * Single-label classifier to use
		 */
		Classifier baseClassifier;
		
		/**
		 * Indicates which labels have been previously trained
		 */
		byte [] trained;
		
		/**
		 * Indicates if predictions are used in training phase, instead of ground truth
		 */
		boolean usePredictions;
		
		/**
		 * Constructor
		 * 
		 * @param labelIndex
		 * @param trainDataset
		 * @param labelIndices
		 * @param chain
		 * @param ensemble
		 * @param baseClassifier
		 * @param trained
		 * @param usePredictions
		 */
		BuildClassifierParallel(int labelIndex, Instances trainDataset, int [] labelIndices, int [] chain,
				FilteredClassifier[] ensemble, Classifier baseClassifier, byte [] trained, boolean usePredictions, 
				Lock lock){
			this.labelIndex = labelIndex;
			this.trainDataset = trainDataset;
			this.labelIndices = labelIndices;
			this.chain = chain;
			this.ensemble = ensemble;
			this.numLabels = labelIndices.length;
			this.baseClassifier = baseClassifier;
			this.trained = trained;
			this.usePredictions = usePredictions;
			this.lock = lock;
		}
		
		/**
		 * Override run method for parallel execution.
		 * It is in charge of building each binary classifier of CC.
		 * Critical code is locked (only one thread simoultaneously)
		 */
		public void run() {
			try {
				//Create copy of data for each classifier
	        	Instances iData = new Instances(trainDataset);
	        	iData.setClassIndex(labelIndices[chain[labelIndex]]);
	        	
	        	//List that store the labels to remove in each case
	        	ArrayList<Integer> toRemoveLabels = new ArrayList<Integer>();

	            ensemble[labelIndex] = new FilteredClassifier();
	            ensemble[labelIndex].setClassifier(AbstractClassifier.makeCopy(baseClassifier));
	            
	            //Lock critical code
	            //Check which labels have been previously trained
	            //Keep labels that have been previously trained and current label; remove the rest
	            lock.lock();
	            for(int j=0; j<numLabels; j++) {
	            	if((j != chain[labelIndex]) && (trained[j] == 0)) {
	            		toRemoveLabels.add(labelIndices[j]);
	            	}
	            }
	            lock.unlock();
	            int [] indicesToRemove = toRemoveLabels.stream().mapToInt(Integer::intValue).toArray();
	            
	            //Remove labels
	            Remove remove = new Remove();
	            remove.setAttributeIndicesArray(indicesToRemove);
	            remove.setInputFormat(iData);
	            remove.setInvertSelection(false);
	            ensemble[labelIndex].setFilter(remove);
	            
	            //Build model
	            iData.setClassIndex(labelIndices[chain[labelIndex]]);
	            ensemble[labelIndex].buildClassifier(iData);

	            //Predict over training instances
	            if(usePredictions) {
	            	//Bipartition for each instance
	            	boolean[] bip = new boolean[iData.numInstances()];
	            	
	            	//Get bipartition for each training instance
		        	for(int j=0; j<iData.numInstances(); j++) {
		            	bip[j] = makePredictionInternal(labelIndex, iData.get(j)).getBipartition()[0];
		        	}
		        	
		        	//Set prediction values in training dataset
		        	for(int j=0; j<iData.numInstances(); j++) {
		                trainDataset.get(j).setValue(labelIndices[chain[labelIndex]], bip[j]? 1:0);
		        	}
		        	
		        	//Comments about the latest for loop 
		        	//We don't mind if we are adding predictions to the data and another thread is copying the dataset
		        	// because this label then is going to be removed
		        	//The important thing is to lock when defining that the given label has been trained
		        	
		        	//Lock critical code
		        	lock.lock();
			        trained[chain[labelIndex]] = 1;
			        lock.unlock();
		        }

			}catch(Exception e) {
			e.printStackTrace();	
			}
		}
		
		/**
	     * Make prediction for a given i-th classifier in the chain and a given instance
	     * 
	     * @param classifierIndex Index of the label to predict
	     * @param instance Instance to predict the label
	     * @return Output predicted by i-th classifier for given instance
	     * @throws Exception
	     */
	    protected MultiLabelOutput makePredictionInternal(int classifierIndex, Instance instance) throws Exception {
	        boolean[] bipartition = new boolean[1];
	        double[] confidences = new double[1];

	        Instance tempInstance = DataUtils.createInstance(instance, instance.weight(), instance.toDoubleArray());

	        double distribution[];
	        try {
	        	distribution = ensemble[classifierIndex].distributionForInstance(tempInstance);
	        } catch (Exception e) {
	        	System.out.println(e);
	            return null;
	        }
	        int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

	        // Ensure correct predictions both for class values {0,1} and {1,0}
	        Attribute classAttribute = ensemble[classifierIndex].getFilter().getOutputFormat().classAttribute();
	        bipartition[0] = (classAttribute.value(maxIndex).equals("1")) ? true : false;

	        // The confidence of the label being equal to 1
	        confidences[0] = distribution[classAttribute.indexOfValue("1")];

	        tempInstance.setValue(labelIndices[0], maxIndex);

	        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
	        return mlo;
	    }
	}
    
}