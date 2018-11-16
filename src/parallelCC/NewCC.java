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

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.ClassicCC;
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
 * Implementation of the Classifier Chain (CC) algorithm but using different implementation than the one of Mulan. 
 * It is able to build CC by using predictions on training; and not only ground truth (as Mulan).
 * For mor information, see <em></em>
 *
 * @author Jose M. Moyano
 * @version 2018.11.16
 */
public class NewCC extends ClassicCC {
  
    /**
	 * 
	 */
	private static final long serialVersionUID = 7419132463415277030L;

	/**
     * Indicates if a given label has been trained yet or not
     */
    byte [] trained;
    
    /**
     * Indicates if predictions of labels are used in training; if not, ground truth is used
     */
    boolean usePredictions = true;

    /**
     * Creates a new instance using J48 as the underlying classifier
     */
    public NewCC() {
        super(new J48());
    }

    /**
     * Creates a new instance given underlying classifier and chain
     * 
     * @param classifier Single-label classifier
     * @param aChain Chain of labels
     */
    public NewCC(Classifier classifier, int[] aChain) {
        super(classifier, aChain);
    }

    /**
     * Creates a new instance given underlying classifier
     * 
     * @param classifier Single-label classifier
     */
    public NewCC(Classifier classifier) {
        super(classifier);
    }
    
    /**
     * Set if predictions of labels are used in training phase instead of ground truth
     * 
     * @param usePredictions Indicates if predictions of labels are used in training phase.
     */
    public void setUsePredictions(boolean usePredictions) {
    	this.usePredictions = usePredictions;
    }

    /**
     * Build CC classifier given a multi-label dataset
     * 
     * This method changes with respect to the original implementation.
     * It takes the basis to be able to make it parallelizable
     */
    protected void buildInternal(MultiLabelInstances train) throws Exception {
        long time_init = System.currentTimeMillis();
    	
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

        //Train classifier for each label in the chain
        for (int i = 0; i < numLabels; i++) {        	
        	trainDataset.setClassIndex(labelIndices[chain[i]]);
        	
        	//List that store the labels to remove in each case
        	ArrayList<Integer> toRemoveLabels = new ArrayList<Integer>();//new ArrayList<Integer>(Arrays.asList(Arrays.stream(chain).boxed().toArray(Integer[]::new)));

            ensemble[i] = new FilteredClassifier();
            ensemble[i].setClassifier(AbstractClassifier.makeCopy(baseClassifier));

            //Generate array with indices of labels to remove
            //We remove the labels that have been not yet trained
            //	i.e., these previous labels in the chain
            for(int j=0; j<numLabels; j++) {
            	if((j != chain[i]) && (trained[j] == 0)) {
            		toRemoveLabels.add(labelIndices[j]);
            	}
            }
            int [] indicesToRemove = toRemoveLabels.stream().mapToInt(Integer::intValue).toArray();
            
            //Remove labels
            Remove remove = new Remove();
            remove.setAttributeIndicesArray(indicesToRemove);
            remove.setInputFormat(trainDataset);
            remove.setInvertSelection(false);
            ensemble[i].setFilter(remove);
            
            //Build model
            trainDataset.setClassIndex(labelIndices[chain[i]]);
            debug("Bulding model " + (i + 1) + "/" + numLabels);
            ensemble[i].buildClassifier(trainDataset);

            //If predictions of labels are used in training ->
            //	-> Predict i-th label for all training instances to use in following classifiers
            if(usePredictions) {
            	for(int j=0; j<trainDataset.numInstances(); j++) {
                	boolean[] bip = makePredictionInternal(i, trainDataset.get(j)).getBipartition();
                	trainDataset.get(j).setValue(labelIndices[chain[i]], bip[0]? 1:0);
                }
            }
            trained[chain[i]] = 1;
        }
        
        timeBuild = System.currentTimeMillis() - time_init;
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