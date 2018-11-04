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
package mulan.classifier.transformation;

import java.util.ArrayList;
import java.util.Arrays;

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
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * <p>Implementation of the Classifier Chain (ParallelCC) algorithm.</p> <p>For more
 * information, see <em>Read, J.; Pfahringer, B.; Holmes, G.; Frank, E.
 * (2011) Classifier Chains for Multi-label Classification. Machine Learning.
 * 85(3):335-359.</em></p>
 *
 * @author Eleftherios Spyromitros-Xioufis
 * @author Konstantinos Sechidis
 * @author Grigorios Tsoumakas
 * @version 2012.02.27
 */
public class ParallelCC4 extends TransformationBasedMultiLabelLearner {

    /**
     * The new chain ordering of the label indices
     */
    private int[] chain;
    /**
     * The ensemble of binary relevance models. These are Weka
     * FilteredClassifier objects, where the filter corresponds to removing all
     * label apart from the one that serves as a target for the corresponding
     * model.
     */
    protected FilteredClassifier[] ensemble;
    
    /**
     * 
     */
    protected Instances predictedLabels;
    byte [] trained;

    /**
     * Creates a new instance using J48 as the underlying classifier
     */
    public ParallelCC4() {
        super(new J48());
    }

    /**
     * Creates a new instance
     *
     * @param classifier the base-level classification algorithm that will be
     * used for training each of the binary models
     * @param aChain contains the order of the label indexes [0..numLabels-1] 
     */
    public ParallelCC4(Classifier classifier, int[] aChain) {
        super(classifier);
        chain = aChain; 
    }

    /**
     * Creates a new instance
     *
     * @param classifier the base-level classification algorithm that will be
     * used for training each of the binary models
     */
    public ParallelCC4(Classifier classifier) {
        super(classifier);
    }

    protected void buildInternal(MultiLabelInstances train) throws Exception {
        if (chain == null) {
            chain = new int[numLabels];
            for (int i = 0; i < numLabels; i++) {
                chain[i] = i;
            }
        }
        
        trained = new byte[chain.length];

        Instances trainDataset;
        numLabels = train.getNumLabels();
        ensemble = new FilteredClassifier[numLabels];
        trainDataset = train.getDataSet();
        
        Remove rem = new Remove();
        rem.setAttributeIndicesArray(labelIndices);
        
        rem.setInvertSelection(true);
        rem.setInputFormat(trainDataset);
        predictedLabels = Filter.useFilter(trainDataset, rem);
        
        ArrayList<Integer> toRemoveLabels = new ArrayList<Integer>(Arrays.asList(Arrays.stream(chain).boxed().toArray(Integer[]::new)));
        
        
        for (int i = 0; i < numLabels; i++) {
            ensemble[i] = new FilteredClassifier();
            ensemble[i].setClassifier(AbstractClassifier.makeCopy(baseClassifier));

            // Indices of attributes to remove first removes numLabels attributes
            // the numLabels - 1 attributes and so on.
            // The loop starts from the last attribute.
            int[] indicesToRemove = new int[numLabels - 1 - i];
            int counter2 = 0;
            for (int counter1 = 0; counter1 < numLabels - i - 1; counter1++) {
                indicesToRemove[counter1] = labelIndices[chain[numLabels - 1 - counter2]];
                counter2++;
            }
            
            if(i > 0) {
            	toRemoveLabels.remove(new Integer(chain[i-1]));
            }
            
            System.out.println("--");
            
            int [] indicesToRemove2 = new int[toRemoveLabels.size()-1];
            
            for(int j=1; j<toRemoveLabels.size(); j++) {
            	indicesToRemove2[j-1] = labelIndices[toRemoveLabels.get(j)];
            }
            
            System.out.println(Arrays.toString(indicesToRemove));
            System.out.println(Arrays.toString(indicesToRemove2));
            
            Remove remove = new Remove();
            remove.setAttributeIndicesArray(indicesToRemove2);
            remove.setInputFormat(trainDataset);
            remove.setInvertSelection(false);
            ensemble[i].setFilter(remove);

            trainDataset.setClassIndex(labelIndices[chain[i]]);
            debug("Bulding model " + (i + 1) + "/" + numLabels);

//            System.out.println(trainDataset.get(0));

            ensemble[i].buildClassifier(trainDataset);

            //Pred train instances
            Attribute att = trainDataset.attribute(labelIndices[chain[i]]);

            for(int j=0; j<trainDataset.numInstances(); j++) {
            	boolean[] bip = makePredictionInternal(i, trainDataset.get(j)).getBipartition();
            	predictedLabels.get(j).setValue(chain[i], bip[0]? 1:0);
            }

            trainDataset.replaceAttributeAt(predictedLabels.attribute(chain[i]), labelIndices[chain[i]]);
            trained[chain[i]] = 1;
            System.out.println(Arrays.toString(trained));
        }
//        System.out.println(predictedLabels.get(0));
    }

    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        boolean[] bipartition = new boolean[numLabels];
        double[] confidences = new double[numLabels];

        Instance tempInstance = DataUtils.createInstance(instance, instance.weight(), instance.toDoubleArray());
        for (int counter = 0; counter < numLabels; counter++) {
            double distribution[];
            try {
                distribution = ensemble[counter].distributionForInstance(tempInstance);
            } catch (Exception e) {
                System.out.println(e);
                return null;
            }
            int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

            // Ensure correct predictions both for class values {0,1} and {1,0}
            Attribute classAttribute = ensemble[counter].getFilter().getOutputFormat().classAttribute();
            bipartition[chain[counter]] = (classAttribute.value(maxIndex).equals("1")) ? true : false;

            // The confidence of the label being equal to 1
            confidences[chain[counter]] = distribution[classAttribute.indexOfValue("1")];

            tempInstance.setValue(labelIndices[chain[counter]], maxIndex);

        }

        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
        return mlo;
    }
    
    protected MultiLabelOutput makePredictionInternal(int classifier, Instance instance) throws Exception {
        boolean[] bipartition = new boolean[1];
        double[] confidences = new double[1];

        Instance tempInstance = DataUtils.createInstance(instance, instance.weight(), instance.toDoubleArray());
//        for (int counter = 0; counter < numLabels; counter++) {
           	int counter = classifier;
        	double distribution[];
            try {
                distribution = ensemble[counter].distributionForInstance(tempInstance);
            } catch (Exception e) {
                System.out.println(e);
                return null;
            }
            int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

            // Ensure correct predictions both for class values {0,1} and {1,0}
            Attribute classAttribute = ensemble[counter].getFilter().getOutputFormat().classAttribute();
            bipartition[0] = (classAttribute.value(maxIndex).equals("1")) ? true : false;

            // The confidence of the label being equal to 1
            confidences[0] = distribution[classAttribute.indexOfValue("1")];

            tempInstance.setValue(labelIndices[0], maxIndex);

//        }

        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
        return mlo;
    }
}