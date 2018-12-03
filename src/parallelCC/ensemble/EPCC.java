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
package parallelCC.ensemble;

import java.util.Arrays;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.ECC;
import mulan.data.MultiLabelInstances;
import parallelCC.ParallelCC;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 * Implementation of the Ensemble of Parallel Classifier Chain (EPCC) algorithm.
 * It is an ensemble (like ECC) but uses PCC as members of the ensemble.
 * Therefore, although each member of the ensemble is built in sequential manner, the single-label classifiers
 * 	of each member are built in parallel.
 * For more information, see <em>https://github.com/i02momuj/ParallelCC</em>
 *
 * @author Jose M. Moyano
 * @version 2018.12.03
 */
public class EPCC extends ECC {

    /**
	 * 
	 */
	private static final long serialVersionUID = 563397438251187303L;
	    
    /**
     * An array of ClassifierChain models
     */
    protected ParallelCC[] ensemble;
    
    /**
     * Number of threads to execute PCC in parallel
     * By default, it obtains all available processors
     */
    int numThreads = Runtime.getRuntime().availableProcessors();

    /**
     * Set the number of threads
     * @param numThreads Number of threads
     */
    public void setNumThreads(int numThreads) {
    	this.numThreads = numThreads;
    }   

    /**
     * Default constructor
     */
    public EPCC() {
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
    public EPCC(Classifier classifier, int aNumOfModels,
            boolean doUseConfidences, boolean doUseSamplingWithReplacement) {
        super(classifier, aNumOfModels, doUseConfidences, doUseSamplingWithReplacement);
        ensemble = new ParallelCC[aNumOfModels];
    }
    
    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
    	long time_init = System.currentTimeMillis();
    	
        Instances dataSet = new Instances(trainingSet.getDataSet());

        for (int i = 0; i < numOfModels; i++) {
            debug("ECC Building Model:" + (i + 1) + "/" + numOfModels);
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
            debug(Arrays.toString(chain));

            //Build PCC
            ensemble[i] = new ParallelCC(baseClassifier, chain);
            ensemble[i].setNumThreads(numThreads);
            ensemble[i].build(train);
        }

        timeBuild = System.currentTimeMillis() - time_init;
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception,
            InvalidDataException {

        int[] sumVotes = new int[numLabels];
        double[] sumConf = new double[numLabels];

        Arrays.fill(sumVotes, 0);
        Arrays.fill(sumConf, 0);

        for (int i = 0; i < numOfModels; i++) {
            MultiLabelOutput ensembleMLO = ensemble[i].makePrediction(instance);
            boolean[] bip = ensembleMLO.getBipartition();
            double[] conf = ensembleMLO.getConfidences();

            for (int j = 0; j < numLabels; j++) {
                sumVotes[j] += bip[j] == true ? 1 : 0;
                sumConf[j] += conf[j];
            }
        }

        double[] confidence = new double[numLabels];
        for (int j = 0; j < numLabels; j++) {
            if (useConfidences) {
                confidence[j] = sumConf[j] / numOfModels;
            } else {
                confidence[j] = sumVotes[j] / (double) numOfModels;
            }
        }

        MultiLabelOutput mlo = new MultiLabelOutput(confidence, 0.5);
        return mlo;
    }
}