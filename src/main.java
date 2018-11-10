import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.ClassicCC;
import mulan.classifier.transformation.ClassifierChain;
import mulan.classifier.transformation.EnsembleOfClassifierChains;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import parallelCC.NewCC;
import parallelCC.ParallelCC;
import weka.classifiers.trees.J48;

public class main {

	public static void main(String [] args) {
		/*
		 * READ DATA
		 */
		
		String dataset = "Yeast";
		
		String train = "data/" + dataset + "-train.arff";
		String test = "data/" + dataset + "-test.arff";
		String xml = "data/" + dataset + ".xml";
		
		try {
			MultiLabelInstances trainData = new MultiLabelInstances(train, xml);
			MultiLabelInstances testData = new MultiLabelInstances(test, xml);
			
//			int [] chain = {12, 9, 4, 5, 15, 7, 10, 13, 6, 3, 2, 0, 14, 8, 11, 16, 1, 18, 17};
			int [] chain = {0, 13, 1, 12, 2, 11, 3, 10, 4, 9, 5, 8, 6, 7};
//			int [] chain = {5, 1, 3, 0, 2, 4};
			
			Evaluator eval = new Evaluator();
			Evaluation results;
			
			long init_time = System.currentTimeMillis();

			BinaryRelevance br = new BinaryRelevance(new J48());
			br.build(trainData.clone());			
			results = eval.evaluate(br, testData.clone(), trainData.clone());
			long end_time = System.currentTimeMillis();			
			System.out.println("BR:  " + results.toCSV());
			System.out.println("Time: " + (end_time - init_time) + " ms.");
			
			init_time = System.currentTimeMillis();
			ClassicCC classicCC = new ClassicCC(new J48(), chain);
			classicCC.build(trainData.clone());
			results = eval.evaluate(classicCC, testData.clone(), trainData.clone());
			end_time = System.currentTimeMillis();			
			System.out.println("classicCC:  " + results.toCSV());
			System.out.println("Time: " + (end_time - init_time) + " ms.");
			
			init_time = System.currentTimeMillis();
			NewCC newCC = new NewCC(new J48(), chain);
			newCC.build(trainData.clone());			
			results = eval.evaluate(newCC, testData.clone(), trainData.clone());
			end_time = System.currentTimeMillis();			
			System.out.println("newCC: " + results.toCSV());
			System.out.println("Time: " + (end_time - init_time) + " ms.");
			
			init_time = System.currentTimeMillis();
			ParallelCC pcc = new ParallelCC(new J48(), chain);
			pcc.build(trainData.clone());			
			results = eval.evaluate(pcc, testData.clone(), trainData.clone());
			end_time = System.currentTimeMillis();			
			System.out.println("pCC: " + results.toCSV());
			System.out.println("Time: " + (end_time - init_time) + " ms.");
			
			
			
						
			/*
			 * TRAINING
			 */
			
			//FOR EACH LABEL
			//	Filter only corresponding label
			//	* If there are predictions, include all as input features
			//	Train BR for corresponding label
			//	* Save predictions
			
			/*
			 * TESTING
			 */
			//FOR EACH LABEL
			//	* See if all needed predictions have finished
			//	* Add predictions as input features
			//	Predict given label
			//	* Store predictions
			
			
			
			
		} catch (InvalidDataFormatException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
