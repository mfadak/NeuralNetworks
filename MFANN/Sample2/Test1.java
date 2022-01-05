import java.io.File;
import java.net.URISyntaxException;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;

import mfann.*;

public class Program {

	public static void main(String[] args) throws URISyntaxException {
		File file = new File(Program.class.getResource("Data.txt").toURI());
		NeuralNetwork<BackPropagation> ann = new MultiLayerPerceptron(TransferFunctionType.SIGMOID,2,5,1);
		MomentumBackpropagation mbp = new MomentumBackpropagation();
		mbp.setLearningRate(0.5);
		mbp.setMomentum(0.8);
				
		DataSet dataset = DataSet.createFromFile(file.getPath(), 2, 1, ",");
		KFoldCrossValidation kfold = new KFoldCrossValidation(4, dataset, ann, mbp, 1000, 0.000001);
		kfold.train();
		
		System.out.println("Train Error: "+kfold.getAverageTrainError());
		System.out.println("Test Error: "+kfold.getAverageTestError());
		
		kfold.displayTrainErrorsChart();
		kfold.displayTestErrorsChart();
	}
}
