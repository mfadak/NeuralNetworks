import java.io.File;
import java.net.URISyntaxException;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.transfer.TransferFunction;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;

import mfann.*;

public class Test {

	public static void main(String[] args) throws URISyntaxException {
		File file = new File(Program.class.getResource("Data.txt").toURI());
		NeuralNetwork<BackPropagation> ann = new MultiLayerPerceptron(TransferFunctionType.SIGMOID,2,5,1);
		BackPropagation bp = new BackPropagation();
		bp.setLearningRate(0.5);		
		
		DataSet dataset = DataSet.createFromFile(file.getPath(), 2, 1, ",");
		DataSet[] ds = dataset.createTrainingAndTestSubsets(0.7, 0.3);
		DataSet egitim = ds[0];
		DataSet test = ds[1];
		
		NNetwork neuralNetwork = new NNetwork(ann, bp, egitim, test, 1000, 0.000001); 
		neuralNetwork.train();
		System.out.println("Train Error: "+neuralNetwork.getTrainError());
		System.out.println("Test Error: "+neuralNetwork.getTestError());
		neuralNetwork.displayErrorChart();
	}

}
