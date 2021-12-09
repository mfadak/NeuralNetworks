package model;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;

public class Ann {
	private static final File trainingFile = new File(Ann.class.getResource("Train.txt").getPath());
    private static final File testFile = new File(Ann.class.getResource("Test.txt").getPath());
    
    private double[] maximums;
    private double[] minimums;
    
    private DataSet trainingDataset;
    private DataSet testingDataset;
    private int hiddenLayerNourons;
    
    MomentumBackpropagation mbp;
    
    public Ann(int inputs,int outputs,int hiddenLayerNourons, double momentum, double learningRate, double minError, int maxEpoch) throws FileNotFoundException {
    	maximums = new double[inputs];
    	minimums = new double[inputs];
        
    	initArrays(inputs);
    	
    	ReadMaxAndMins(trainingFile, inputs, outputs);
    	ReadMaxAndMins(testFile, inputs, outputs);
    	
    	trainingDataset = readDataSet(trainingFile, inputs, outputs);
    	testingDataset = readDataSet(testFile, inputs, outputs);
    	
    	setupLearningRule(momentum, learningRate, minError, maxEpoch);

        this.hiddenLayerNourons = hiddenLayerNourons;
    }
    public void train() {
    	MultiLayerPerceptron nn = new MultiLayerPerceptron(TransferFunctionType.SIGMOID,trainingDataset.getInputSize(),hiddenLayerNourons,trainingDataset.getOutputSize());        
    	nn.setLearningRule(mbp);
    	nn.learn(trainingDataset);
    	nn.save("trained.nnet");
        System.out.println("Training was completed!");
    }
    private void setupLearningRule(double momentum, double learningRate, double minError, int maxEpoch) {
    	mbp = new MomentumBackpropagation();
    	mbp.setMomentum(momentum);
    	mbp.setLearningRate(learningRate);
    	mbp.setMaxError(minError);
    	mbp.setMaxIterations(maxEpoch);
    }
    
    private void initArrays(int inputs) {
    	for(int i=0;i<inputs;i++){
    		maximums[i]=Double.MIN_VALUE;
    		minimums[i]=Double.MAX_VALUE;
        }
    }
    private void ReadMaxAndMins(File file,int inputs,int outputs) throws FileNotFoundException {
    	Scanner in = new Scanner(file);
        while(in.hasNextDouble())
        {
            for(int i=0;i<inputs;i++){
                double d = in.nextDouble();
                if(d > maximums[i]) maximums[i]=d;
                if(d < minimums[i]) minimums[i]=d;
            }
            // pass outputs
            for(int i=0;i<outputs;i++) {
            	in.nextDouble();
            }
        }        
        in.close();
    }
    public double TrainError() {
    	return mbp.getTotalNetworkError();
    }
    public double test() {
    	NeuralNetwork nn = NeuralNetwork.createFromFile("trained.nnet");
        double totalError=0;
        for(DataSetRow dr : testingDataset){
        	nn.setInput(dr.getInput());
        	nn.calculate();
        	totalError += mse(dr.getDesiredOutput(),nn.getOutput());
        }
        return totalError / testingDataset.size();

    }
    public String uniqueTest(double[] inputs){
        for(int i=0;i<8;i++){
            inputs[i] = minMax(maximums[i], minimums[i], inputs[i]);
        }
        NeuralNetwork nn = NeuralNetwork.createFromFile("trained.nnet");
        nn.setInput(inputs);
        nn.calculate();
        return Output(nn.getOutput());
    }
    private String Output(double[] outputs) {
    	int index=0;
        double max = outputs[0];
        if(outputs[1] > max) { max = outputs[1]; index=1; }
        if(outputs[2] > max) index=2;
        
        if(index == 0) return "Bad";
        else if(index == 1) return "Normal";
        return "Good";

    }
    private double mse(double[] desiredOutputs,double[] outputs) {
    	double totalRowError=0;
        for(int i=0;i<trainingDataset.getOutputSize();i++)
        	totalRowError += Math.pow(desiredOutputs[i] - outputs[i],2);
        return totalRowError/trainingDataset.getOutputSize();
    }
    private DataSet readDataSet(File file,int inputs,int outputs) throws FileNotFoundException {
    	Scanner in = new Scanner(file);
        
        DataSet dataset = new DataSet(inputs,outputs);
        while(in.hasNextDouble())
        {
            double []inputArray = new double[inputs];
            for(int i=0;i<inputs;i++){
                double d = in.nextDouble();
                inputArray[i] = minMax(maximums[i], minimums[i], d);
            }
            
            double []outputArray = new double[outputs];
            for(int i=0;i<outputs;i++){
                double d = in.nextDouble();
                outputArray[i] = d;
            }
           DataSetRow row = new DataSetRow(inputArray, outputArray);
           dataset.addRow(row);
        }
        in.close();
        return dataset;

    }
    private double minMax(double max,double min,double x){
        return (x-min)/(max-min);
    }

}
