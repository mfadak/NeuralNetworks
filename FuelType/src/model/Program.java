package model;

import java.io.IOException;
import java.util.Scanner;

public class Program {

	public static void main(String[] args) throws IOException {
		Scanner in = new Scanner(System.in);
        int hiddenLayerNourons;
        double momentum,learningRate,minError;
        int maxEpoch,menu=0;
        Ann ann=null;
        do{    
            System.out.println("1. Train Network & Test");
            System.out.println("2. Unique Data Test");
            System.out.println("3. Exit");
            System.out.print("=>");
            menu = in.nextInt();
            switch(menu){
                case 1:
                    System.out.print("Number of neurons in hidden layer:");
                    hiddenLayerNourons = in.nextInt();
                    System.out.print("Momentum:");
                    momentum = in.nextDouble();
                    System.out.print("Learning Rate:");
                    learningRate = in.nextDouble();
                    System.out.print("Min Error:");
                    minError = in.nextDouble();
                    System.out.print("Max Epoch:");
                    maxEpoch = in.nextInt();
                    ann = new Ann(8, 3, hiddenLayerNourons, momentum, learningRate, minError, maxEpoch);
                    ann.train();
                    System.out.println("Train Error Rate: "+ann.TrainError());
                    System.out.println("Test Error Rate: "+ann.test());
                    break;
                case 2:
                    if(ann == null){
                        System.out.println("First network should be trained!");
                        System.in.read();
                        break;
                    }
                    double []inputs = new double[8];
                    System.out.print("Cylinder:");
                    inputs[0] = in.nextDouble();
                    System.out.print("Friction:");
                    inputs[1] = in.nextDouble();
                    System.out.print("Horse power:");
                    inputs[2] = in.nextDouble();
                    System.out.print("Weight:");
                    inputs[3] = in.nextDouble();
                    System.out.print("Acceleration:");
                    inputs[4] = in.nextDouble();
                    System.out.print("Model:");
                    inputs[5] = in.nextDouble();
                    System.out.print("Region:");
                    String region = in.next();
                    double[] reg = regionToArray(region);
                    inputs[6] = reg[0];
                    inputs[7] = reg[1];
                    System.out.println("Fuel Type: "+ann.uniqueTest(inputs)); 
                    System.in.read();
                    break;
            }                
        }while(menu != 3);

	}
	public static double[] regionToArray(String region) {
		switch(region) {
		case "America":
			return new double[] {0,0};
		case "Asia":
			return new double[] {0,1};
		case "Europe":
			return new double[] {1,0};
			default:
				throw new IllegalArgumentException();
		}
	}

}
