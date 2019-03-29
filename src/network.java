import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.swing.SwingUtilities;
import javax.swing.WindowConstants;

public class network {

    public static void main(String[] args){
        //bitmap inputs
        double[][] X = {
                            {0,0,1,0,0,0,1,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0},
                            {0,1,1,1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,1,1,1},
                            {0,1,1,1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,1,1,1,0},
                            {0,0,0,1,0,0,0,1,1,0,0,0,1,1,0,0,1,0,1,0,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,0,0,0,1,0,0,0,0,1,0},
                            {1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,1,1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,1,1,1,0},
                            {0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0},
                            {1,1,1,1,1,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0},
                            {0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0},
                            {0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,1,1,1,0},
                            {0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0},
                        };
        //Expected outputs
        double[][] Y = {
                            {1,0,0,0,0,0,0,0,0,0},
                            {0,1,0,0,0,0,0,0,0,0},
                            {0,0,1,0,0,0,0,0,0,0},
                            {0,0,0,1,0,0,0,0,0,0},
                            {0,0,0,0,1,0,0,0,0,0},
                            {0,0,0,0,0,1,0,0,0,0},
                            {0,0,0,0,0,0,1,0,0,0},
                            {0,0,0,0,0,0,0,1,0,0},
                            {0,0,0,0,0,0,0,0,1,0},
                            {0,0,0,0,0,0,0,0,0,1}
                        };

        // constants
        int nSamples = 10;
        int nInputs = 45;
        int nodes = 5;
        int nOutputs = 10;
        int epocs = 20000;
        double learnRate = 0.01;

        
        //layer 1 weights (ninputs columns, nodes rows)
        //mth row is mth input, nth column is nth node
        double[][] W1 = np.random(nInputs, nodes);
        double[][] b1 = new double[nSamples][nodes];

        //layer 2 (output layer) weights
        double[][] W2 = np.random(nodes, nOutputs);
        double[][] b2 = new double[nSamples][nOutputs];


        //START LOOP HERE
        
        List<double[][]> toPlot = new ArrayList<double[][]>();
        for(int q = 0; q<epocs;q++){
            
            //apply weights layer 1
            double[][] sumL1 = np.add(np.dot(X,W1), b1);

            //apply sigmoid layer 1
            double[][] Z1 = np.sigmoid(sumL1);

            //apply weights layer 2
            double[][] sumL2 = np.add(np.dot(Z1,W2),b2);
            double[][] Z2 = np.sigmoid(sumL2);

            //BACK PROPaGATION

            //Layer 2
            double[][] deltaE2 = np.subtract(Z2, Y);

            double[][] dW2 = np.divide(np.dot(np.T(Z1), deltaE2), nSamples);

            double[][] db2 = np.divide(deltaE2, nSamples);


            //double[][] deltaW2 = np.dot(np.multiply(learnRate, Z1),deltaE2);
            //W2 = np.add(deltaW2, W2);

            //layer 1
            double[][] deltaE1 = np.multiply(np.dot(deltaE2, np.T(W2)), np.subtract(1.0,np.power(Z1, 2)));

            double[][] dW1 = np.divide(np.dot(np.T(X),deltaE1),nSamples);

            double[][] db1 = np.divide(deltaE1, nSamples);

            W1 = np.subtract(W1, np.multiply(learnRate, dW1));
            b1 = np.subtract(b1, np.multiply(learnRate, db1));

            W2 = np.subtract(W2, np.multiply(learnRate, dW2));
            b2 = np.subtract(b2, np.multiply(learnRate, db2));

            //grab values for plotting/printing
            if(q == epocs-1){
        		printMatrix(Z2);
        	}
            if(q % (epocs/10) == 0){
            	
            	toPlot.add(new double[][]{
            		{(double)q, SSE(deltaE2[0])},
            		{(double)q, SSE(deltaE2[1])},
            		{(double)q, SSE(deltaE2[2])},
            		{(double)q, SSE(deltaE2[3])},
            		{(double)q, SSE(deltaE2[4])},
            		{(double)q, SSE(deltaE2[5])},
            		{(double)q, SSE(deltaE2[6])},
            		{(double)q, SSE(deltaE2[7])},
            		{(double)q, SSE(deltaE2[8])},
            		{(double)q, SSE(deltaE2[9])},
            	});
            	
            }

        }
        List<double[]> newPoints = new ArrayList<double[]>();
        List<double[][]> readyToPlot= new ArrayList<double[][]>();
        
        //plot values
        for(int i = 0; i<toPlot.get(0).length;i++){
        	newPoints = new ArrayList<double[]>();
        	for(int j=0;j<toPlot.size();j++){
        		newPoints.add(toPlot.get(j)[i]);
        	}
        	readyToPlot.add( new double[][]{
        				newPoints.get(0),
        				newPoints.get(1),
        				newPoints.get(2),
        				newPoints.get(3),
        				newPoints.get(4),
        				newPoints.get(5),
        				newPoints.get(6),
        				newPoints.get(7),
        				newPoints.get(8),
        				newPoints.get(9)
        	});
        }
        
        for(int i=0;i<readyToPlot.size();i++){
        	final int index = i;
            SwingUtilities.invokeLater(() -> {
      	      plot example = new plot("Number: "+index, readyToPlot.get(index));
      	      example.setSize(800, 400);
      	      example.setLocationRelativeTo(null);
      	      example.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
      	      example.setVisible(true);
      	    });
        }
       
        
        
    }

    //function for printing mxn matrix
    public static void printMatrix(double[][] mat){

        System.out.print("[");
        for(int i = 0; i < mat.length; i++){
            for(int j = 0; j< mat[i].length; j++){
                System.out.print(mat[i][j] + ", ");
            }
            System.out.println("]");
            System.out.print("[");
        }
    }
    
    //function for printing an array
    public static void printMatrix(double[] mat){
    	for(int i = 0; i< mat.length;i++){
    		System.out.println(mat[i]);
    	}
    }
    
    //function to calculate the sum squared error
    public static double SSE(double[] mat){
    	double sum = 0;
    	for(int i = 0; i<mat.length;i++){
    		sum += mat[i];
    	}
    	
    	double mean = sum/(mat.length);
    	sum = 0;
    	for(int i = 0; i<mat.length;i++){
			sum += Math.pow((mat[i] - mean),2);
    	}
    	
    	return sum;
    	
    }

}