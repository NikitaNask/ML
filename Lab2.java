import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;

public class Lab2{
    final static double survivalBorder = 0.02;
    final static int numberOfSurvivors = 5;
    final static int numberOfGenerations = 20;
    final static double accuracy = 0.5;

    public static double countNRMSE(int[] Y, double[] YPredicted){
        double sum = 0.0;

        for (int i = 0; i < Y.length; ++i)
            sum += Math.pow(Y[i] - YPredicted[i], 2);

        return Math.sqrt(sum / Y.length);
    }

    public static double[] produceRandomState(double[] weights){

        for (int i = 0; i < weights.length; ++i){
            Random random = new Random(System.currentTimeMillis());

            if (weights[i] != 0.0)
                weights[i] += ((double)random.nextInt(50) - 100.0) / 10.0;
        }

        return weights;
    }

    public static double[] mutate(double[] weights){
        for (int i = 0; i < weights.length; ++i){

            if (Math.abs(weights[i]) < survivalBorder)
                weights[i] = 0.0;
        }

        return weights;
    }

    public static double[] fit(double[] weights, int[][] data){
        double[] result = new double[data.length];

        for (int i = 0; i < data.length; ++i){
            for(int j = 0; j < data[0].length - 1; ++j){
                result[i] += data[i][j] * weights[j];
            }
        }

        return result;
    }

    public static void calculate(int numberOfNewSpeciesFromEachSurvivor) {
        try (Scanner scanner = new Scanner(new File("C:\\Users\\1358365\\IdeaProjects\\ML\\resources\\1_train.txt"));) {
            int N = scanner.nextInt();
            int M = scanner.nextInt();

            int[][] data = new int[N][M];
            int[] target = new int[N];

            for (int i = 0; i < N; ++i){
                for (int j = 0; j < M; ++j){
                    data[i][j] = scanner.nextInt();
                }
                target[i] = scanner.nextInt();
            }


        double[][] survivors = new double[numberOfSurvivors][M + 1];

        Random random = new Random(System.currentTimeMillis());
        for (int j = 0; j < numberOfSurvivors; ++j)
            for (int i = 0; i < M + 1; i++)
                survivors[j][i] = random.nextInt(5);

        int generataion = 0;
        double[] results = new double[(numberOfNewSpeciesFromEachSurvivor + 1 ) * numberOfSurvivors];
        double[][] currentSpecies = new double[(numberOfNewSpeciesFromEachSurvivor + 1 ) * numberOfSurvivors][M + 1];

        while (generataion < numberOfGenerations) {

            for (int i = 0; i < numberOfSurvivors; ++i){
                 currentSpecies[i] = survivors[i];
                 double[] YPredicted = fit(survivors[i], data);
                 results[i] = countNRMSE(target, YPredicted);
            }

            for (int j = 1; j <= numberOfSurvivors; ++j) {
                for (int i = 0; i < numberOfNewSpeciesFromEachSurvivor; ++i) {
                    currentSpecies[j * numberOfSurvivors + i] = produceRandomState(survivors[j - 1]);
                    double[] YPredicted = fit(currentSpecies[j * numberOfSurvivors + i], data);
                    results[j * numberOfSurvivors + i] = countNRMSE(target, YPredicted);
                }
            }

            for (int i = 0; i < numberOfSurvivors * ( 1 + numberOfNewSpeciesFromEachSurvivor); ++i){
                for (int j = i; j < numberOfSurvivors * (1 + numberOfNewSpeciesFromEachSurvivor); ++j) {
                    if (results[i] < results[j]) {
                        double tmp = results[i];
                        results[i] = results[j];
                        results[j] = tmp;

                        double[] tmpArr = currentSpecies[i];
                        currentSpecies[i] = currentSpecies[j];
                        currentSpecies[j] = tmpArr;
                    }
                }
            }

            for (int i = 0; i < numberOfSurvivors; ++i){
                survivors[i] = currentSpecies[i];
            }

            generataion++;
        }


            System.out.printf("%1$+03.7f ", results[0]);

            try (Scanner otherScanner = new Scanner(new File("C:\\Users\\1358365\\IdeaProjects\\ML\\resources\\1_test.txt"));) {
                int NN = otherScanner.nextInt();

                int[][] testData = new int[NN][M];
                int[] testTarget = new int[NN];

                for (int i = 0; i < NN; ++i) {
                    for (int j = 0; j < M; ++j) {
                        testData[i][j] = otherScanner.nextInt();
                    }
                    testTarget[i] = otherScanner.nextInt();
                }

                double[] result = fit(survivors[0], testData);
                System.out.printf("%1$+03.7f ", countNRMSE(testTarget, result));
                System.out.println();
            }
            } catch (FileNotFoundException exception) {
            System.out.println("File not found");
        }


    }

    public static void main(String[] args) {
        for (int i = 5; i < 45; ++i )
            calculate(i);
    }
}