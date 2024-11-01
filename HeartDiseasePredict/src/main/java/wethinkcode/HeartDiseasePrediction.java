package wethinkcode;

import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import org.knowm.xchart.*;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.SwingWrapper;
import java.io.File;

public class HeartDiseasePrediction {

    public static void main(String[] args) throws Exception {
        // Load CSV data
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("src/main/resources/heart_disease_prediction.csv"));
        Instances data = loader.getDataSet();

        // Set class index
        data.setClassIndex(data.numAttributes() - 1);

        // Normalize the data
        Normalize normalize = new Normalize();
        normalize.setInputFormat(data);
        Instances normalizedData = Filter.useFilter(data, normalize);

        // Split data into training and testing sets
        int trainSize = (int) Math.round(normalizedData.numInstances() * 0.8);
        int testSize = normalizedData.numInstances() - trainSize;
        Instances trainData = new Instances(normalizedData, 0, trainSize);
        Instances testData = new Instances(normalizedData, trainSize, testSize);

        // Build KNN classifier
        IBk knn = new IBk(5);
        knn.buildClassifier(trainData);

        // Evaluate model accuracy
        int correct = 0;
        int incorrect = 0;
        for (int i = 0; i < testData.numInstances(); i++) {
            double actualClass = testData.instance(i).classValue();
            double predictedClass = knn.classifyInstance(testData.instance(i));
            if (actualClass == predictedClass) {
                correct++;
            } else {
                incorrect++;
            }
        }

        double accuracy = (double) correct / testData.numInstances() * 100;
        System.out.println("Model Accuracy: " + accuracy + "%");

        // Plot age distribution
        plotAgeDistribution(normalizedData);
    }

    public static void plotAgeDistribution(Instances data) {
        int numBins = 10;
        double minAge = Double.MAX_VALUE;
        double maxAge = Double.MIN_VALUE;

        // Get the index of the age attribute
        int ageIndex = data.attribute("Age").index();

        // Determine min and max age values
        for (int i = 0; i < data.numInstances(); i++) {
            double age = data.instance(i).value(ageIndex);
            if (age < minAge) minAge = age;
            if (age > maxAge) maxAge = age;
        }

        double binWidth = (maxAge - minAge) / numBins;
        double[] bins = new double[numBins];
        double[] binCounts = new double[numBins];

        // Count the number of occurrences in each bin
        for (int i = 0; i < data.numInstances(); i++) {
            double age = data.instance(i).value(ageIndex);
            int binIndex = (int) ((age - minAge) / binWidth);
            if (binIndex >= numBins) binIndex = numBins - 1;
            binCounts[binIndex]++;
        }

        // Set the midpoints for the bins
        for (int i = 0; i < numBins; i++) {
            bins[i] = minAge + (i + 0.5) * binWidth;
        }

        // Create a chart
        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(600)
                .title("Age Distribution")
                .xAxisTitle("Age")
                .yAxisTitle("Frequency")
                .build();

        chart.addSeries("Age Distribution", bins, binCounts);

        new SwingWrapper<>(chart).displayChart();
    }



}
