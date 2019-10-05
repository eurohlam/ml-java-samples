package org.roag.nlp;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.Instances;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.activations.ActivationSoftmax;
import weka.dl4j.layers.Layer;
import weka.dl4j.layers.OutputLayer;
import weka.dl4j.lossfunctions.LossMCXENT;
import weka.dl4j.updater.Adam;

import java.io.FileReader;
import java.util.Random;

public class WekaAnalyzer {

    public static void main(String[] args) {
        try {
// Create a new Multi-Layer-Perceptron classifier
            Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
// Set a seed for reproducable results
            clf.setSeed(1);

// Load the iris dataset and set its class index
            Instances data = new Instances(new FileReader("src/main/resources/data/iris.arff"));
            data.setClassIndex(data.numAttributes() - 1);

// Define the output layer
            OutputLayer outputLayer = new OutputLayer();
            outputLayer.setActivationFunction(new ActivationSoftmax());
            outputLayer.setLossFn(new LossMCXENT());

            NeuralNetConfiguration nnc = new NeuralNetConfiguration();
            nnc.setUpdater(new Adam());

// Add the layers to the classifier
            clf.setLayers(new Layer[]{outputLayer});
            clf.setNeuralNetConfiguration(nnc);

// Evaluate the network
            Evaluation eval = new Evaluation(data);
            int numFolds = 10;
            eval.crossValidateModel(clf, data, numFolds, new Random(1));

            System.out.println(eval.toSummaryString());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
