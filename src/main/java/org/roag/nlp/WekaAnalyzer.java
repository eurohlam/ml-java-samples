package org.roag.nlp;

import com.sun.xml.internal.ws.policy.privateutil.PolicyUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.Instances;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.activations.ActivationSoftmax;
import weka.dl4j.layers.Layer;
import weka.dl4j.layers.OutputLayer;
import weka.dl4j.lossfunctions.LossMCXENT;
import weka.dl4j.updater.Adam;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

public class WekaAnalyzer {

    private static final Logger LOG = LogManager.getLogger(WekaAnalyzer.class);

    private final Instances trainData;
    private final Instances testData;

    public WekaAnalyzer(final String trainData, final String testData) throws IOException {
        this.trainData = getDataset(trainData);
        this.testData = getDataset(testData);
    }

    private Instances getDataset(String path) throws IOException {
        Instances data = new Instances(new FileReader(path));
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    public Instances getTestDataset() throws IOException {
        return testData;
    }

    public Instances getTrainDataset() throws IOException {
        return trainData;
    }


    public WekaAnalyzer deepLearning() throws Exception {
        // Create a new Multi-Layer-Perceptron classifier
        Dl4jMlpClassifier dl4jMlpClassifier = new Dl4jMlpClassifier();
        // Set a seed for reproduceable results
        dl4jMlpClassifier.setSeed(1);

        // Define the output layer
        OutputLayer outputLayer = new OutputLayer();
        outputLayer.setActivationFunction(new ActivationSoftmax());
        outputLayer.setLossFn(new LossMCXENT());

        NeuralNetConfiguration neuralNetConfiguration = new NeuralNetConfiguration();
        neuralNetConfiguration.setUpdater(new Adam());

        // Add the layers to the classifier
        dl4jMlpClassifier.setLayers(new Layer[]{outputLayer});
        dl4jMlpClassifier.setNeuralNetConfiguration(neuralNetConfiguration);

        // Evaluate the network
        Evaluation trainEval = new Evaluation(getTrainDataset());
        int numFolds = 5;
        trainEval.crossValidateModel(dl4jMlpClassifier, getTrainDataset(), numFolds, new Random(1));

        LOG.info("====== DEEP LEARNING TRAINING RESULTS ========");
        LOG.info(trainEval.toSummaryString());
        trainEval.predictions().forEach(
                p -> LOG.info("Predictions: actual: {}; predicted: {}", p.actual(), p.predicted())
        );


        LOG.info("====== DEEP LEARNING PREDICTIONS ========");
        //TODO: it does not work
        Evaluation testEval = new Evaluation(getTrainDataset());
        testEval.evaluateModelOnce(dl4jMlpClassifier, testData.instance(0));
        LOG.info(testEval.toSummaryString());
        return this;
    }

    public WekaAnalyzer naiveBayes() throws Exception {
        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.buildClassifier(getTrainDataset());

        LOG.info(" ====== NAIVE BAYES PREDICTIONS =====");
        getTestDataset().forEach(
                i -> {
                    try {
                        LOG.info("Prediction for instance {}: {}",
                                i.stringValue(i.numAttributes() - 1),
                                naiveBayes.classifyInstance(i));
                    } catch (Exception e) {
                        LOG.error(e);
                    }
                }
        );
        return this;
    }

    public void hoeffdingTree() {

    }

    public static void main(String[] args) {
        try {
            new WekaAnalyzer("src/main/resources/data/weka_vote.arff",
                    "src/main/resources/data/weka_vote_test.arff")
                    .naiveBayes()
                    .deepLearning();

        } catch (Exception e) {
            LOG.error(e);
        }
    }
}
