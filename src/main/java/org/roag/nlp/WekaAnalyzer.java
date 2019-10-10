package org.roag.nlp;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.classifiers.trees.HoeffdingTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.LogConfiguration;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.activations.ActivationReLU;
import weka.dl4j.activations.ActivationSoftmax;
import weka.dl4j.layers.DenseLayer;
import weka.dl4j.layers.OutputLayer;
import weka.dl4j.lossfunctions.LossMCXENT;
import weka.dl4j.updater.Adam;

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
        data.setClassIndex(data.numAttributes() - 1);//we suppose that the last field is always a class
        return data;
    }

    public Instances getTestDataset() {
        return testData;
    }

    public Instances getTrainDataset() {
        return trainData;
    }


    public WekaAnalyzer deepLearning() throws Exception {
        // Create a new Multi-Layer-Perceptron classifier
        Dl4jMlpClassifier dl4jMlpClassifier = new Dl4jMlpClassifier();
        // Set a seed for reproduceable results
        dl4jMlpClassifier.setSeed(1);

        DenseLayer denseLayer = new DenseLayer();
        denseLayer.setNOut(10);
        denseLayer.setActivationFunction(new ActivationReLU());

        // Define the output layer
        OutputLayer outputLayer = new OutputLayer();
        outputLayer.setActivationFunction(new ActivationSoftmax());
        outputLayer.setLossFn(new LossMCXENT());

        NeuralNetConfiguration neuralNetConfiguration = new NeuralNetConfiguration();
        neuralNetConfiguration.setUpdater(new Adam());

        // Add the layers to the classifier
        dl4jMlpClassifier.setLayers(denseLayer, outputLayer);
        dl4jMlpClassifier.setNeuralNetConfiguration(neuralNetConfiguration);


        LOG.info("====== DEEP LEARNING TRAINING RESULTS ========");
        // Evaluate the network
        Evaluation trainEval = new Evaluation(getTrainDataset());
        int numFolds = 5;
        trainEval.crossValidateModel(dl4jMlpClassifier, getTrainDataset(), numFolds, new Random(1));
        LOG.info(trainEval.toSummaryString());


        LOG.info("====== DEEP LEARNING PREDICTIONS ========");
        dl4jMlpClassifier.buildClassifier(getTrainDataset());
        Evaluation testEval = new Evaluation(getTrainDataset());
        testEval.evaluateModel(dl4jMlpClassifier, getTestDataset());
        LOG.info(testEval.toSummaryString());

        //TODO: why classifyIt does not work for dl4j?
        classifyIt(dl4jMlpClassifier);
        LOG.info("===============================");
        testEval.evaluateModelOnceAndRecordPrediction(dl4jMlpClassifier, getTestDataset().get(0));
        LOG.info(testEval.toSummaryString());
        LOG.info("Correct: {}; Incorrect: {}", testEval.pctCorrect(), testEval.pctIncorrect());
        testEval.predictions().forEach(
                p -> LOG.info("Predictions: actual: {}; predicted: {}", p.actual(), p.predicted())
        );
        return this;
    }

    public WekaAnalyzer naiveBayes() throws Exception {
        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.buildClassifier(getTrainDataset());
        LOG.info(" ====== NAIVE BAYES PREDICTIONS =====");
        Evaluation testEval = new Evaluation(getTrainDataset());
        testEval.evaluateModel(naiveBayes, getTestDataset());
        LOG.info(testEval.toSummaryString());
        classifyIt(naiveBayes);
        return this;
    }

    public WekaAnalyzer hoeffdingTree() throws Exception {
        HoeffdingTree hoeffdingTree = new HoeffdingTree();
        hoeffdingTree.buildClassifier(trainData);
        LOG.info(" ====== HOEFFDING TREE PREDICTIONS =====");
        Evaluation testEval = new Evaluation(getTrainDataset());
        testEval.evaluateModel(hoeffdingTree, getTestDataset());
        LOG.info(testEval.toSummaryString());
        classifyIt(hoeffdingTree);
        return this;

    }

    private void classifyIt(AbstractClassifier classifier) throws Exception {
        int idx = 0;
        for (Instance i : getTestDataset()) {
            double label = classifier.classifyInstance(i);
            LOG.info("{} - Prediction for instance {} ({}): {} ({})",
                    idx++,
                    i.stringValue(i.numAttributes() - 1),
                    i.classValue(),
                    i.classAttribute().value((int) label),
                    label
            );
        }

    }

    public static void main(String[] args) {
        try {
            new WekaAnalyzer("src/main/resources/data/weka_vote.arff",
                    "src/main/resources/data/weka_vote_test.arff")
                    .naiveBayes()
                    .hoeffdingTree()
                    .deepLearning();

        } catch (Exception e) {
            LOG.error(e);
        }
    }
}
