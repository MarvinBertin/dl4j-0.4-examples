package org.deeplearning4j.examples.Marvin.mlp;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

//import org.deeplearning4j.plot.NeuralNetPlotter;

import java.io.IOException;
import java.util.Arrays;

/**
 * Created by marvinbertin on 11/14/15.
 */
public class MLPBackpropIris {

    private static Logger log = LoggerFactory.getLogger(MLPBackpropIris.class);

    public static void main(String[] args) throws IOException {

        Nd4j.MAX_SLICES_TO_PRINT = 10;
        Nd4j.MAX_ELEMENTS_PER_SLICE = 10;



        int numOutputs = 3;
        int numSamples = 150;
        int batchSize = 150;
        int iterations = 100;
        long seed = 6;
        int listenerFreq = iterations / 5;

        log.info("load data...");
        DataSetIterator iter = new IrisDataSetIterator(batchSize, numSamples);

        log.info("build model...");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .learningRate(1e-1)
                .l1(0).regularization(true).l2(1e-2)
                .constrainGradientToUnitNorm(true)
                .list(3)
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs).nOut(3)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(3).nOut(2)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunction.MCXENT)
                        .nIn(2).nOut(numOutputs)
                        .activation("softmax")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        log.info("Train model...");
        while(iter.hasNext()) {
            DataSet iris = iter.next();
            iris.normalizeZeroMeanZeroUnitVariance();
            model.fit(iris);
        }
        iter.reset();

        log.info("Evaluate weights...");
        for(org.deeplearning4j.nn.api.Layer layer : model.getLayers()) {
            INDArray w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
            INDArray b = layer.getParam(DefaultParamInitializer.BIAS_KEY);
            log.info("Weights " + w);
            log.info("bias" + b);
        }

        log.info("Evaluate model...");
        Evaluation eval = new Evaluation(numOutputs);
        DataSetIterator iterTest = new IrisDataSetIterator(numSamples, numSamples);
        DataSet test = iterTest.next();
        test.normalizeZeroMeanZeroUnitVariance();

        INDArray output = model.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(), output);
        log.info(eval.stats());
        log.info("******OVER*****");

//        NeuralNetPlotter plot = new NeuralNetPlotter();
//        plot.renderFilter(model.getLayer(1),10);

    }

}
