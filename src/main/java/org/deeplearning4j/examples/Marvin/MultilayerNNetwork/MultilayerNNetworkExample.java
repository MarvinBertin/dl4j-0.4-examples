package org.deeplearning4j.examples.Marvin.MultilayerNNetwork;

import org.canova.api.conf.Configuration;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.canova.api.writable.Writable;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.examples.csv.CSVExample;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;


/**
 * Created by marvinbertin on 11/14/15.
 */
public class MultilayerNNetworkExample {

    private static Logger log = LoggerFactory.getLogger(MultilayerNNetworkExample.class);

    public static void main(String[] args) throws Exception {
        RecordReader recordReader = new CSVRecordReader(0, ",");
        recordReader.initialize(
                new FileSplit(
                        new ClassPathResource("iris.txt")
                                .getFile()));

        //reader,label index,number of possible labels
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 4, 3);
        DataSet next = iterator.next();

        Nd4j.MAX_SLICES_TO_PRINT = 10;
        Nd4j.MAX_ELEMENTS_PER_SLICE = 10;

        final int numInputs = 4;
        int numOutputs = 3;
        int iterations = 5;
        long seed = 0;
        int listenerFreq = iterations;

        log.info("Build model...");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .useDropConnect(true)
                .learningRate(1e-1)
                .l1(1).regularization(true).l2(1)
                .list(3)
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs).nOut(3)
                        .activation("sigmoid").dropOut(0.5)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(3).nOut(2)
                        .activation("sigmoid")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(2).nOut(numOutputs)
                        .activation("softmax")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Arrays
                .asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        next.normalizeZeroMeanZeroUnitVariance();
        next.shuffle();

        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(0.6);
        model.fit(testAndTrain.getTrain());

        Evaluation eval = new Evaluation(numOutputs);
        DataSet test = testAndTrain.getTest();
        INDArray output = model.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(), output);
        log.info(eval.stats());

    }
}
