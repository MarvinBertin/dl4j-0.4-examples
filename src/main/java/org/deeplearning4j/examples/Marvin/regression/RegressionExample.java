package org.deeplearning4j.examples.Marvin.regression;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.optimize.solvers.LBFGS;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Identity;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

/**
 * Created by marvinbertin on 11/13/15.
 */
public class RegressionExample {

    private static Logger log = LoggerFactory.getLogger(RegressionExample.class);

    public static void main(String[] args) throws Exception {
        int seed = 3;
        int iterations = 1000;

        RecordReader reader =  new CSVRecordReader();
        reader.initialize(
                new FileSplit(
                        new ClassPathResource("regression-example.txt")
                                .getFile()));

        DataSetIterator iter = new RecordReaderDataSetIterator(reader,null,2029,12,1,true);
        DataSet next = iter.next();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .learningRate(1e-2f)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .l2(1).regularization(true)
                .list(1)
                .layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(12)
                        .nOut(1)
                        .activation("identity")
                        .weightInit(WeightInit.XAVIER)
                        .build()
                )
                .build();

        next.normalizeZeroMeanZeroUnitVariance();
        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(0.9);

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.setListeners(new ScoreIterationListener(50));
        network.fit(testAndTrain.getTrain());

        Evaluation eval = new Evaluation(2);
        DataSet test = testAndTrain.getTest();
        INDArray output = network.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(), output);
        log.info(eval.stats());


    }
}
