package br.com.mm.dlj.generic;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 *
 * @author mertins
 */
public class MLP {

    private Mnist mnist;
    private Model model;
    private Trainer trainer;

    private void readDS() throws IOException, TranslateException {
        int batchSize = 32;
        this.mnist = Mnist.builder().setSampling(batchSize, true).build();
        this.mnist.prepare();
    }

    private void makeModel() {
        this.model = Model.newInstance("mpl");
        model.setBlock(new Mlp(28 * 28, 10, new int[]{128, 64}));
    }

    private void training(int epoch) throws IOException, TranslateException {
        //softmaxCrossEntropyLoss is a standard loss for classification problems
        // Use accuracy so we humans can understand how accurate the model is
        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging());
        // Now that we have our training configuration, we should create a new trainer for our model
        this.trainer = model.newTrainer(config);
        this.trainer.initialize(new Shape(1, 28 * 28));
        EasyTrain.fit(trainer, epoch, mnist, null);
    }

    private void save(int epoch) throws IOException, TranslateException {
        Path modelDir = Paths.get("src/main/buildModel/mlpJava");
        Files.createDirectories(modelDir);
        model.setProperty("Epoch", String.valueOf(epoch));
        model.save(modelDir, "mlp");
        System.out.println(model);
    }

    private void doModel() throws IOException, TranslateException {
        int epoch = 2;
        this.readDS();
        this.makeModel();
        this.training(epoch);
        this.save(epoch);
    }

    public static void main(String[] args) throws IOException, TranslateException {
        MLP mlp = new MLP();
        mlp.doModel();

    }
}
