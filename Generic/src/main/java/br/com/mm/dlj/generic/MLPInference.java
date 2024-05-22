package br.com.mm.dlj.generic;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 *
 * @author mertins
 */
public class MLPInference {

    private static class MyTranslator implements Translator<Image, Classifications> {

        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            // Convert Image to NDArray
            NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.GRAYSCALE);
            return new NDList(NDImageUtils.toTensor(array));
        }

        @Override
        public Classifications processOutput(TranslatorContext ctx, NDList list) {
            // Create a Classifications with the output probabilities
            NDArray probabilities = list.singletonOrThrow().softmax(0);
            List<String> classNames = IntStream.range(0, 10).mapToObj(String::valueOf).collect(
                    Collectors.toList());
            return new Classifications(classNames, probabilities);
        }

        @Override
        public Batchifier getBatchifier() {
            // The Batchifier describes how to combine a batch together
            // Stacking, the most common batchifier, takes N [X1, X2, ...] arrays to a single [N, X1, X2, ...] array
            return Batchifier.STACK;
        }
    }

    public static void main(String[] args)
            throws FileNotFoundException, IOException, MalformedModelException, TranslateException {
        var img = ImageFactory.getInstance().fromInputStream(new FileInputStream("src/main/data/mnist/5.png"));
        Path modelDir = Paths.get("src/main/buildModel/mlpJava");
        Model model = Model.newInstance("mlp");
        model.setBlock(new Mlp(28 * 28, 10, new int[]{128, 64}));
        model.load(modelDir);

        Translator translator = new MyTranslator();
        var predictor = model.newPredictor(translator);
        var classifications = predictor.predict(img);
        System.out.println(classifications);
    }
}
