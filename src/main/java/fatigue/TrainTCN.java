package fatigue;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;

import ai.djl.nn.convolutional.Conv1d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;

import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;

import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Dataset;

import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.tracker.Tracker;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import java.nio.FloatBuffer;
import java.nio.LongBuffer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import java.util.ArrayList;
import java.util.List;


public class TrainTCN {

    // =========================================================
    // CONFIGURATION
    // =========================================================

    private static final int TIME_STEPS = 20;
    private static final int FEATURES = 10;
    private static final int NUM_CLASSES = 2;

    private static final int BATCH_SIZE = 32;
    private static final int EPOCHS = 20;

    private static final float LEARNING_RATE = 0.001f;

    private static final String TRAIN_FILE =
            "data/processed/train.csv";

    private static final String VALIDATION_FILE =
            "data/processed/validation.csv";

    private static final String TEST_FILE =
            "data/processed/test.csv";

    private static final String MODEL_DIR =
            "models";

    // =========================================================
    // MAIN
    // =========================================================

    public static void main(String[] args) {

        System.out.println();
        System.out.println("======================================");
        System.out.println("       HAMSTRING FATIGUE TCN");
        System.out.println("======================================");

        System.out.println();
        System.out.println("DJL Engine : "
                + Engine.getInstance().getEngineName());

        System.out.println("DJL Version: "
                + Engine.getInstance().getVersion());

        System.out.println();

        System.out.println("Expected input:");
        System.out.println("(samples, 20, 10)");

        System.out.println();

        try (NDManager manager = NDManager.newBaseManager()) {

            // =================================================
            // LOAD DATA
            // =================================================

            System.out.println("======================================");
            System.out.println("        LOADING DATA");
            System.out.println("======================================");

            DataSet trainData =
                    readCSV(TRAIN_FILE);

            DataSet validationData =
                    readCSV(VALIDATION_FILE);

            DataSet testData =
                    readCSV(TEST_FILE);

            printDataInfo(
                    "TRAIN",
                    trainData
            );

            printDataInfo(
                    "VALIDATION",
                    validationData
            );

            printDataInfo(
                    "TEST",
                    testData
            );

            // =================================================
            // CREATE NDARRAYS
            // =================================================

            System.out.println();
            System.out.println("======================================");
            System.out.println("       CREATING NDARRAY DATA");
            System.out.println("======================================");

            NDArray trainX =
                    createInputArray(
                            manager,
                            trainData.features
                    );

            NDArray trainY =
                    createLabelArray(
                            manager,
                            trainData.labels
                    );

            NDArray validationX =
                    createInputArray(
                            manager,
                            validationData.features
                    );

            NDArray validationY =
                    createLabelArray(
                            manager,
                            validationData.labels
                    );

            NDArray testX =
                    createInputArray(
                            manager,
                            testData.features
                    );

            NDArray testY =
                    createLabelArray(
                            manager,
                            testData.labels
                    );

            System.out.println();
            System.out.println(
                    "TRAIN X : " + trainX.getShape()
            );

            System.out.println(
                    "TRAIN Y : " + trainY.getShape()
            );

            System.out.println(
                    "VALID X : " + validationX.getShape()
            );

            System.out.println(
                    "VALID Y : " + validationY.getShape()
            );

            System.out.println(
                    "TEST X  : " + testX.getShape()
            );

            System.out.println(
                    "TEST Y  : " + testY.getShape()
            );

            // =================================================
            // CREATE DATASETS
            // =================================================

            Dataset trainingSet =
                    new ArrayDataset.Builder()
                            .setData(trainX)
                            .optLabels(trainY)
                            .setSampling(
                                    trainData.features.length,
                                    false
                            )
                            .build();

            Dataset validationSet =
                    new ArrayDataset.Builder()
                            .setData(validationX)
                            .optLabels(validationY)
                            .setSampling(
                                    validationData.features.length,
                                    false
                            )
                            .build();

            // =================================================
            // CREATE TCN MODEL
            // =================================================

            System.out.println();
            System.out.println("======================================");
            System.out.println("          BUILDING TCN");
            System.out.println("======================================");

            Model model =
                    Model.newInstance(
                            "HamstringFatigueTCN"
                    );

            SequentialBlock tcn =
                    buildTCN();

            model.setBlock(tcn);

            System.out.println();
            System.out.println("TCN architecture created.");

            // =================================================
            // TRAINING CONFIGURATION
            // =================================================

            Tracker learningRate =
                    Tracker.fixed(
                            LEARNING_RATE
                    );

            Adam optimizer =
                    Adam.builder()
                            .optLearningRateTracker(
                                    learningRate
                            )
                            .build();

            Loss loss =
                    Loss.softmaxCrossEntropyLoss();

            DefaultTrainingConfig config =
                    new DefaultTrainingConfig(loss)
                            .optOptimizer(optimizer)
                            .addEvaluator(
                                    new Accuracy()
                            );

            // =================================================
            // TRAIN
            // =================================================

            System.out.println();
            System.out.println("======================================");
            System.out.println("          STARTING TRAINING");
            System.out.println("======================================");

            try (Trainer trainer =
                         model.newTrainer(config)) {

                trainer.initialize(
                        new Shape(
                                1,
                                FEATURES,
                                TIME_STEPS
                        )
                );

                System.out.println(
                        "Model input shape: "
                                + new Shape(
                                1,
                                FEATURES,
                                TIME_STEPS
                        )
                );

                TrainingResult result =
                        EasyTrain.fit(
                                trainer,
                                EPOCHS,
                                trainingSet,
                                validationSet
                        );

                System.out.println();
                System.out.println(
                        "======================================"
                );

                System.out.println(
                        "        TRAINING COMPLETED"
                );

                System.out.println(
                        "======================================"
                );

                System.out.println(
                        "Epochs completed : "
                                + result.getEpoch()
                );

                System.out.println(
                        "Training loss    : "
                                + result.getTrainLoss()
                );

                // =================================================
                // SAVE MODEL
                // =================================================

                Path modelPath =
                        Paths.get(MODEL_DIR);

                Files.createDirectories(
                        modelPath
                );

                model.save(
                        modelPath,
                        "hamstring_fatigue_tcn"
                );

                System.out.println();
                System.out.println(
                        "Model saved to:"
                );

                System.out.println(
                        "models/hamstring_fatigue_tcn"
                );
            }

            model.close();

            // =================================================
            // TEST
            // =================================================

            System.out.println();
            System.out.println("======================================");
            System.out.println("           TEST DATA READY");
            System.out.println("======================================");

            System.out.println(
                    "Test samples : "
                            + testData.features.length
            );

            System.out.println(
                    "Test labels  : "
                            + testData.labels.length
            );

            System.out.println();
            System.out.println(
                    "Next step: evaluate the trained TCN."
            );

        } catch (Exception e) {

            System.err.println();
            System.err.println(
                    "======================================"
            );

            System.err.println(
                    "             ERROR"
            );

            System.err.println(
                    "======================================"
            );

            e.printStackTrace();
        }
    }


    // =========================================================
    // BUILD TCN
    // =========================================================

    private static SequentialBlock buildTCN() {

        SequentialBlock block =
                new SequentialBlock();

        /*
         * Input:
         *
         * (batch, 10 features, 20 time steps)
         *
         * Conv1d expects NCW layout:
         *
         * N = batch
         * C = channels/features
         * W = time
         */

        // -----------------------------------------------------
        // TCN BLOCK 1
        // -----------------------------------------------------

        block.add(
                new Conv1d.Builder()
                        .setKernelShape(
                                new Shape(3)
                        )
                        .setFilters(32)
                        .optPadding(
                                new Shape(2)
                        )
                        .optDilation(
                                new Shape(1)
                        )
                        .build()
        );

        block.add(
                Activation::relu
        );

        block.add(
                Dropout.builder()
                        .optRate(0.20f)
                        .build()
        );


        // -----------------------------------------------------
        // TCN BLOCK 2
        // -----------------------------------------------------

        block.add(
                new Conv1d.Builder()
                        .setKernelShape(
                                new Shape(3)
                        )
                        .setFilters(32)
                        .optPadding(
                                new Shape(2)
                        )
                        .optDilation(
                                new Shape(2)
                        )
                        .build()
        );

        block.add(
                Activation::relu
        );

        block.add(
                Dropout.builder()
                        .optRate(0.20f)
                        .build()
        );


        // -----------------------------------------------------
        // TCN BLOCK 3
        // -----------------------------------------------------

        block.add(
                new Conv1d.Builder()
                        .setKernelShape(
                                new Shape(3)
                        )
                        .setFilters(64)
                        .optPadding(
                                new Shape(4)
                        )
                        .optDilation(
                                new Shape(4)
                        )
                        .build()
        );

        block.add(
                Activation::relu
        );

        block.add(
                Dropout.builder()
                        .optRate(0.20f)
                        .build()
        );


        // -----------------------------------------------------
        // FLATTEN
        // -----------------------------------------------------

        block.add(
                Blocks.batchFlattenBlock()
        );


        // -----------------------------------------------------
        // DENSE LAYER
        // -----------------------------------------------------

        block.add(
                Linear.builder()
                        .setUnits(64)
                        .build()
        );

        block.add(
                Activation::relu
        );

        block.add(
                Dropout.builder()
                        .optRate(0.30f)
                        .build()
        );


        // -----------------------------------------------------
        // OUTPUT
        // -----------------------------------------------------

        block.add(
                Linear.builder()
                        .setUnits(NUM_CLASSES)
                        .build()
        );

        return block;
    }


    // =========================================================
    // READ CSV
    // =========================================================

    private static DataSet readCSV(
            String file
    ) throws IOException {

        System.out.println();
        System.out.println(
                "Reading: " + file
        );

        List<float[]> featureList =
                new ArrayList<>();

        List<Long> labelList =
                new ArrayList<>();

        try (BufferedReader br =
                     new BufferedReader(
                             new FileReader(file)
                     )) {

            String line;

            boolean header = true;

            while ((line = br.readLine()) != null) {

                if (header) {
                    header = false;
                    continue;
                }

                if (line.trim().isEmpty()) {
                    continue;
                }

                String[] parts =
                        line.split(",", -1);

                /*
                 * CSV structure:
                 *
                 * column 0   = Subject
                 * columns 1-201 = 200 sequence values
                 * column 202 = Label
                 *
                 * 200 = 20 * 10
                 */

                if (parts.length < 203) {
                    continue;
                }

                float[] features =
                        new float[
                                TIME_STEPS * FEATURES
                        ];

                for (int i = 0; i < 200; i++) {

                    String value =
                            parts[i + 1]
                                    .trim();

                    features[i] =
                            Float.parseFloat(
                                    value
                            );
                }

                long label =
                        Long.parseLong(
                                parts[202]
                                        .trim()
                        );

                featureList.add(
                        features
                );

                labelList.add(
                        label
                );
            }
        }

        float[][] features =
                featureList.toArray(
                        new float[0][]
                );

        long[] labels =
                new long[
                        labelList.size()
                ];

        for (int i = 0;
             i < labelList.size();
             i++) {

            labels[i] =
                    labelList.get(i);
        }

        return new DataSet(
                features,
                labels
        );
    }


    // =========================================================
    // CREATE INPUT ARRAY
    // =========================================================

    private static NDArray createInputArray(
            NDManager manager,
            float[][] data
    ) {

        int samples =
                data.length;

        float[] flat =
                new float[
                        samples
                                * TIME_STEPS
                                * FEATURES
                ];

        int index = 0;

        for (float[] row : data) {

            for (float value : row) {

                flat[index++] =
                        value;
            }
        }

        /*
         * Original:
         *
         * (samples, 20, 10)
         *
         * TCN requires:
         *
         * (samples, 10, 20)
         */

        NDArray array =
                manager.create(
                        FloatBuffer.wrap(flat),
                        new Shape(
                                samples,
                                TIME_STEPS,
                                FEATURES
                        ),
                        DataType.FLOAT32
                );

        return array.transpose(
                0,
                2,
                1
        );
    }


    // =========================================================
    // CREATE LABEL ARRAY
    // =========================================================

    private static NDArray createLabelArray(
            NDManager manager,
            long[] labels
    ) {

        return manager.create(
                LongBuffer.wrap(labels),
                new Shape(labels.length),
                DataType.INT64
        );
    }


    // =========================================================
    // PRINT DATA INFO
    // =========================================================

    private static void printDataInfo(
            String name,
            DataSet data
    ) {

        System.out.println();

        System.out.println(
                name
                        + " samples : "
                        + data.features.length
        );

        System.out.println(
                name
                        + " features: "
                        + (
                        data.features.length > 0
                                ? data.features[0].length
                                : 0
                )
        );

        int nonFatigue = 0;
        int fatigue = 0;

        for (long label : data.labels) {

            if (label == 0) {
                nonFatigue++;
            } else if (label == 1) {
                fatigue++;
            }
        }

        System.out.println(
                name
                        + " non-fatigue: "
                        + nonFatigue
        );

        System.out.println(
                name
                        + " fatigue    : "
                        + fatigue
        );
    }


    // =========================================================
    // DATASET CLASS
    // =========================================================

    private static class DataSet {

        float[][] features;
        long[] labels;

        DataSet(
                float[][] features,
                long[] labels
        ) {

            this.features =
                    features;

            this.labels =
                    labels;
        }
    }
}