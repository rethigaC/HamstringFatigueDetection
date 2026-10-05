package fatigue;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class TrainModel {

    // ==========================================
    // PROJECT CONFIGURATION
    // ==========================================

    private static final int TIME_STEPS = 20;
    private static final int FEATURES_PER_STEP = 10;
    private static final int TOTAL_FEATURES = TIME_STEPS * FEATURES_PER_STEP;

    private static final String TRAIN_FILE =
            "data/processed/train.csv";

    private static final String VALIDATION_FILE =
            "data/processed/validation.csv";

    private static final String TEST_FILE =
            "data/processed/test.csv";


    public static void main(String[] args) {

        System.out.println();
        System.out.println("======================================");
        System.out.println("     HAMSTRING FATIGUE ML TRAINING");
        System.out.println("======================================");

        System.out.println();
        System.out.println("Expected input shape:");
        System.out.println("(samples, 20, 10)");

        System.out.println();
        System.out.println("Total features per sequence : "
                + TOTAL_FEATURES);

        // ------------------------------------------
        // LOAD TRAINING DATA
        // ------------------------------------------

        System.out.println();
        System.out.println("======================================");
        System.out.println("        LOADING TRAINING DATA");
        System.out.println("======================================");

        Dataset train = loadDataset(TRAIN_FILE);

        printDatasetInfo("TRAINING", train);


        // ------------------------------------------
        // LOAD VALIDATION DATA
        // ------------------------------------------

        System.out.println();
        System.out.println("======================================");
        System.out.println("       LOADING VALIDATION DATA");
        System.out.println("======================================");

        Dataset validation = loadDataset(VALIDATION_FILE);

        printDatasetInfo("VALIDATION", validation);


        // ------------------------------------------
        // LOAD TEST DATA
        // ------------------------------------------

        System.out.println();
        System.out.println("======================================");
        System.out.println("          LOADING TEST DATA");
        System.out.println("======================================");

        Dataset test = loadDataset(TEST_FILE);

        printDatasetInfo("TEST", test);


        // ------------------------------------------
        // CREATE NDARRAYS
        // ------------------------------------------

        System.out.println();
        System.out.println("======================================");
        System.out.println("       CREATING ML INPUT TENSORS");
        System.out.println("======================================");

        try (NDManager manager = NDManager.newBaseManager()) {

            NDArray trainX = manager.create(
                    train.features
            ).reshape(
                    train.features.length,
                    TIME_STEPS,
                    FEATURES_PER_STEP
            );

            NDArray validationX = manager.create(
                    validation.features
            ).reshape(
                    validation.features.length,
                    TIME_STEPS,
                    FEATURES_PER_STEP
            );

            NDArray testX = manager.create(
                    test.features
            ).reshape(
                    test.features.length,
                    TIME_STEPS,
                    FEATURES_PER_STEP
            );


            NDArray trainY = manager.create(train.labels);

            NDArray validationY =
                    manager.create(validation.labels);

            NDArray testY =
                    manager.create(test.labels);


            System.out.println();
            System.out.println("TRAIN X shape      : "
                    + trainX.getShape());

            System.out.println("TRAIN Y shape      : "
                    + trainY.getShape());

            System.out.println("VALIDATION X shape : "
                    + validationX.getShape());

            System.out.println("VALIDATION Y shape : "
                    + validationY.getShape());

            System.out.println("TEST X shape       : "
                    + testX.getShape());

            System.out.println("TEST Y shape       : "
                    + testY.getShape());


            // ------------------------------------------
            // PRINT LABEL DISTRIBUTION
            // ------------------------------------------

            printLabelDistribution(
                    "TRAINING",
                    train.labels
            );

            printLabelDistribution(
                    "VALIDATION",
                    validation.labels
            );

            printLabelDistribution(
                    "TEST",
                    test.labels
            );


            // ------------------------------------------
            // DATA READY
            // ------------------------------------------

            System.out.println();
            System.out.println("======================================");
            System.out.println("       ML DATA PREPARATION DONE");
            System.out.println("======================================");

            System.out.println();
            System.out.println("Training input:");
            System.out.println(trainX.getShape());

            System.out.println();
            System.out.println("Validation input:");
            System.out.println(validationX.getShape());

            System.out.println();
            System.out.println("Testing input:");
            System.out.println(testX.getShape());

            System.out.println();
            System.out.println("The datasets are ready for TCN training.");
        }
    }


    // ============================================================
    // DATASET LOADER
    // ============================================================

    private static Dataset loadDataset(String filePath) {

        List<float[]> featureRows = new ArrayList<>();
        List<Float> labelRows = new ArrayList<>();

        try (BufferedReader reader =
                     new BufferedReader(
                             new FileReader(filePath))) {

            String header = reader.readLine();

            if (header == null) {
                throw new RuntimeException(
                        "Dataset is empty: " + filePath
                );
            }

            String[] headerColumns =
                    header.split(",", -1);

            System.out.println();
            System.out.println("Reading: " + filePath);
            System.out.println(
                    "CSV columns: " + headerColumns.length
            );

            // ------------------------------------------
            // FIND IMPORTANT COLUMNS
            // ------------------------------------------

            int subjectIndex = -1;
            int sequenceIndex = -1;
            int labelIndex = -1;

            for (int i = 0;
                 i < headerColumns.length;
                 i++) {

                String column =
                        headerColumns[i].trim();

                if (column.equalsIgnoreCase("subject")) {
                    subjectIndex = i;
                }

                if (column.equalsIgnoreCase("sequence_id")) {
                    sequenceIndex = i;
                }

                if (column.equalsIgnoreCase("fatigue_label")) {
                    labelIndex = i;
                }
            }

            if (subjectIndex == -1) {
                throw new RuntimeException(
                        "subject column not found."
                );
            }

            if (sequenceIndex == -1) {
                throw new RuntimeException(
                        "sequence_id column not found."
                );
            }

            if (labelIndex == -1) {
                throw new RuntimeException(
                        "fatigue_label column not found."
                );
            }

            System.out.println(
                    "Subject column index : "
                            + subjectIndex
            );

            System.out.println(
                    "Sequence column index: "
                            + sequenceIndex
            );

            System.out.println(
                    "Label column index   : "
                            + labelIndex
            );


            // ------------------------------------------
            // READ DATA
            // ------------------------------------------

            String line;

            int rowNumber = 1;

            while ((line = reader.readLine()) != null) {

                rowNumber++;

                if (line.trim().isEmpty()) {
                    continue;
                }

                String[] values =
                        line.split(",", -1);

                if (values.length !=
                        headerColumns.length) {

                    throw new RuntimeException(
                            "Invalid column count at row "
                                    + rowNumber
                                    + ". Expected "
                                    + headerColumns.length
                                    + " but found "
                                    + values.length
                    );
                }


                // --------------------------------------
                // EXTRACT 200 FEATURES
                // --------------------------------------

                float[] features =
                        new float[TOTAL_FEATURES];

                int featurePosition = 0;

                for (int i = 0;
                     i < values.length;
                     i++) {

                    // Skip metadata
                    if (i == subjectIndex ||
                            i == sequenceIndex ||
                            i == labelIndex) {

                        continue;
                    }

                    if (featurePosition >=
                            TOTAL_FEATURES) {

                        break;
                    }

                    String value =
                            values[i].trim();

                    try {

                        features[featurePosition] =
                                Float.parseFloat(value);

                    } catch (NumberFormatException e) {

                        throw new RuntimeException(
                                "Invalid numeric value at row "
                                        + rowNumber
                                        + ", column "
                                        + headerColumns[i]
                                        + ": "
                                        + value
                        );
                    }

                    featurePosition++;
                }


                // --------------------------------------
                // VERIFY 200 FEATURES
                // --------------------------------------

                if (featurePosition !=
                        TOTAL_FEATURES) {

                    throw new RuntimeException(
                            "Expected "
                                    + TOTAL_FEATURES
                                    + " features but found "
                                    + featurePosition
                                    + " at row "
                                    + rowNumber
                    );
                }


                // --------------------------------------
                // FATIGUE LABEL
                // --------------------------------------

                String labelValue =
                        values[labelIndex].trim();

                float label;

                if (labelValue.equals("0")) {

                    label = 0.0f;

                } else if (labelValue.equals("1")) {

                    label = 1.0f;

                } else {

                    throw new RuntimeException(
                            "Invalid fatigue label at row "
                                    + rowNumber
                                    + ": "
                                    + labelValue
                    );
                }


                featureRows.add(features);
                labelRows.add(label);
            }

        } catch (IOException e) {

            throw new RuntimeException(
                    "Could not read dataset: "
                            + filePath,
                    e
            );
        }


        // ------------------------------------------
        // CONVERT LISTS TO ARRAYS
        // ------------------------------------------

        float[][] features =
                featureRows.toArray(
                        new float[0][]
                );

        float[] labels =
                new float[labelRows.size()];

        for (int i = 0;
             i < labelRows.size();
             i++) {

            labels[i] =
                    labelRows.get(i);
        }

        return new Dataset(
                features,
                labels
        );
    }


    // ============================================================
    // PRINT DATASET INFORMATION
    // ============================================================

    private static void printDatasetInfo(
            String name,
            Dataset dataset) {

        System.out.println();
        System.out.println(
                name + " samples : "
                        + dataset.features.length
        );

        System.out.println(
                name + " features: "
                        + TOTAL_FEATURES
        );

        System.out.println(
                name + " labels  : "
                        + dataset.labels.length
        );
    }


    // ============================================================
    // LABEL DISTRIBUTION
    // ============================================================

    private static void printLabelDistribution(
            String name,
            float[] labels) {

        int nonFatigue = 0;
        int fatigue = 0;

        for (float label : labels) {

            if (label == 0.0f) {
                nonFatigue++;
            }

            else if (label == 1.0f) {
                fatigue++;
            }
        }

        System.out.println();
        System.out.println(
                name + " LABEL DISTRIBUTION"
        );

        System.out.println(
                "Non-Fatigue (0): "
                        + nonFatigue
        );

        System.out.println(
                "Fatigue (1)    : "
                        + fatigue
        );
    }


    // ============================================================
    // DATASET CLASS
    // ============================================================

    private static class Dataset {

        float[][] features;
        float[] labels;

        Dataset(
                float[][] features,
                float[] labels) {

            this.features = features;
            this.labels = labels;
        }
    }
}