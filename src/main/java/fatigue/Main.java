package fatigue;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

public class Main {

    // =========================================================
    // FILE PATHS
    // =========================================================

    private static final String BF_INPUT =
            "data/raw/BicepsFemoris_all_subjects.csv";

    private static final String HAM_INPUT =
            "data/raw/Hamstring_all_subjects.csv";

    private static final String BF_CLEANED =
            "data/processed/cleaned_BF_features.csv";

    private static final String BF_NORMALIZED =
            "data/processed/normalized_BF_features.csv";

    private static final String HAM_CLEANED =
            "data/processed/cleaned_Hamstring_features.csv";

    private static final String HAM_NORMALIZED =
            "data/processed/normalized_Hamstring_features.csv";

    private static final String COMBINED =
            "data/processed/combined_BF_Hamstring_features.csv";

    private static final String TEMPORAL =
            "data/processed/temporal_sequences.csv";

    private static final String LABELED =
            "data/processed/labeled_temporal_sequences.csv";

    // 5 BF + 5 Hamstring features
    private static final int FEATURES_PER_STEP = 10;

    // Temporal window
    private static final int SEQUENCE_LENGTH = 20;


    // =========================================================
    // MAIN
    // =========================================================

    public static void main(String[] args) {

        System.out.println();
        System.out.println("======================================");
        System.out.println("   HAMSTRING FATIGUE ML PIPELINE");
        System.out.println("======================================");

        try {

            Files.createDirectories(Paths.get("data/processed"));

            // =================================================
            // STEP 1 - BICEPS FEMORIS
            // =================================================

            System.out.println();
            System.out.println("======================================");
            System.out.println(" STEP 1: BICEPS FEMORIS PREPROCESSING");
            System.out.println("======================================");

            System.out.println();
            System.out.println("Reading BF dataset...");

            List<String[]> bfRaw = readCSV(BF_INPUT);

            validateDataset(bfRaw, "Biceps Femoris");

            System.out.println();
            System.out.println("Cleaning BF dataset...");

            List<String[]> bfCleaned = cleanDataset(bfRaw);

            saveCSV(bfCleaned, BF_CLEANED);

            System.out.println();
            System.out.println("Normalizing BF features...");

            List<String[]> bfNormalized =
                    normalizeFeatures(bfCleaned);

            saveCSV(bfNormalized, BF_NORMALIZED);

            System.out.println();
            System.out.println("BF preprocessing completed.");

            System.out.println(
                    "Cleaned BF file: " + BF_CLEANED);

            System.out.println(
                    "Normalized BF file: " + BF_NORMALIZED);


            // =================================================
            // STEP 2 - HAMSTRING
            // =================================================

            System.out.println();
            System.out.println("======================================");
            System.out.println(" STEP 2: HAMSTRING PREPROCESSING");
            System.out.println("======================================");

            System.out.println();
            System.out.println("Reading Hamstring dataset...");

            List<String[]> hamRaw = readCSV(HAM_INPUT);

            validateDataset(hamRaw, "Hamstring");

            System.out.println();
            System.out.println("Cleaning Hamstring dataset...");

            List<String[]> hamCleaned =
                    cleanDataset(hamRaw);

            saveCSV(hamCleaned, HAM_CLEANED);

            System.out.println();
            System.out.println("Normalizing Hamstring features...");

            List<String[]> hamNormalized =
                    normalizeFeatures(hamCleaned);

            saveCSV(hamNormalized, HAM_NORMALIZED);

            System.out.println();
            System.out.println("Hamstring preprocessing completed.");

            System.out.println(
                    "Cleaned Hamstring file: " + HAM_CLEANED);

            System.out.println(
                    "Normalized Hamstring file: " + HAM_NORMALIZED);


            // =================================================
            // STEP 3 - FEATURE FUSION
            // =================================================

            System.out.println();
            System.out.println("======================================");
            System.out.println(" STEP 3: FEATURE FUSION");
            System.out.println("======================================");

            System.out.println();
            System.out.println(
                    "Combining Biceps Femoris + Hamstring features...");

            List<String[]> combined =
                    fuseFeatures(
                            bfNormalized,
                            hamNormalized
                    );

            saveCSV(combined, COMBINED);

            System.out.println();
            System.out.println(
                    "Combined dataset saved to:");

            System.out.println(COMBINED);

            System.out.println();
            System.out.println(
                    "Features per time step: "
                            + FEATURES_PER_STEP);


            // =================================================
            // STEP 4 - TEMPORAL SEQUENCES
            // =================================================

            System.out.println();
            System.out.println("======================================");
            System.out.println(" STEP 4: TEMPORAL SEQUENCE GENERATION");
            System.out.println("======================================");

            System.out.println();
            System.out.println(
                    "Creating "
                            + SEQUENCE_LENGTH
                            + "-step temporal sequences...");

            List<String[]> sequences =
                    generateSequences(combined);

            saveCSV(sequences, TEMPORAL);

            int sequenceCount = sequences.size() - 1;

            System.out.println();
            System.out.println(
                    "Total temporal sequences: "
                            + sequenceCount);

            System.out.println();
            System.out.println("Output file:");
            System.out.println(TEMPORAL);

            System.out.println();
            System.out.println("Expected ML input shape:");
            System.out.println(
                    "("
                            + sequenceCount
                            + ", "
                            + SEQUENCE_LENGTH
                            + ", "
                            + FEATURES_PER_STEP
                            + ")");


            // =================================================
            // STEP 5 - FATIGUE LABELING
            // =================================================

            System.out.println();
            System.out.println("======================================");
            System.out.println(" STEP 5: FATIGUE LABELING");
            System.out.println("======================================");

            List<String[]> labeled =
                    generateFatigueLabels(sequences);

            saveCSV(labeled, LABELED);

            int nonFatigue = 0;
            int fatigue = 0;

            for (int i = 1; i < labeled.size(); i++) {

                String label =
                        labeled.get(i)[labeled.get(i).length - 1];

                if ("1".equals(label)) {
                    fatigue++;
                } else {
                    nonFatigue++;
                }
            }

            System.out.println();
            System.out.println(
                    "Non-Fatigue samples : " + nonFatigue);

            System.out.println(
                    "Fatigue samples     : " + fatigue);

            System.out.println(
                    "Total labeled       : "
                            + (labeled.size() - 1));

            System.out.println();
            System.out.println(
                    "Labeled dataset: " + LABELED);


            // =================================================
            // COMPLETE
            // =================================================

            System.out.println();
            System.out.println("======================================");
            System.out.println("      COMPLETE PIPELINE FINISHED");
            System.out.println("======================================");

            System.out.println();
            System.out.println("Pipeline stages completed:");

            System.out.println(
                    "1. All-subject BF Dataset Reading");

            System.out.println(
                    "2. All-subject Hamstring Dataset Reading");

            System.out.println(
                    "3. Data Validation");

            System.out.println(
                    "4. Data Cleaning");

            System.out.println(
                    "5. Feature Normalization");

            System.out.println(
                    "6. BF + Hamstring Feature Fusion");

            System.out.println(
                    "7. Subject-wise Temporal Sequence Generation");

            System.out.println(
                    "8. Provisional Fatigue Labeling");

            System.out.println();
            System.out.println("Generated files:");

            System.out.println(BF_CLEANED);
            System.out.println(BF_NORMALIZED);
            System.out.println(HAM_CLEANED);
            System.out.println(HAM_NORMALIZED);
            System.out.println(COMBINED);
            System.out.println(TEMPORAL);
            System.out.println(LABELED);

            System.out.println();
            System.out.println("======================================");
            System.out.println("       READY FOR ML MODEL");
            System.out.println("======================================");

        } catch (Exception e) {

            System.out.println();
            System.out.println("======================================");
            System.out.println("          PIPELINE ERROR");
            System.out.println("======================================");

            System.out.println();
            System.out.println(
                    "Error during processing:");

            e.printStackTrace();
        }
    }


    // =========================================================
    // READ CSV
    // =========================================================

    private static List<String[]> readCSV(String file)
            throws IOException {

        List<String[]> rows = new ArrayList<>();

        try (BufferedReader br =
                     Files.newBufferedReader(
                             Paths.get(file))) {

            String line;

            while ((line = br.readLine()) != null) {

                if (!line.trim().isEmpty()) {

                    rows.add(line.split(",", -1));
                }
            }
        }

        return rows;
    }


    // =========================================================
    // VALIDATE DATASET
    // =========================================================

    private static void validateDataset(
            List<String[]> data,
            String name) {

        System.out.println();
        System.out.println("======================================");
        System.out.println("          DATA VALIDATION");
        System.out.println("======================================");

        if (data.isEmpty()) {

            throw new IllegalArgumentException(
                    name + " dataset is empty.");
        }

        String[] header = data.get(0);

        System.out.println(
                "Total rows    : "
                        + (data.size() - 1));

        System.out.println(
                "Total columns : "
                        + header.length);

        if (header.length != 12) {

            throw new IllegalArgumentException(
                    name
                            + " dataset must contain 12 columns.");
        }

        int valid = 0;
        int missing = 0;
        int invalid = 0;

        for (int i = 1; i < data.size(); i++) {

            String[] row = data.get(i);

            boolean rowValid = true;

            if (row.length != 12) {

                invalid++;
                continue;
            }

            for (int j = 0; j < row.length; j++) {

                if (row[j] == null
                        || row[j].trim().isEmpty()) {

                    missing++;
                    rowValid = false;
                }
            }

            // Numeric columns
            for (int j = 2; j < 12; j++) {

                try {

                    Double.parseDouble(row[j]);

                } catch (Exception e) {

                    invalid++;
                    rowValid = false;
                    break;
                }
            }

            if (rowValid) {
                valid++;
            }
        }

        System.out.println(
                "Valid rows     : " + valid);

        System.out.println(
                "Missing values : " + missing);

        System.out.println(
                "Invalid values : " + invalid);

        System.out.println();
        System.out.println(
                "----------- DATASET CHECK -----------");

        Set<String> subjects =
                new LinkedHashSet<>();

        for (int i = 1; i < data.size(); i++) {

            if (data.get(i).length >= 1) {

                subjects.add(data.get(i)[0]);
            }
        }

        System.out.println(
                "Subjects found : "
                        + subjects.size());

        System.out.println(
                "Subject IDs    : "
                        + subjects);

        System.out.println(
                "--------------------------------------");
    }


    // =========================================================
    // CLEAN DATASET
    // =========================================================

    private static List<String[]> cleanDataset(
            List<String[]> data) {

        List<String[]> cleaned =
                new ArrayList<>();

        cleaned.add(data.get(0));

        int removed = 0;

        for (int i = 1; i < data.size(); i++) {

            String[] row = data.get(i);

            if (row.length != 12) {

                removed++;
                continue;
            }

            boolean valid = true;

            // Required text columns
            if (row[0].trim().isEmpty()
                    || row[1].trim().isEmpty()) {

                valid = false;
            }

            // Numeric columns
            for (int j = 2; j < 12; j++) {

                try {

                    double value =
                            Double.parseDouble(
                                    row[j].trim());

                    if (Double.isNaN(value)
                            || Double.isInfinite(value)) {

                        valid = false;
                        break;
                    }

                } catch (Exception e) {

                    valid = false;
                    break;
                }
            }

            if (valid) {

                cleaned.add(row);

            } else {

                removed++;
            }
        }

        System.out.println();
        System.out.println("======================================");
        System.out.println("           DATA CLEANING");
        System.out.println("======================================");

        System.out.println(
                "Original rows : "
                        + (data.size() - 1));

        System.out.println(
                "Removed rows  : "
                        + removed);

        System.out.println(
                "Cleaned rows  : "
                        + (cleaned.size() - 1));

        return cleaned;
    }


    // =========================================================
    // NORMALIZATION
    //
    // Features:
    // RMS  -> column 7
    // MAV  -> column 8
    // VAR  -> column 9
    // MNF  -> column 10
    // MDF  -> column 11
    // =========================================================

    private static List<String[]> normalizeFeatures(
            List<String[]> data) {

        int[] featureColumns =
                {7, 8, 9, 10, 11};

        double[] means =
                new double[featureColumns.length];

        double[] stds =
                new double[featureColumns.length];

        int n = data.size() - 1;

        // -----------------------------------------------------
        // MEAN
        // -----------------------------------------------------

        for (int i = 1; i < data.size(); i++) {

            String[] row = data.get(i);

            for (int j = 0;
                 j < featureColumns.length;
                 j++) {

                means[j] +=
                        Double.parseDouble(
                                row[featureColumns[j]]);
            }
        }

        for (int j = 0;
             j < means.length;
             j++) {

            means[j] /= n;
        }

        // -----------------------------------------------------
        // STANDARD DEVIATION
        // -----------------------------------------------------

        for (int i = 1; i < data.size(); i++) {

            String[] row = data.get(i);

            for (int j = 0;
                 j < featureColumns.length;
                 j++) {

                double value =
                        Double.parseDouble(
                                row[featureColumns[j]]);

                double diff =
                        value - means[j];

                stds[j] += diff * diff;
            }
        }

        for (int j = 0;
             j < stds.length;
             j++) {

            stds[j] =
                    Math.sqrt(stds[j] / n);

            if (stds[j] == 0) {
                stds[j] = 1.0;
            }
        }

        // -----------------------------------------------------
        // CREATE NORMALIZED DATA
        // -----------------------------------------------------

        List<String[]> normalized =
                new ArrayList<>();

        String[] newHeader =
                Arrays.copyOf(
                        data.get(0),
                        17);

        newHeader[12] = "RMS_Z";
        newHeader[13] = "MAV_Z";
        newHeader[14] = "VAR_Z";
        newHeader[15] = "MNF_Z";
        newHeader[16] = "MDF_Z";

        normalized.add(newHeader);

        for (int i = 1; i < data.size(); i++) {

            String[] oldRow = data.get(i);

            String[] newRow =
                    Arrays.copyOf(oldRow, 17);

            for (int j = 0;
                 j < featureColumns.length;
                 j++) {

                double value =
                        Double.parseDouble(
                                oldRow[
                                        featureColumns[j]]);

                double z =
                        (value - means[j])
                                / stds[j];

                newRow[12 + j] =
                        Double.toString(z);
            }

            normalized.add(newRow);
        }

        System.out.println();
        System.out.println("======================================");
        System.out.println("       NORMALIZATION STATISTICS");
        System.out.println("======================================");

        System.out.printf(
                Locale.US,
                "RMS  -> Mean: %.10e | Std: %.10e%n",
                means[0], stds[0]);

        System.out.printf(
                Locale.US,
                "MAV  -> Mean: %.10e | Std: %.10e%n",
                means[1], stds[1]);

        System.out.printf(
                Locale.US,
                "VAR  -> Mean: %.10e | Std: %.10e%n",
                means[2], stds[2]);

        System.out.printf(
                Locale.US,
                "MNF  -> Mean: %.6f | Std: %.6f%n",
                means[3], stds[3]);

        System.out.printf(
                Locale.US,
                "MDF  -> Mean: %.6f | Std: %.6f%n",
                means[4], stds[4]);

        return normalized;
    }


    // =========================================================
    // FEATURE FUSION
    // =========================================================

    private static List<String[]> fuseFeatures(
            List<String[]> bf,
            List<String[]> ham) {

        System.out.println();
        System.out.println("======================================");
        System.out.println("       BF + HAMSTRING FUSION");
        System.out.println("======================================");

        // -----------------------------------------------------
        // Header
        // -----------------------------------------------------

        List<String[]> result =
                new ArrayList<>();

        result.add(new String[]{
                "subject",
                "window_id",
                "start_time_s",
                "end_time_s",
                "BF_RMS",
                "BF_MAV",
                "BF_VAR",
                "BF_MNF",
                "BF_MDF",
                "HAM_RMS",
                "HAM_MAV",
                "HAM_VAR",
                "HAM_MNF",
                "HAM_MDF"
        });

        // -----------------------------------------------------
        // Create Hamstring lookup
        // key = subject + window
        // -----------------------------------------------------

        Map<String, String[]> hamMap =
                new HashMap<>();

        for (int i = 1; i < ham.size(); i++) {

            String[] row = ham.get(i);

            String key =
                    row[0] + "_" + row[3];

            hamMap.put(key, row);
        }

        int matched = 0;
        int unmatched = 0;

        // -----------------------------------------------------
        // Match BF with Hamstring
        // -----------------------------------------------------

        for (int i = 1; i < bf.size(); i++) {

            String[] bfRow = bf.get(i);

            String key =
                    bfRow[0] + "_" + bfRow[3];

            String[] hamRow =
                    hamMap.get(key);

            if (hamRow == null) {

                unmatched++;
                continue;
            }

            String[] combined = new String[14];

            combined[0] = bfRow[0];
            combined[1] = bfRow[3];
            combined[2] = bfRow[4];
            combined[3] = bfRow[5];

            // BF normalized features
            combined[4] = bfRow[12];
            combined[5] = bfRow[13];
            combined[6] = bfRow[14];
            combined[7] = bfRow[15];
            combined[8] = bfRow[16];

            // Hamstring normalized features
            combined[9] = hamRow[12];
            combined[10] = hamRow[13];
            combined[11] = hamRow[14];
            combined[12] = hamRow[15];
            combined[13] = hamRow[16];

            result.add(combined);

            matched++;
        }

        System.out.println();
        System.out.println(
                "BF rows        : "
                        + (bf.size() - 1));

        System.out.println(
                "Hamstring rows : "
                        + (ham.size() - 1));

        System.out.println(
                "Matched windows: "
                        + matched);

        System.out.println(
                "Unmatched BF   : "
                        + unmatched);

        System.out.println(
                "Combined rows  : "
                        + (result.size() - 1));

        return result;
    }


    // =========================================================
    // TEMPORAL SEQUENCE GENERATION
    //
    // VERY IMPORTANT:
    // Sequences are generated separately for each subject.
    // =========================================================

    private static List<String[]> generateSequences(
            List<String[]> combined) {

        List<String[]> result =
                new ArrayList<>();

        // -----------------------------------------------------
        // HEADER
        // -----------------------------------------------------

        List<String> header =
                new ArrayList<>();

        header.add("subject");
        header.add("sequence_id");

        for (int t = 1;
             t <= SEQUENCE_LENGTH;
             t++) {

            for (int f = 1;
                 f <= FEATURES_PER_STEP;
                 f++) {

                header.add(
                        "t"
                                + t
                                + "_f"
                                + f);
            }
        }

        result.add(
                header.toArray(
                        new String[0]));

        // -----------------------------------------------------
        // Group rows by subject
        // -----------------------------------------------------

        Map<String, List<String[]>> subjectData =
                new LinkedHashMap<>();

        for (int i = 1;
             i < combined.size();
             i++) {

            String[] row = combined.get(i);

            subjectData
                    .computeIfAbsent(
                            row[0],
                            k -> new ArrayList<>())
                    .add(row);
        }

        System.out.println();
        System.out.println("======================================");
        System.out.println("     TEMPORAL SEQUENCE GENERATION");
        System.out.println("======================================");

        System.out.println();
        System.out.println(
                "Subjects         : "
                        + subjectData.size());

        System.out.println(
                "Sequence length  : "
                        + SEQUENCE_LENGTH);

        System.out.println(
                "Features/step    : "
                        + FEATURES_PER_STEP);

        int totalSequences = 0;

        // -----------------------------------------------------
        // Generate sequences independently for each subject
        // -----------------------------------------------------

        for (Map.Entry<String,
                List<String[]>> entry
                : subjectData.entrySet()) {

            String subject =
                    entry.getKey();

            List<String[]> rows =
                    entry.getValue();

            // Sort by window ID
            rows.sort(
                    Comparator.comparingInt(
                            r -> Integer.parseInt(r[1])
                    )
            );

            int subjectSequences =
                    rows.size()
                            - SEQUENCE_LENGTH
                            + 1;

            if (subjectSequences < 0) {
                subjectSequences = 0;
            }

            System.out.println(
                    subject
                            + " -> "
                            + rows.size()
                            + " windows -> "
                            + subjectSequences
                            + " sequences");

            for (int start = 0;
                 start < subjectSequences;
                 start++) {

                List<String> sequence =
                        new ArrayList<>();

                sequence.add(subject);

                sequence.add(
                        Integer.toString(
                                start + 1));

                // 20 time steps
                for (int t = 0;
                     t < SEQUENCE_LENGTH;
                     t++) {

                    String[] row =
                            rows.get(start + t);

                    // Combined feature columns:
                    // 4 through 13
                    for (int f = 4;
                         f <= 13;
                         f++) {

                        sequence.add(row[f]);
                    }
                }

                result.add(
                        sequence.toArray(
                                new String[0]));

                totalSequences++;
            }
        }

        System.out.println();
        System.out.println(
                "Total sequences  : "
                        + totalSequences);

        System.out.println();
        System.out.println(
                "Expected ML input shape:");

        System.out.println(
                "("
                        + totalSequences
                        + ", "
                        + SEQUENCE_LENGTH
                        + ", "
                        + FEATURES_PER_STEP
                        + ")");

        return result;
    }


    // =========================================================
    // FATIGUE LABELING
    //
    // This is a PROVISIONAL label for the synthetic dataset.
    //
    // A sequence is labelled:
    // 0 = Non-Fatigue
    // 1 = Fatigue
    //
    // based on temporal progression.
    // =========================================================

    private static List<String[]> generateFatigueLabels(
            List<String[]> sequences) {

        List<String[]> labeled =
                new ArrayList<>();

        String[] oldHeader =
                sequences.get(0);

        String[] newHeader =
                Arrays.copyOf(
                        oldHeader,
                        oldHeader.length + 1);

        newHeader[
                newHeader.length - 1] =
                "fatigue_label";

        labeled.add(newHeader);

        int featureStart = 2;

        // Number of feature values:
        // 20 × 10 = 200
        int featureCount =
                SEQUENCE_LENGTH
                        * FEATURES_PER_STEP;

        int nonFatigue = 0;
        int fatigue = 0;

        for (int i = 1;
             i < sequences.size();
             i++) {

            String[] row =
                    sequences.get(i);

            String[] newRow =
                    Arrays.copyOf(
                            row,
                            row.length + 1);

            /*
             * The last time step contains the final
             * 10 normalized features.
             *
             * We use the temporal sequence's final
             * position to create a provisional label.
             *
             * For this synthetic dataset:
             * later recording progression = fatigue.
             */

            int finalStepStart =
                    featureStart
                            + ((SEQUENCE_LENGTH - 1)
                            * FEATURES_PER_STEP);

            double rms =
                    Double.parseDouble(
                            row[finalStepStart]);

            double mav =
                    Double.parseDouble(
                            row[finalStepStart + 1]);

            double var =
                    Double.parseDouble(
                            row[finalStepStart + 2]);

            /*
             * Provisional fatigue score.
             *
             * Higher muscle activation features are
             * considered as an indicator for the
             * synthetic fatigue class.
             */
            double score =
                    (rms + mav + var) / 3.0;

            int label;

            if (score >= 0.5) {
                label = 1;
                fatigue++;
            } else {
                label = 0;
                nonFatigue++;
            }

            newRow[
                    newRow.length - 1] =
                    Integer.toString(label);

            labeled.add(newRow);
        }

        System.out.println();
        System.out.println("======================================");
        System.out.println("       FATIGUE LABEL GENERATION");
        System.out.println("======================================");

        System.out.println();
        System.out.println(
                "Non-Fatigue samples : "
                        + nonFatigue);

        System.out.println(
                "Fatigue samples     : "
                        + fatigue);

        System.out.println(
                "Total labeled       : "
                        + (labeled.size() - 1));

        return labeled;
    }


    // =========================================================
    // SAVE CSV
    // =========================================================

    private static void saveCSV(
            List<String[]> data,
            String file)
            throws IOException {

        Path path =
                Paths.get(file);

        if (path.getParent() != null) {

            Files.createDirectories(
                    path.getParent());
        }

        try (BufferedWriter writer =
                     Files.newBufferedWriter(path)) {

            for (String[] row : data) {

                writer.write(
                        String.join(",", row));

                writer.newLine();
            }
        }

        System.out.println();
        System.out.println(
                "Dataset saved to:");

        System.out.println(file);
    }
}