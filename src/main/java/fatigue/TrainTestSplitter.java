package fatigue;

import java.io.*;
import java.nio.file.*;
import java.util.*;

public class TrainTestSplitter {

    private static final String INPUT_FILE =
            "data/processed/labeled_temporal_sequences.csv";

    private static final String OUTPUT_DIR =
            "data/processed/split/";

    public static void main(String[] args) {

        System.out.println("======================================");
        System.out.println("       SUBJECT-WISE DATASET SPLIT");
        System.out.println("======================================");

        // -------------------------------
        // SUBJECT-WISE SPLIT
        // -------------------------------

        Set<String> trainSubjects = new LinkedHashSet<>(
                Arrays.asList(
                        "S01", "S02", "S03", "S04",
                        "S05", "S06", "S07", "S08",
                        "S09", "S10", "S11", "S12",
                        "S13", "S14"
                )
        );

        Set<String> validationSubjects = new LinkedHashSet<>(
                Arrays.asList(
                        "S15", "S16", "S17"
                )
        );

        Set<String> testSubjects = new LinkedHashSet<>(
                Arrays.asList(
                        "S18", "S19", "S20"
                )
        );

        System.out.println();
        System.out.println("Training subjects   : " + trainSubjects);
        System.out.println("Validation subjects : " + validationSubjects);
        System.out.println("Testing subjects    : " + testSubjects);

        // Create output directory
        try {
            Files.createDirectories(Paths.get(OUTPUT_DIR));
        } catch (IOException e) {
            System.err.println("Could not create output directory.");
            e.printStackTrace();
            return;
        }

        // Output files
        String trainFile = OUTPUT_DIR + "train.csv";
        String validationFile = OUTPUT_DIR + "validation.csv";
        String testFile = OUTPUT_DIR + "test.csv";

        int trainCount = 0;
        int validationCount = 0;
        int testCount = 0;
        int unknownCount = 0;

        try (
                BufferedReader reader =
                        new BufferedReader(new FileReader(INPUT_FILE));

                BufferedWriter trainWriter =
                        new BufferedWriter(new FileWriter(trainFile));

                BufferedWriter validationWriter =
                        new BufferedWriter(new FileWriter(validationFile));

                BufferedWriter testWriter =
                        new BufferedWriter(new FileWriter(testFile))
        ) {

            // --------------------------------
            // READ HEADER
            // --------------------------------

            String header = reader.readLine();

            if (header == null) {
                System.err.println("Input CSV is empty.");
                return;
            }

            // Write header to all files
            trainWriter.write(header);
            trainWriter.newLine();

            validationWriter.write(header);
            validationWriter.newLine();

            testWriter.write(header);
            testWriter.newLine();

            // --------------------------------
            // PROCESS DATA ROWS
            // --------------------------------

            String line;

            while ((line = reader.readLine()) != null) {

                if (line.trim().isEmpty()) {
                    continue;
                }

                // Subject is the first column
                String[] columns = line.split(",", -1);

                if (columns.length == 0) {
                    continue;
                }

                String subject = columns[0].trim();

                // ----------------------------
                // TRAINING
                // ----------------------------

                if (trainSubjects.contains(subject)) {

                    trainWriter.write(line);
                    trainWriter.newLine();

                    trainCount++;
                }

                // ----------------------------
                // VALIDATION
                // ----------------------------

                else if (validationSubjects.contains(subject)) {

                    validationWriter.write(line);
                    validationWriter.newLine();

                    validationCount++;
                }

                // ----------------------------
                // TESTING
                // ----------------------------

                else if (testSubjects.contains(subject)) {

                    testWriter.write(line);
                    testWriter.newLine();

                    testCount++;
                }

                // ----------------------------
                // UNKNOWN SUBJECT
                // ----------------------------

                else {

                    unknownCount++;
                    System.out.println(
                            "Unknown subject found: " + subject
                    );
                }
            }

        } catch (IOException e) {

            System.err.println("Error processing dataset.");
            e.printStackTrace();
            return;
        }

        // --------------------------------
        // DISPLAY RESULTS
        // --------------------------------

        System.out.println();
        System.out.println("--------------------------------------");
        System.out.println("             SPLIT RESULTS");
        System.out.println("--------------------------------------");

        System.out.println(
                "Training samples   : " + trainCount
        );

        System.out.println(
                "Validation samples : " + validationCount
        );

        System.out.println(
                "Testing samples    : " + testCount
        );

        System.out.println(
                "Unknown samples    : " + unknownCount
        );

        System.out.println();
        System.out.println("Files generated:");

        System.out.println(
                "data/processed/split/train.csv"
        );

        System.out.println(
                "data/processed/split/validation.csv"
        );

        System.out.println(
                "data/processed/split/test.csv"
        );

        System.out.println();
        System.out.println("======================================");
        System.out.println("       DATASET SPLIT COMPLETED");
        System.out.println("======================================");
    }
}