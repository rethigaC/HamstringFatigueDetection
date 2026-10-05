package fatigue;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Splits the labeled temporal dataset subject-wise.
 *
 * Input:
 * data/processed/labeled_temporal_sequences.csv
 *
 * Output:
 * data/processed/train.csv
 * data/processed/validation.csv
 * data/processed/test.csv
 *
 * Dataset:
 * 20 subjects
 * 20 time steps
 * 10 features per time step
 */
public class DatasetSplitter {

    private static final String INPUT_FILE =
            "data/processed/labeled_temporal_sequences.csv";

    private static final String TRAIN_FILE =
            "data/processed/train.csv";

    private static final String VALIDATION_FILE =
            "data/processed/validation.csv";

    private static final String TEST_FILE =
            "data/processed/test.csv";

    public static void main(String[] args) {

        System.out.println();
        System.out.println("======================================");
        System.out.println("      SUBJECT-WISE DATASET SPLITTER");
        System.out.println("======================================");

        try {

            Path inputPath = Paths.get(INPUT_FILE);

            if (!Files.exists(inputPath)) {
                System.out.println();
                System.out.println("ERROR: Input file not found:");
                System.out.println(INPUT_FILE);
                return;
            }

            System.out.println();
            System.out.println("Reading:");
            System.out.println(INPUT_FILE);

            List<String> lines = Files.readAllLines(inputPath);

            if (lines.size() < 2) {
                System.out.println("ERROR: Dataset is empty.");
                return;
            }

            String header = lines.get(0);

            /*
             * Store rows according to subject.
             */
            Map<String, List<String>> subjectData =
                    new LinkedHashMap<>();

            for (int i = 1; i < lines.size(); i++) {

                String line = lines.get(i).trim();

                if (line.isEmpty()) {
                    continue;
                }

                String[] parts = line.split(",", -1);

                if (parts.length < 3) {
                    continue;
                }

                String subject = parts[0].trim();

                subjectData
                        .computeIfAbsent(subject,
                                k -> new ArrayList<>())
                        .add(line);
            }

            System.out.println();
            System.out.println("======================================");
            System.out.println("          DATASET INFORMATION");
            System.out.println("======================================");

            System.out.println(
                    "Total rows     : " + (lines.size() - 1)
            );

            System.out.println(
                    "Subjects found : " + subjectData.size()
            );

            System.out.println(
                    "Subjects       : " + subjectData.keySet()
            );

            /*
             * Sort subjects naturally:
             * S01, S02, ..., S20
             */
            List<String> subjects =
                    new ArrayList<>(subjectData.keySet());

            subjects.sort(Comparator.comparingInt(
                    DatasetSplitter::subjectNumber
            ));

            /*
             * Subject-wise split.
             *
             * S01-S14 -> Training
             * S15-S17 -> Validation
             * S18-S20 -> Testing
             */
            List<String> trainSubjects = new ArrayList<>();
            List<String> validationSubjects = new ArrayList<>();
            List<String> testSubjects = new ArrayList<>();

            for (int i = 0; i < subjects.size(); i++) {

                String subject = subjects.get(i);

                if (i < 14) {
                    trainSubjects.add(subject);
                }
                else if (i < 17) {
                    validationSubjects.add(subject);
                }
                else {
                    testSubjects.add(subject);
                }
            }

            /*
             * Prepare output directories.
             */
            Files.createDirectories(
                    Paths.get("data/processed")
            );

            /*
             * Write datasets.
             */
            writeDataset(
                    TRAIN_FILE,
                    header,
                    trainSubjects,
                    subjectData
            );

            writeDataset(
                    VALIDATION_FILE,
                    header,
                    validationSubjects,
                    subjectData
            );

            writeDataset(
                    TEST_FILE,
                    header,
                    testSubjects,
                    subjectData
            );

            /*
             * Count rows.
             */
            int trainRows =
                    countRows(trainSubjects, subjectData);

            int validationRows =
                    countRows(validationSubjects, subjectData);

            int testRows =
                    countRows(testSubjects, subjectData);

            /*
             * Print summary.
             */
            System.out.println();
            System.out.println("======================================");
            System.out.println("          SUBJECT-WISE SPLIT");
            System.out.println("======================================");

            System.out.println();
            System.out.println(
                    "TRAINING SUBJECTS     : "
                            + trainSubjects
            );

            System.out.println(
                    "TRAINING ROWS         : "
                            + trainRows
            );

            System.out.println();
            System.out.println(
                    "VALIDATION SUBJECTS   : "
                            + validationSubjects
            );

            System.out.println(
                    "VALIDATION ROWS       : "
                            + validationRows
            );

            System.out.println();
            System.out.println(
                    "TEST SUBJECTS         : "
                            + testSubjects
            );

            System.out.println(
                    "TEST ROWS             : "
                            + testRows
            );

            System.out.println();
            System.out.println("======================================");
            System.out.println("          OUTPUT FILES");
            System.out.println("======================================");

            System.out.println();
            System.out.println(
                    "Training dataset:");
            System.out.println(TRAIN_FILE);

            System.out.println();
            System.out.println(
                    "Validation dataset:");
            System.out.println(VALIDATION_FILE);

            System.out.println();
            System.out.println(
                    "Testing dataset:");
            System.out.println(TEST_FILE);

            System.out.println();
            System.out.println("======================================");
            System.out.println("       DATASET SPLIT COMPLETED");
            System.out.println("======================================");

            System.out.println();
            System.out.println(
                    "Next step: ML model training"
            );

        }
        catch (Exception e) {

            System.out.println();
            System.out.println("======================================");
            System.out.println("             SPLITTER ERROR");
            System.out.println("======================================");

            System.out.println();
            System.out.println(
                    "Error: " + e.getMessage()
            );

            e.printStackTrace();
        }
    }

    /**
     * Write one dataset.
     */
    private static void writeDataset(
            String fileName,
            String header,
            List<String> subjects,
            Map<String, List<String>> subjectData)
            throws IOException {

        try (BufferedWriter writer =
                     Files.newBufferedWriter(
                             Paths.get(fileName))) {

            writer.write(header);
            writer.newLine();

            for (String subject : subjects) {

                List<String> rows =
                        subjectData.get(subject);

                if (rows == null) {
                    continue;
                }

                for (String row : rows) {
                    writer.write(row);
                    writer.newLine();
                }
            }
        }
    }

    /**
     * Count rows belonging to selected subjects.
     */
    private static int countRows(
            List<String> subjects,
            Map<String, List<String>> subjectData) {

        int count = 0;

        for (String subject : subjects) {

            List<String> rows =
                    subjectData.get(subject);

            if (rows != null) {
                count += rows.size();
            }
        }

        return count;
    }

    /**
     * Extract numeric part of subject ID.
     *
     * S01 -> 1
     * S02 -> 2
     * S20 -> 20
     */
    private static int subjectNumber(String subject) {

        try {
            return Integer.parseInt(
                    subject.replaceAll("[^0-9]", "")
            );
        }
        catch (Exception e) {
            return Integer.MAX_VALUE;
        }
    }
}