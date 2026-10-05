package fatigue;

import java.io.*;
import java.util.*;

public class DatasetLoader {

    public static void main(String[] args) {

        String trainFile = "data/processed/split/train.csv";
        String validationFile = "data/processed/split/validation.csv";
        String testFile = "data/processed/split/test.csv";

        System.out.println("======================================");
        System.out.println("       DATASET LOADER & CHECK");
        System.out.println("======================================");

        checkDataset(trainFile, "TRAINING");
        checkDataset(validationFile, "VALIDATION");
        checkDataset(testFile, "TEST");

        System.out.println();
        System.out.println("======================================");
        System.out.println("       DATASET CHECK COMPLETED");
        System.out.println("======================================");
    }

    private static void checkDataset(String filePath, String datasetName) {

        System.out.println();
        System.out.println("--------------------------------------");
        System.out.println(datasetName + " DATASET");
        System.out.println("--------------------------------------");

        File file = new File(filePath);

        if (!file.exists()) {
            System.out.println("ERROR: File not found:");
            System.out.println(filePath);
            return;
        }

        int rows = 0;
        int columns = 0;
        int fatigue = 0;
        int nonFatigue = 0;

        Set<String> subjects = new LinkedHashSet<>();

        try (BufferedReader br = new BufferedReader(new FileReader(file))) {

            String header = br.readLine();

            if (header == null) {
                System.out.println("ERROR: Empty file.");
                return;
            }

            String[] headerColumns = header.split(",", -1);
            columns = headerColumns.length;

            int subjectIndex = -1;
            int labelIndex = -1;

            for (int i = 0; i < headerColumns.length; i++) {

                if (headerColumns[i].equalsIgnoreCase("subject")) {
                    subjectIndex = i;
                }

                if (headerColumns[i].equalsIgnoreCase("fatigue_label")) {
                    labelIndex = i;
                }
            }

            String line;

            while ((line = br.readLine()) != null) {

                if (line.trim().isEmpty()) {
                    continue;
                }

                String[] values = line.split(",", -1);

                if (values.length != columns) {
                    System.out.println(
                            "WARNING: Invalid column count at row "
                                    + (rows + 2)
                    );
                    continue;
                }

                rows++;

                if (subjectIndex >= 0) {
                    subjects.add(values[subjectIndex]);
                }

                if (labelIndex >= 0) {

                    String label = values[labelIndex].trim();

                    if (label.equals("1")) {
                        fatigue++;
                    }
                    else if (label.equals("0")) {
                        nonFatigue++;
                    }
                }
            }

            System.out.println("File              : " + filePath);
            System.out.println("Rows              : " + rows);
            System.out.println("Columns           : " + columns);
            System.out.println("Subjects          : " + subjects);
            System.out.println("Number of subjects: " + subjects.size());
            System.out.println("Non-Fatigue       : " + nonFatigue);
            System.out.println("Fatigue           : " + fatigue);

        }
        catch (IOException e) {

            System.out.println("ERROR reading dataset:");
            e.printStackTrace();
        }
    }
}