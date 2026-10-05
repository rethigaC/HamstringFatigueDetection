package fatigue;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class DatasetSplit {

    public static void main(String[] args) throws Exception {

        String input =
                "data/processed/labeled_temporal_sequences.csv";

        String outputDir =
                "data/processed/split/";

        Files.createDirectories(Paths.get(outputDir));

        System.out.println("======================================");
        System.out.println("       SUBJECT-WISE DATASET SPLIT");
        System.out.println("======================================");

        List<String> lines = Files.readAllLines(Paths.get(input));

        if (lines.size() <= 1) {
            throw new RuntimeException("Dataset is empty.");
        }

        String header = lines.get(0);

        // Group rows by subject
        Map<String, List<String>> subjectData =
                new LinkedHashMap<>();

        for (int i = 1; i < lines.size(); i++) {

            String line = lines.get(i);

            if (line.trim().isEmpty()) {
                continue;
            }

            String[] parts = line.split(",", 3);

            if (parts.length < 3) {
                continue;
            }

            String subject = parts[0];

            subjectData
                    .computeIfAbsent(subject,
                            k -> new ArrayList<>())
                    .add(line);
        }

        List<String> subjects =
                new ArrayList<>(subjectData.keySet());

        Collections.sort(subjects);

        System.out.println("Total subjects : " + subjects.size());
        System.out.println("Subjects       : " + subjects);

        // Shuffle subjects reproducibly
        Collections.shuffle(subjects, new Random(42));

        int totalSubjects = subjects.size();

        int trainCount =
                (int) Math.round(totalSubjects * 0.70);

        int validationCount =
                (int) Math.round(totalSubjects * 0.15);

        int testCount =
                totalSubjects - trainCount - validationCount;

        List<String> trainSubjects =
                subjects.subList(0, trainCount);

        List<String> validationSubjects =
                subjects.subList(
                        trainCount,
                        trainCount + validationCount);

        List<String> testSubjects =
                subjects.subList(
                        trainCount + validationCount,
                        totalSubjects);

        System.out.println();
        System.out.println("----------- SUBJECT SPLIT -----------");

        System.out.println("Training subjects   : "
                + trainSubjects);

        System.out.println("Validation subjects : "
                + validationSubjects);

        System.out.println("Testing subjects    : "
                + testSubjects);

        System.out.println("-------------------------------------");

        // Write datasets
        writeDataset(
                outputDir + "train.csv",
                header,
                trainSubjects,
                subjectData
        );

        writeDataset(
                outputDir + "validation.csv",
                header,
                validationSubjects,
                subjectData
        );

        writeDataset(
                outputDir + "test.csv",
                header,
                testSubjects,
                subjectData
        );

        System.out.println();
        System.out.println("======================================");
        System.out.println("          SPLIT COMPLETED");
        System.out.println("======================================");

        System.out.println("Training rows   : "
                + countRows(trainSubjects, subjectData));

        System.out.println("Validation rows : "
                + countRows(validationSubjects, subjectData));

        System.out.println("Testing rows    : "
                + countRows(testSubjects, subjectData));

        System.out.println();
        System.out.println("Generated files:");

        System.out.println(
                outputDir + "train.csv");

        System.out.println(
                outputDir + "validation.csv");

        System.out.println(
                outputDir + "test.csv");
    }

    private static void writeDataset(
            String file,
            String header,
            List<String> subjects,
            Map<String, List<String>> subjectData)
            throws IOException {

        try (BufferedWriter writer =
                     Files.newBufferedWriter(
                             Paths.get(file))) {

            writer.write(header);
            writer.newLine();

            for (String subject : subjects) {

                for (String row :
                        subjectData.get(subject)) {

                    writer.write(row);
                    writer.newLine();
                }
            }
        }
    }

    private static int countRows(
            List<String> subjects,
            Map<String, List<String>> subjectData) {

        int count = 0;

        for (String subject : subjects) {
            count += subjectData.get(subject).size();
        }

        return count;
    }
}