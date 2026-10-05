package fatigue;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class MLDatasetPreparer {

    private static final int TRAIN_PERCENT = 70;
    private static final int VALIDATION_PERCENT = 15;

    public static void prepareDataset(
            String inputFile,
            String trainFile,
            String validationFile,
            String testFile) throws IOException {

        List<String> lines = Files.readAllLines(Paths.get(inputFile));

        if (lines.size() <= 1) {
            throw new IOException("Labeled dataset is empty.");
        }

        String header = lines.get(0);

        List<String> data = new ArrayList<>(lines.subList(1, lines.size()));

        // -----------------------------------------------------
        // IMPORTANT:
        // Do NOT randomly shuffle because temporal sequences
        // overlap heavily.
        //
        // We use chronological splitting:
        // 70% Train
        // 15% Validation
        // 15% Test
        // -----------------------------------------------------

        int total = data.size();

        int trainSize =
                (int) Math.round(total * TRAIN_PERCENT / 100.0);

        int validationSize =
                (int) Math.round(total * VALIDATION_PERCENT / 100.0);

        int testStart = trainSize + validationSize;

        List<String> train =
                data.subList(0, trainSize);

        List<String> validation =
                data.subList(trainSize, testStart);

        List<String> test =
                data.subList(testStart, total);

        writeFile(trainFile, header, train);
        writeFile(validationFile, header, validation);
        writeFile(testFile, header, test);

        System.out.println();
        System.out.println("======================================");
        System.out.println("       ML DATASET PREPARATION");
        System.out.println("======================================");

        System.out.println();
        System.out.println("Total samples      : " + total);
        System.out.println("Training samples   : " + train.size());
        System.out.println("Validation samples : " + validation.size());
        System.out.println("Testing samples    : " + test.size());

        System.out.println();
        System.out.println("Train file:");
        System.out.println(trainFile);

        System.out.println();
        System.out.println("Validation file:");
        System.out.println(validationFile);

        System.out.println();
        System.out.println("Test file:");
        System.out.println(testFile);

        System.out.println();
        System.out.println("Expected input shape:");
        System.out.println("(samples, 20, 10)");
    }

    private static void writeFile(
            String filename,
            String header,
            List<String> rows) throws IOException {

        Path path = Paths.get(filename);

        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }

        try (BufferedWriter writer =
                     Files.newBufferedWriter(path)) {

            writer.write(header);
            writer.newLine();

            for (String row : rows) {
                writer.write(row);
                writer.newLine();
            }
        }
    }
}