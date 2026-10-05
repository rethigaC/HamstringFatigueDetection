package fatigue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class FatigueLabeler {

    // Fatigue threshold based on exercise/window progression
    private static final double FATIGUE_THRESHOLD = 0.70;

    public static void generateLabels(
            String inputFile,
            String outputFile) throws IOException {

        System.out.println();
        System.out.println("======================================");
        System.out.println("       FATIGUE LABEL GENERATION");
        System.out.println("======================================");

        List<String> lines = Files.readAllLines(
                Paths.get(inputFile)
        );

        if (lines.isEmpty()) {
            throw new IOException("Input temporal sequence file is empty.");
        }

        String header = lines.get(0);

        // Add fatigue label to the output
        String newHeader = header + ",fatigue_label";

        List<String> output = new ArrayList<>();
        output.add(newHeader);

        int nonFatigue = 0;
        int fatigue = 0;

        /*
         * Each temporal sequence contains 20 time steps.
         * We use the sequence position/window progression
         * as a provisional fatigue indicator.
         *
         * sequence_id:
         * 1 ... early stage
         * later sequences ... increasing fatigue exposure
         */

        for (int i = 1; i < lines.size(); i++) {

            String line = lines.get(i);

            if (line.trim().isEmpty()) {
                continue;
            }

            String[] parts = line.split(",", -1);

            /*
             * Column 1 = subject
             * Column 2 = sequence_id
             */
            if (parts.length < 2) {
                continue;
            }

            int sequenceId;

            try {
                sequenceId = Integer.parseInt(
                        parts[1].trim()
                );
            } catch (NumberFormatException e) {
                continue;
            }

            /*
             * There are 241 sequences.
             *
             * Calculate relative progress:
             *
             * sequence 1 -> near 0
             * sequence 241 -> near 1
             */
            double progress;

            if (lines.size() <= 2) {
                progress = 0.0;
            } else {
                progress =
                        (double) (sequenceId - 1)
                        / (double) (lines.size() - 2);
            }

            int label;

            if (progress >= FATIGUE_THRESHOLD) {
                label = 1;
                fatigue++;
            } else {
                label = 0;
                nonFatigue++;
            }

            output.add(line + "," + label);
        }

        Files.createDirectories(
                Paths.get(outputFile).getParent()
        );

        Files.write(
                Paths.get(outputFile),
                output
        );

        System.out.println();
        System.out.println("Input file:");
        System.out.println(inputFile);

        System.out.println();
        System.out.println("Output file:");
        System.out.println(outputFile);

        System.out.println();
        System.out.println("Fatigue threshold : "
                + FATIGUE_THRESHOLD);

        System.out.println();
        System.out.println("Non-Fatigue samples : "
                + nonFatigue);

        System.out.println("Fatigue samples     : "
                + fatigue);

        System.out.println("Total labeled       : "
                + (nonFatigue + fatigue));

        System.out.println();
        System.out.println("Fatigue labeling completed.");
    }
}