package fatigue;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class SequenceGenerator {

    public static void generate(
            String inputFile,
            String outputFile,
            int sequenceLength) throws IOException {

        List<String[]> rows = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(
                new FileReader(inputFile))) {

            String line;

            // Read header
            String header = br.readLine();

            if (header == null) {
                throw new IOException("Input CSV is empty.");
            }

            // Read data
            while ((line = br.readLine()) != null) {

                if (line.trim().isEmpty()) {
                    continue;
                }

                rows.add(line.split(",", -1));
            }
        }

        if (rows.isEmpty()) {
            throw new IOException("No data rows found.");
        }

        /*
         * Combined dataset format:
         *
         * 0  subject
         * 1  window_id
         * 2  start_time_s
         * 3  end_time_s
         *
         * 4-13 = 10 features
         */

        int featuresPerStep = 10;

        System.out.println();
        System.out.println("======================================");
        System.out.println("     TEMPORAL SEQUENCE GENERATION");
        System.out.println("======================================");

        System.out.println(
                "Total input rows : " + rows.size()
        );

        System.out.println(
                "Sequence length  : " + sequenceLength
        );

        System.out.println(
                "Features/step    : " + featuresPerStep
        );


        // =====================================================
        // GROUP DATA BY SUBJECT
        // =====================================================

        Map<String, List<String[]>> subjectData =
                new LinkedHashMap<>();

        for (String[] row : rows) {

            if (row.length < 14) {
                continue;
            }

            String subject = row[0];

            subjectData
                    .computeIfAbsent(
                            subject,
                            k -> new ArrayList<>()
                    )
                    .add(row);
        }


        System.out.println(
                "Subjects         : " + subjectData.size()
        );


        // =====================================================
        // CREATE OUTPUT
        // =====================================================

        File output = new File(outputFile);

        File parent = output.getParentFile();

        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }


        try (BufferedWriter writer =
                     new BufferedWriter(
                             new FileWriter(outputFile))) {


            // =================================================
            // HEADER
            // =================================================

            writer.write("subject,sequence_id");

            for (int t = 1; t <= sequenceLength; t++) {

                for (int f = 1; f <= featuresPerStep; f++) {

                    writer.write(
                            ",t" + t + "_f" + f
                    );
                }
            }

            writer.newLine();


            // =================================================
            // SEQUENCES
            // =================================================

            int totalSequences = 0;

            for (Map.Entry<String, List<String[]>> entry :
                    subjectData.entrySet()) {

                String subject = entry.getKey();

                List<String[]> subjectRows =
                        entry.getValue();


                if (subjectRows.size() < sequenceLength) {
                    continue;
                }


                for (
                        int start = 0;
                        start <= subjectRows.size()
                                - sequenceLength;
                        start++
                ) {

                    writer.write(subject);
                    writer.write(",");
                    writer.write(
                            String.valueOf(totalSequences + 1)
                    );


                    // -----------------------------------------
                    // 20 TIME STEPS
                    // -----------------------------------------

                    for (
                            int t = 0;
                            t < sequenceLength;
                            t++
                    ) {

                        String[] row =
                                subjectRows.get(start + t);


                        // Features are columns 4-13
                        for (
                                int f = 0;
                                f < featuresPerStep;
                                f++
                        ) {

                            writer.write(",");

                            writer.write(
                                    row[4 + f]
                            );
                        }
                    }

                    writer.newLine();

                    totalSequences++;
                }
            }


            System.out.println();
            System.out.println(
                    "Total sequences  : "
                            + totalSequences
            );


            // =================================================
            // INFORMATION
            // =================================================

            System.out.println();
            System.out.println(
                    "Output file:"
            );

            System.out.println(outputFile);

            System.out.println();
            System.out.println(
                    "Expected ML input shape:"
            );

            System.out.println(
                    "("
                    + totalSequences
                    + ", "
                    + sequenceLength
                    + ", "
                    + featuresPerStep
                    + ")"
            );
        }
    }
}