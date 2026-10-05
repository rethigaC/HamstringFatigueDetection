package fatigue;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FeatureFusion {

    // ============================================================
    // Find column index
    // ============================================================

    private static int getColumnIndex(
            String[] header,
            String columnName) {

        for (int i = 0; i < header.length; i++) {

            if (header[i].trim()
                    .equalsIgnoreCase(columnName)) {

                return i;
            }
        }

        return -1;
    }

    // ============================================================
    // Read CSV
    // ============================================================

    private static List<String[]> readCSV(
            String filePath) throws IOException {

        List<String[]> data =
                new ArrayList<>();

        try (BufferedReader reader =
                     Files.newBufferedReader(
                             Paths.get(filePath))) {

            String line;

            while ((line = reader.readLine()) != null) {

                if (!line.trim().isEmpty()) {

                    data.add(line.split(",", -1));
                }
            }
        }

        return data;
    }

    // ============================================================
    // Save CSV
    // ============================================================

    private static void saveCSV(
            List<String[]> data,
            String filePath) throws IOException {

        Path path =
                Paths.get(filePath);

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
    }

    // ============================================================
    // Create unique key
    // ============================================================

    private static String createKey(
            String subject,
            String windowId) {

        return subject.trim()
                + "_"
                + windowId.trim();
    }

    // ============================================================
    // MAIN FUSION FUNCTION
    // ============================================================

    public static void combineFeatures(
            String bfFile,
            String hamstringFile,
            String outputFile) throws IOException {

        System.out.println();
        System.out.println(
                "======================================");

        System.out.println(
                "       BF + HAMSTRING FUSION");

        System.out.println(
                "======================================");

        // --------------------------------------------------------
        // Read datasets
        // --------------------------------------------------------

        List<String[]> bfData =
                readCSV(bfFile);

        List<String[]> hamData =
                readCSV(hamstringFile);

        if (bfData.size() <= 1 ||
                hamData.size() <= 1) {

            throw new IllegalArgumentException(
                    "One of the datasets is empty.");
        }

        String[] bfHeader =
                bfData.get(0);

        String[] hamHeader =
                hamData.get(0);

        // --------------------------------------------------------
        // Find required columns
        // --------------------------------------------------------

        int bfSubject =
                getColumnIndex(
                        bfHeader,
                        "subject");

        int bfWindow =
                getColumnIndex(
                        bfHeader,
                        "window_id");

        int bfStart =
                getColumnIndex(
                        bfHeader,
                        "start_time_s");

        int bfEnd =
                getColumnIndex(
                        bfHeader,
                        "end_time_s");

        int[] bfFeatures = {

            getColumnIndex(bfHeader, "RMS_Z"),
            getColumnIndex(bfHeader, "MAV_Z"),
            getColumnIndex(bfHeader, "VAR_Z"),
            getColumnIndex(bfHeader, "MNF_Z"),
            getColumnIndex(bfHeader, "MDF_Z")
        };

        int hamSubject =
                getColumnIndex(
                        hamHeader,
                        "subject");

        int hamWindow =
                getColumnIndex(
                        hamHeader,
                        "window_id");

        int[] hamFeatures = {

            getColumnIndex(hamHeader, "RMS_Z"),
            getColumnIndex(hamHeader, "MAV_Z"),
            getColumnIndex(hamHeader, "VAR_Z"),
            getColumnIndex(hamHeader, "MNF_Z"),
            getColumnIndex(hamHeader, "MDF_Z")
        };

        // --------------------------------------------------------
        // Validate columns
        // --------------------------------------------------------

        if (bfSubject == -1 ||
                bfWindow == -1 ||
                hamSubject == -1 ||
                hamWindow == -1) {

            throw new IllegalArgumentException(
                    "subject or window_id column is missing.");
        }

        for (int index : bfFeatures) {

            if (index == -1) {

                throw new IllegalArgumentException(
                        "Missing BF normalized feature.");
            }
        }

        for (int index : hamFeatures) {

            if (index == -1) {

                throw new IllegalArgumentException(
                        "Missing hamstring normalized feature.");
            }
        }

        // --------------------------------------------------------
        // Store hamstring rows using subject + window_id
        // --------------------------------------------------------

        Map<String, String[]> hamstringMap =
                new HashMap<>();

        for (int i = 1;
             i < hamData.size();
             i++) {

            String[] row =
                    hamData.get(i);

            String key =
                    createKey(
                            row[hamSubject],
                            row[hamWindow]);

            hamstringMap.put(key, row);
        }

        // --------------------------------------------------------
        // Output header
        // --------------------------------------------------------

        List<String[]> combined =
                new ArrayList<>();

        String[] outputHeader = {

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
        };

        combined.add(outputHeader);

        // --------------------------------------------------------
        // Combine matching windows
        // --------------------------------------------------------

        int matched = 0;
        int unmatched = 0;

        for (int i = 1;
             i < bfData.size();
             i++) {

            String[] bfRow =
                    bfData.get(i);

            String key =
                    createKey(
                            bfRow[bfSubject],
                            bfRow[bfWindow]);

            String[] hamRow =
                    hamstringMap.get(key);

            // No matching hamstring window
            if (hamRow == null) {

                unmatched++;
                continue;
            }

            String[] combinedRow =
                    new String[14];

            combinedRow[0] =
                    bfRow[bfSubject];

            combinedRow[1] =
                    bfRow[bfWindow];

            combinedRow[2] =
                    bfStart != -1
                            ? bfRow[bfStart]
                            : "";

            combinedRow[3] =
                    bfEnd != -1
                            ? bfRow[bfEnd]
                            : "";

            // BF features
            combinedRow[4] =
                    bfRow[bfFeatures[0]];

            combinedRow[5] =
                    bfRow[bfFeatures[1]];

            combinedRow[6] =
                    bfRow[bfFeatures[2]];

            combinedRow[7] =
                    bfRow[bfFeatures[3]];

            combinedRow[8] =
                    bfRow[bfFeatures[4]];

            // Hamstring features
            combinedRow[9] =
                    hamRow[hamFeatures[0]];

            combinedRow[10] =
                    hamRow[hamFeatures[1]];

            combinedRow[11] =
                    hamRow[hamFeatures[2]];

            combinedRow[12] =
                    hamRow[hamFeatures[3]];

            combinedRow[13] =
                    hamRow[hamFeatures[4]];

            combined.add(combinedRow);

            matched++;
        }

        // --------------------------------------------------------
        // Save
        // --------------------------------------------------------

        saveCSV(
                combined,
                outputFile);

        // --------------------------------------------------------
        // Results
        // --------------------------------------------------------

        System.out.println();

        System.out.println(
                "BF rows        : "
                        + (bfData.size() - 1));

        System.out.println(
                "Hamstring rows : "
                        + (hamData.size() - 1));

        System.out.println(
                "Matched windows: "
                        + matched);

        System.out.println(
                "Unmatched BF   : "
                        + unmatched);

        System.out.println(
                "Combined rows  : "
                        + (combined.size() - 1));

        System.out.println();

        System.out.println(
                "Combined dataset saved to:");

        System.out.println(outputFile);

        System.out.println();

        System.out.println(
                "Features per time step: 10");
    }
}