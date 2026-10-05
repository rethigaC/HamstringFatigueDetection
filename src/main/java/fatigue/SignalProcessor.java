package fatigue;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class SignalProcessor {

    // Features used in our project
    private static final String[] FEATURE_NAMES = {
        "RMS",
        "MAV",
        "VAR",
        "MNF_Hz",
        "MDF_Hz"
    };

    // ============================================================
    // 1. READ CSV
    // ============================================================

    public List<String[]> readCSV(String filePath) throws IOException {

        List<String[]> records = new ArrayList<>();

        try (BufferedReader br =
                     Files.newBufferedReader(Paths.get(filePath))) {

            String line;

            while ((line = br.readLine()) != null) {

                if (!line.trim().isEmpty()) {
                    records.add(line.split(",", -1));
                }
            }
        }

        return records;
    }

    // ============================================================
    // 2. FIND COLUMN INDEX
    // ============================================================

    private int getColumnIndex(
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
    // 3. DATA VALIDATION
    // ============================================================

    public void validateDataset(
            List<String[]> data) {

        if (data == null || data.isEmpty()) {

            System.out.println("Dataset is empty.");
            return;
        }

        String[] header = data.get(0);

        System.out.println();
        System.out.println("======================================");
        System.out.println("          DATA VALIDATION");
        System.out.println("======================================");

        System.out.println(
                "Total rows    : " + (data.size() - 1));

        System.out.println(
                "Total columns : " + header.length);

        // Find feature columns
        int[] featureIndexes =
                new int[FEATURE_NAMES.length];

        for (int i = 0;
             i < FEATURE_NAMES.length;
             i++) {

            featureIndexes[i] =
                    getColumnIndex(
                            header,
                            FEATURE_NAMES[i]);

            if (featureIndexes[i] == -1) {

                System.out.println(
                        "WARNING: Column not found -> "
                                + FEATURE_NAMES[i]);
            }
        }

        int missingValues = 0;
        int invalidValues = 0;
        int validRows = 0;

        double[] min =
                new double[FEATURE_NAMES.length];

        double[] max =
                new double[FEATURE_NAMES.length];

        for (int i = 0;
             i < FEATURE_NAMES.length;
             i++) {

            min[i] = Double.POSITIVE_INFINITY;
            max[i] = Double.NEGATIVE_INFINITY;
        }

        // Check every row
        for (int row = 1;
             row < data.size();
             row++) {

            String[] values = data.get(row);

            boolean rowValid = true;

            for (int i = 0;
                 i < FEATURE_NAMES.length;
                 i++) {

                int index = featureIndexes[i];

                if (index == -1 ||
                        index >= values.length) {

                    rowValid = false;
                    continue;
                }

                String value =
                        values[index].trim();

                // Missing value
                if (value.isEmpty()) {

                    missingValues++;
                    rowValid = false;
                    continue;
                }

                try {

                    double number =
                            Double.parseDouble(value);

                    // NaN / Infinity
                    if (Double.isNaN(number) ||
                            Double.isInfinite(number)) {

                        invalidValues++;
                        rowValid = false;
                        continue;
                    }

                    if (number < min[i]) {
                        min[i] = number;
                    }

                    if (number > max[i]) {
                        max[i] = number;
                    }

                } catch (NumberFormatException e) {

                    invalidValues++;
                    rowValid = false;
                }
            }

            if (rowValid) {
                validRows++;
            }
        }

        System.out.println();
        System.out.println(
                "Valid rows     : " + validRows);

        System.out.println(
                "Missing values : " + missingValues);

        System.out.println(
                "Invalid values : " + invalidValues);

        System.out.println();
        System.out.println(
                "----------- FEATURE RANGES -----------");

        for (int i = 0;
             i < FEATURE_NAMES.length;
             i++) {

            System.out.println(
                    FEATURE_NAMES[i]
                            + " : Min = "
                            + min[i]
                            + " | Max = "
                            + max[i]);
        }

        System.out.println(
                "--------------------------------------");
    }

    // ============================================================
    // 4. CLEAN DATASET
    // ============================================================

    public List<String[]> cleanDataset(
            List<String[]> data) {

        List<String[]> cleaned =
                new ArrayList<>();

        if (data == null || data.isEmpty()) {
            return cleaned;
        }

        String[] header = data.get(0);

        cleaned.add(header);

        int[] featureIndexes =
                new int[FEATURE_NAMES.length];

        for (int i = 0;
             i < FEATURE_NAMES.length;
             i++) {

            featureIndexes[i] =
                    getColumnIndex(
                            header,
                            FEATURE_NAMES[i]);
        }

        int removedRows = 0;

        for (int row = 1;
             row < data.size();
             row++) {

            String[] values = data.get(row);

            boolean valid = true;

            for (int index : featureIndexes) {

                if (index == -1 ||
                        index >= values.length) {

                    valid = false;
                    break;
                }

                String value =
                        values[index].trim();

                if (value.isEmpty()) {

                    valid = false;
                    break;
                }

                try {

                    double number =
                            Double.parseDouble(value);

                    if (Double.isNaN(number) ||
                            Double.isInfinite(number)) {

                        valid = false;
                        break;
                    }

                } catch (NumberFormatException e) {

                    valid = false;
                    break;
                }
            }

            if (valid) {

                cleaned.add(values);

            } else {

                removedRows++;
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
                        + removedRows);

        System.out.println(
                "Cleaned rows  : "
                        + (cleaned.size() - 1));

        return cleaned;
    }

    // ============================================================
    // 5. CALCULATE MEAN
    // ============================================================

    private double calculateMean(
            List<String[]> data,
            int columnIndex) {

        double sum = 0.0;
        int count = 0;

        for (int i = 1;
             i < data.size();
             i++) {

            double value =
                    Double.parseDouble(
                            data.get(i)[columnIndex]);

            sum += value;
            count++;
        }

        return sum / count;
    }

    // ============================================================
    // 6. CALCULATE STANDARD DEVIATION
    // ============================================================

    private double calculateStandardDeviation(
            List<String[]> data,
            int columnIndex,
            double mean) {

        double sum = 0.0;
        int count = 0;

        for (int i = 1;
             i < data.size();
             i++) {

            double value =
                    Double.parseDouble(
                            data.get(i)[columnIndex]);

            double difference =
                    value - mean;

            sum += difference * difference;
            count++;
        }

        return Math.sqrt(sum / count);
    }

    // ============================================================
    // 7. Z-SCORE NORMALIZATION
    // ============================================================

    public List<String[]> normalizeFeatures(
            List<String[]> data) {

        List<String[]> normalized =
                new ArrayList<>();

        if (data == null ||
                data.size() <= 1) {

            return normalized;
        }

        String[] header = data.get(0);

        // Find feature columns
        int rmsIndex =
                getColumnIndex(header, "RMS");

        int mavIndex =
                getColumnIndex(header, "MAV");

        int varIndex =
                getColumnIndex(header, "VAR");

        int mnfIndex =
                getColumnIndex(header, "MNF_Hz");

        int mdfIndex =
                getColumnIndex(header, "MDF_Hz");

        // Verify columns
        if (rmsIndex == -1 ||
                mavIndex == -1 ||
                varIndex == -1 ||
                mnfIndex == -1 ||
                mdfIndex == -1) {

            throw new IllegalArgumentException(
                    "Required feature column is missing.");
        }

        // --------------------------------------------------------
        // Calculate means
        // --------------------------------------------------------

        double rmsMean =
                calculateMean(data, rmsIndex);

        double mavMean =
                calculateMean(data, mavIndex);

        double varMean =
                calculateMean(data, varIndex);

        double mnfMean =
                calculateMean(data, mnfIndex);

        double mdfMean =
                calculateMean(data, mdfIndex);

        // --------------------------------------------------------
        // Calculate standard deviations
        // --------------------------------------------------------

        double rmsStd =
                calculateStandardDeviation(
                        data,
                        rmsIndex,
                        rmsMean);

        double mavStd =
                calculateStandardDeviation(
                        data,
                        mavIndex,
                        mavMean);

        double varStd =
                calculateStandardDeviation(
                        data,
                        varIndex,
                        varMean);

        double mnfStd =
                calculateStandardDeviation(
                        data,
                        mnfIndex,
                        mnfMean);

        double mdfStd =
                calculateStandardDeviation(
                        data,
                        mdfIndex,
                        mdfMean);

        // Prevent division by zero
        if (rmsStd == 0) rmsStd = 1;
        if (mavStd == 0) mavStd = 1;
        if (varStd == 0) varStd = 1;
        if (mnfStd == 0) mnfStd = 1;
        if (mdfStd == 0) mdfStd = 1;

        // --------------------------------------------------------
        // Display statistics
        // --------------------------------------------------------

        System.out.println();
        System.out.println(
                "======================================");

        System.out.println(
                "       NORMALIZATION STATISTICS");

        System.out.println(
                "======================================");

        System.out.printf(
                "RMS  -> Mean: %.10e | Std: %.10e%n",
                rmsMean,
                rmsStd);

        System.out.printf(
                "MAV  -> Mean: %.10e | Std: %.10e%n",
                mavMean,
                mavStd);

        System.out.printf(
                "VAR  -> Mean: %.10e | Std: %.10e%n",
                varMean,
                varStd);

        System.out.printf(
                "MNF  -> Mean: %.6f | Std: %.6f%n",
                mnfMean,
                mnfStd);

        System.out.printf(
                "MDF  -> Mean: %.6f | Std: %.6f%n",
                mdfMean,
                mdfStd);

        // --------------------------------------------------------
        // Create new header
        // --------------------------------------------------------

        String[] newHeader =
                new String[header.length + 5];

        System.arraycopy(
                header,
                0,
                newHeader,
                0,
                header.length);

        newHeader[header.length] = "RMS_Z";
        newHeader[header.length + 1] = "MAV_Z";
        newHeader[header.length + 2] = "VAR_Z";
        newHeader[header.length + 3] = "MNF_Z";
        newHeader[header.length + 4] = "MDF_Z";

        normalized.add(newHeader);

        // --------------------------------------------------------
        // Normalize every row
        // --------------------------------------------------------

        for (int i = 1;
             i < data.size();
             i++) {

            String[] originalRow =
                    data.get(i);

            String[] newRow =
                    new String[originalRow.length + 5];

            System.arraycopy(
                    originalRow,
                    0,
                    newRow,
                    0,
                    originalRow.length);

            double rms =
                    Double.parseDouble(
                            originalRow[rmsIndex]);

            double mav =
                    Double.parseDouble(
                            originalRow[mavIndex]);

            double variance =
                    Double.parseDouble(
                            originalRow[varIndex]);

            double mnf =
                    Double.parseDouble(
                            originalRow[mnfIndex]);

            double mdf =
                    Double.parseDouble(
                            originalRow[mdfIndex]);

            double rmsZ =
                    (rms - rmsMean) / rmsStd;

            double mavZ =
                    (mav - mavMean) / mavStd;

            double varZ =
                    (variance - varMean) / varStd;

            double mnfZ =
                    (mnf - mnfMean) / mnfStd;

            double mdfZ =
                    (mdf - mdfMean) / mdfStd;

            newRow[originalRow.length] =
                    String.valueOf(rmsZ);

            newRow[originalRow.length + 1] =
                    String.valueOf(mavZ);

            newRow[originalRow.length + 2] =
                    String.valueOf(varZ);

            newRow[originalRow.length + 3] =
                    String.valueOf(mnfZ);

            newRow[originalRow.length + 4] =
                    String.valueOf(mdfZ);

            normalized.add(newRow);
        }

        return normalized;
    }

    // ============================================================
    // 8. SAVE CSV
    // ============================================================

    public void saveCSV(
            List<String[]> data,
            String outputPath) throws IOException {

        Path path =
                Paths.get(outputPath);

        // Create directory if required
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

        System.out.println(outputPath);
    }
}