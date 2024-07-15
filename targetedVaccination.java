import java.util.*;
import java.lang.Math;

public class targetedVaccination {
    public static void main(String[] args) {
        int n = 10;
        double vaccinationPercentage = 0.5;
        double transmissionChance = 0.1;
        double recoveryChance = 0.1;
        double connectionFormingChance = 1;

        int previousStatus[] = new int[n];

        int intercourseChart[][] = new int[n][n];

        // Fill the intercourse chart
        fillIntercourseArray(intercourseChart, n, connectionFormingChance);
        printArray(intercourseChart, n);

        // Fill the status array
        fillStatusArray(previousStatus, n, vaccinationPercentage);
        printArray(previousStatus, n);

        // Run the simulation
        int[] infectionCounts = runSimulation(transmissionChance, recoveryChance, intercourseChart, previousStatus, n);
        System.out.println(Arrays.toString(infectionCounts));
    }

    public static int[] runSimulation(double transmissionChance, double recoveryChance, int[][] intercourseChart, int[] previousStatus, int n) {
        ArrayList<Integer> infectionCounts = new ArrayList<>();
        double r;
        int i, j;
        int maxDay = 1000;
        int day = 1;
        int[] currentStatus = new int[n];

        while (day <= maxDay) {
            boolean allRecovered = true;
            currentStatus = Arrays.copyOf(previousStatus, n);
            for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                    if ((intercourseChart[i][j] == 1) && (previousStatus[i] != previousStatus[j]) && (previousStatus[i] != -1) && (previousStatus[j] != -1)) {
                        r = Math.random();
                        if (r < transmissionChance) {
                            currentStatus[i] = 1;
                            currentStatus[j] = 1;
                        }
                    }
                }
            }
            int infectedCount = 0;
            for (i = 0; i < n; i++) {
                if (currentStatus[i] == 1) {
                    r = Math.random();
                    if (r < recoveryChance) {
                        currentStatus[i] = 0;
                    }
                }
                if (currentStatus[i] == 1) {
                    allRecovered = false;
                    infectedCount++;
                }
            }

            infectionCounts.add(infectedCount);
            previousStatus = Arrays.copyOf(currentStatus, n);
            if (allRecovered) {
                break;
            }
            day++;
        }

        // Add zeros for remaining days if allRecovered
        while (day <= maxDay) {
            infectionCounts.add(0);
            day++;
        }

        // Convert ArrayList to int[]
        int[] result = new int[infectionCounts.size()];
        for (i = 0; i < infectionCounts.size(); i++) {
            result[i] = infectionCounts.get(i);
        }

        return result;
    }

    public static void fillIntercourseArray(int[][] array, int n, double connectionFormingChance) {
        Random rand = new Random();

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (j < i) {
                    array[i][j] = rand.nextDouble() < connectionFormingChance ? 1 : 0;
                } else {
                    array[i][j] = 0;
                }
            }
        }
    }

    public static void fillStatusArray(int[] array, int n, double vaccinationPercentage) {
        Random rand = new Random();
        int vaccinationCount = (int) Math.round(vaccinationPercentage * n);
        Arrays.fill(array, 0); // Ensure array is initialized with zeros
        int count = 0;
        while (count < vaccinationCount) {
            int index = rand.nextInt(n);
            if (array[index] != -1) {
                array[index] = -1;
                count++;
            }
        }
        while (true) {
            int index = rand.nextInt(n);
            if (array[index] != -1) {
                array[index] = 1;
                break;
            }
        }
    }

    public static void printArray(int[][] array, int n) {
        System.out.println("Intercourse Chart:");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                System.out.print(array[i][j] + " ");
            }
            System.out.println();
        }
    }

    public static void printArray(int[] array, int n) {
        System.out.println("Status Array:");
        for (int i = 0; i < n; i++) {
            System.out.print(array[i] + " ");
        }
        System.out.println();
    }
}
