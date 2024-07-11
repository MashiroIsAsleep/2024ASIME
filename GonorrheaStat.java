import java.util.*;
import java.lang.Math;

public class GonorrheaStat {

    public static int runSimulation() {
        double r;
        int day = 1;
        double infectionPercent = 0.3;
        double recoverPercent = 0.5;
        int[] infectedState = new int[]{0, 0, 0, 0, 1, 0};
        int[] futureState = new int[infectedState.length];
        int[][] interaction = new int[][]{
            {0, 0, 0, 0, 0, 0},
            {1, 0, 0, 0, 0, 0},
            {1, 0, 0, 0, 0, 0},
            {0, 1, 1, 0, 0, 0},
            {1, 1, 0, 1, 0, 0},
            {0, 1, 1, 0, 0, 0},
        };

        int maxDay = 25;
        int i, j;

        while (day < maxDay) {
            boolean allRecovered = true;
            // Copy current state to future state
            futureState = Arrays.copyOf(infectedState, infectedState.length);

            // Check interactions and update future state for new infections
            for (i = 0; i < infectedState.length; i++) {
                for (j = 0; j < infectedState.length; j++) {
                    if ((interaction[i][j] == 1) && (infectedState[i] != infectedState[j])) {
                        r = Math.random();
                        if (r < infectionPercent) {
                            futureState[i] = 1;
                            futureState[j] = 1;
                        }
                    }
                }
            }

            // Apply recoveries to the future state
            for (i = 0; i < futureState.length; i++) {
                if (futureState[i] == 1) {
                    r = Math.random();
                    if (r < recoverPercent) {
                        futureState[i] = 0;
                    }
                }
                if (futureState[i] == 1) {
                    allRecovered = false;
                }
            }

            // Update infectedState to the new future state
            infectedState = Arrays.copyOf(futureState, futureState.length);

            // Check if all are recovered
            if (allRecovered) {
                return day;
            }

            day++;
        }
        return maxDay;
    }

    public static void main(String[] args) {
        int totalDays = 0;
        int runs = 10000;

        for (int i = 0; i < runs; i++) {
            totalDays += runSimulation();
        }

        double averageDays = (double) totalDays / runs;
        System.out.println("Average days for all to recover: " + averageDays);
    }
}
