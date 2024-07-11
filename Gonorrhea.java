import java.util.*;
import java.lang.Math;

public class Gonorrhea {
    public static void main (String[] args){
        double r;
        int infectedCount;
        int day = 1;
        double infectionPercent = 0.3;
        double recoverPercent = 0.5;
        int[] infectedState = new int[]{0,0,0,0,1,0};
        int[] futureState = new int[infectedState.length];
        int[][] interaction = new int[][]{{0,0,0,0,0,0},
                                          {1,0,0,0,0,0},
                                          {1,0,0,0,0,0},
                                          {0,1,1,0,0,0},
                                          {1,1,0,1,0,0},
                                          {0,1,1,0,0,0},
                                        };

        int maxDay = 25;
        int i, j;

        while (day < maxDay){
            System.out.print("Day " + day + ": ");
            // Copy current state to future state
            futureState = Arrays.copyOf(infectedState, infectedState.length);

            // Check interactions and update future state for new infections
            for (i = 0; i < infectedState.length; i++){
                for (j = 0; j < infectedState.length; j++){
                    if ((interaction[i][j] == 1) && (infectedState[i] != infectedState[j])){
                        r = Math.random();
                        if (r < infectionPercent){
                            futureState[i] = 1;
                            futureState[j] = 1;
                        }
                    }
                }
            }

            // Apply recoveries to the future state
            for (i = 0; i < futureState.length; i++){
                if (futureState[i] == 1){
                    r = Math.random();
                    if (r < recoverPercent){
                        futureState[i] = 0;
                    }
                }
                System.out.print(futureState[i] + ", ");
            }

            // Update infectedState to the new future state
            infectedState = Arrays.copyOf(futureState, futureState.length);
            infectedCount = 0;
            for (i = 0; i < futureState.length; i++){
                infectedCount += futureState[i];
            }

            System.out.println("Infected Count: " + infectedCount);
            day++;
        }
    }
}
