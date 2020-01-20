#include <stdio.h>
#include <stdlib.h>

int main(){

    FILE *myFile;
    char buffer[100];

    // Count length of file
    myFile = fopen("test.txt", "r");
    fgets(buffer, 100, myFile);
    int counter = 0;
    double dummy;
    while (fscanf(myFile, "%lf", &dummy) != EOF){
        counter++;
    }

    int numPts = counter;
    fclose(myFile);
    myFile = fopen("test.txt", "r");
    fgets(buffer, 100, myFile);


    //read file into array
    double numberArray1[numPts];
    double numberArray2[numPts];
    int numberArray3[numPts];
    int i;

    if (myFile == NULL){
        printf("Error Reading File\n");
        exit (0);
    }

    double dum1, dum2, dum3;

    for (i = 0; i < (int)numPts/3.; i++){
        fscanf(myFile, "%lf", &dum1);
        fscanf(myFile, "%lf", &dum2);
        fscanf(myFile, "%lf", &dum3);

        numberArray1[i] = (float)dum1;
        numberArray2[i] = (float)dum2;
        numberArray3[i] = (int)dum3;
        printf("%f\t%f\t%u\n", numberArray1[i], numberArray2[i], numberArray3[i]);
    }

    printf("%s\n", buffer);
    printf("%u\n", counter);


    fclose(myFile);

    return 0;
}