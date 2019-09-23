#include <stdio.h>
#include <stdlib.h>

int main(){

    FILE *myFile;
    myFile = fopen("test.txt", "r");

    //read file into array
    uint numPts = 100 * 3;
    float numberArray1[numPts];
    float numberArray2[numPts];
    int numberArray3[numPts];
    int i;

    if (myFile == NULL){
        printf("Error Reading File\n");
        exit (0);
    }

    char buffer[100];
    fgets(buffer, 100, myFile);

    float dummy;

    for (i = 0; i < (int)numPts/3.; i++){
        fscanf(myFile, "%f", &numberArray1[i] );
        fscanf(myFile, "%f", &numberArray2[i] );
        fscanf(myFile, "%f", &dummy);
        numberArray3[i] = (int)dummy;
    }

    printf("%s", buffer);
    for (i = 0; i < (int)numPts/3.; i++){
        printf("%f\t%f\t%u\n", numberArray1[i], numberArray2[i], numberArray3[i]);
    }


    fclose(myFile);

    return 0;
}