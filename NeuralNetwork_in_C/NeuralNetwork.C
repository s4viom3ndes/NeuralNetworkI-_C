// XOR function -> input (0,0) -> 0 ; (1,0) -> 1; (0,1) - > 1; (1,1) -> 0
// We are going to use some wheihts and layers
// Activation funtion
// Add some bias
// add fully coneccted layer
// the acctivation funciont is a somatoryy of the input times data. plus bias (sigmoid).

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Simple NN that can learn xor


double sigmoid(double x){
    return (1 / 1 + exp(-x));
}
double dSigmoid(double x){
    return x * (1 - x);
}

double init_weigths(){return((double)rand())/((double)RAND_MAX);}


void shuffle(int *array, size_t n){
    if (n>1){
     size_t i;
     for (i = 0; i<n-1; i++){
        size_t j = i + rand() / (RAND_MAX / (n-1) + 1);
        int t = array[j];
        array[j] = array[i];
        array[i] = t;
     }
    }
}


#define numInputs 2
#define numHiddenNodes 2
#define numOutpus 1
#define numTrainingSets 4

int main(void){

    const double lr = 0.1f; // learning rate

    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutpus];

    double hiddenLayerBias[numHiddenNodes];
    double outputlayersbias[numOutpus];

    double hiddenWeigths[numInputs][numHiddenNodes];
    double outputsWeigths[numHiddenNodes][numOutpus];

    double training_inputs[numTrainingSets][numInputs] = {{0.0f, 0.0f}, // Variaveis idependentes
                                                          {1.0f, 0.0f},
                                                          {0.0f, 1.0f},
                                                          {1.0f, 1.0f}};


    double training_outputs[numTrainingSets][numOutpus] = {{0.0f}, // Variaveis dependentes (Y - verdade)
                                                          {1.0f},
                                                          {0.0f},
                                                          {1.0f}};

    for(int i = 0; i < numInputs; i++){
        for(int j = 0; j< numHiddenNodes; j++){
            hiddenWeigths[i][j] = init_weigths(); // inicializando os pesos na rede

        }
    }

    for(int i = 0; i < numHiddenNodes; i++){
        for(int j = 0; j< numOutpus; j++){
            outputsWeigths[i][j] = init_weigths(); // inicializando os pesos na rede

        }
    }

    for(int i = 0; i<numOutpus; i++){
        outputlayersbias[i] = init_weigths();
    }

    int trainingSetOrder[] = {0,1,2,3};

    int numberOfEpochs = 1000;

    //Train the NN for a certain number of epochs

    for(int epoch = 0; epoch < numberOfEpochs; epoch++){
        shuffle(trainingSetOrder, numTrainingSets);

        for(int x = 0 ; x < numTrainingSets; x++){
           int i = trainingSetOrder[x];

           //Forward pass

           //Compute Layer acctivation

           for(int j = 0; j< numHiddenNodes; j++){
                double activation = hiddenLayerBias[j];

                for(int k = 0; k < numInputs; k++){
                    activation += training_inputs[i][k] * hiddenWeigths[k][j]; 
                }

                hiddenLayer[j] = sigmoid(activation);
            }

            
           //Compute output Layer activation

           for(int j = 0; j< numOutpus; j++){
                double activation = outputlayersbias[j];

                for(int k = 0; k < numInputs; k++){
                    activation += hiddenLayer[k] * outputsWeigths[k][j]; 
                }

                outputLayer[j] = sigmoid(activation);
            }

            printf("Input: %g  Output: %g  Predicted Output: %g \n",
            training_inputs[i][0], training_inputs[i][1],
            outputLayer[0], training_outputs[i][0]);


            //Back-propagation
        
            // Compute change in output weigths

            double deltaOutput[numOutpus];

            for(int j = 0; j<numOutpus; j++){
                double error = (training_outputs[i][j] - outputLayer[j]);

                deltaOutput[j] = error * dSigmoid(outputLayer[j]);
            } 

            //Compute change in hidden Layer
            double deltaHidden[numHiddenNodes];

            for(int j = 0; j<numHiddenNodes; j++){
                double error = 0.0f;
                for(int k = 0; k < numOutpus; k++){
                    error += deltaOutput[k] * outputsWeigths[j][k];
                }
                deltaHidden[j] = error * dSigmoid(hiddenLayer[j]);
            }

            //Apply change in output weights

            for(int j = 0; j < numOutpus; j++){
                outputlayersbias[j] += deltaOutput[j] * lr;
                for(int k = 0; k < numHiddenNodes; k++){
                    outputsWeigths[k][j] += hiddenLayer[k] * deltaOutput[j] * lr; 
                }
            }

            //Apply change in hidden weights

            for(int j = 0; j < numHiddenNodes; j++){
                hiddenLayer[j] += deltaHidden[j] * lr;
                for(int k = 0; k < numInputs; k++){
                    hiddenWeigths[k][j] += training_inputs[i][k] * deltaHidden[j] * lr; 
                }
            }

            // Show the result

            fputs("\nFinal hidden Weights\n[", stdout);
            for(int j = 0; j < numHiddenNodes; j++){
                fputs("[ ", stdout);
                for(int k = 0; k < numInputs; k++){
                    printf("%f ", hiddenWeigths[k][j]);
                }
                fputs("] ", stdout);
            }            


    
            fputs("]\nFinal hidden Layers Biases\n[", stdout);
            for(int j = 0; j < numHiddenNodes; j++){
                printf("%f", hiddenLayerBias[j]);
            }

            fputs("]\nFinal Output Biases\n[", stdout);
            for(int j = 0; j < numOutpus; j++){
                printf("%f", outputlayersbias[j]);
            }

        }
    } 
}

