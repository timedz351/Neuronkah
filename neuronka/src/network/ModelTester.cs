namespace neuronka;

public class ModelTester
{
    public static float TestModel(NeuralNetwork network, float[,] X, int[] Y)
    {
        float[,] output = network.Forward(X);
        int[] predictions = network.GetPredictions(output);
        float accuracy = network.GetAccuracy(predictions, Y);
        return accuracy;
    }
}



