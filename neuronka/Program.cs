namespace neuronka;

using System;
using System.Diagnostics;
using System.IO;
using neuronka.dataLoading;


class Program
{
    private static int _batchSize = 60000;
    private static int _iterations = 250;
    private static float _alpha = 0.075f;

    static void Main()
    {
        // LOADING
        var fullTimer = Stopwatch.StartNew();
        string projectRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", ".."));
        var loaderTimer = Stopwatch.StartNew();
        var (trainData, testData) = LoadData(projectRoot);
        var (trainImages, trainLabels) = trainData;
        var (testImages, testLabels) = testData;
        Console.WriteLine($"Loaded {trainImages.GetLength(1)} images and labels for training");
        Console.WriteLine($"Loaded {testImages.GetLength(1)} images and labels for testing");
        loaderTimer.Stop();
        Console.WriteLine($"Data loaded in {loaderTimer.ElapsedMilliseconds} ms");

        // CREATE SMALLER BATCH
        int batchSize = _batchSize;
        float[,] X_batch = new float[trainImages.GetLength(0), batchSize];
        int[] Y_batch = new int[batchSize];
        for (int j = 0; j < batchSize; j++)
        {
            for (int i = 0; i < trainImages.GetLength(0); i++)
                X_batch[i, j] = trainImages[i, j];
            Y_batch[j] = trainLabels[j];
        }
        Console.WriteLine($"X_Batch size: {X_batch.GetLength(0)}x{X_batch.GetLength(1)}");
        Console.WriteLine($"Y_Batch size: {Y_batch.GetLength(0)}");

        // CREATE NETWORK
        var network = new NeuralNetwork("cross_entropy");
        network.AddLayer(new Layer("hidden1", 784, 30, "relu"));
        network.AddLayer(new Layer("output", 30, 10, "softmax"));

        // TRAIN MODEL
        var trainingTimer = Stopwatch.StartNew();
        network.Train(X_batch, Y_batch, _alpha, _iterations, batchSize);
        trainingTimer.Stop();
        Console.WriteLine($"Training completed in {trainingTimer.ElapsedMilliseconds} ms");

        // TEST MODEL
        var accuracy = ModelTester.TestModel(network, testImages, testLabels);
        Console.WriteLine($"Test accuracy: {accuracy:P2}\n");
        Console.WriteLine($"Model ran in {fullTimer.ElapsedMilliseconds / 60000} min");
    }

    static ((float[,] Images, int[] Labels) Train, (float[,] Images, int[] Labels) Test) LoadData(string projectRoot)
    {
        var loader = new DataLoader(projectRoot);
        var data = loader.LoadData();
        return data;
    }
}
