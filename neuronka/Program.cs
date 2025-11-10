namespace neuronka;

using System;
using System.Diagnostics;
using System.IO;
using neuronka.dataLoading;


class Program
{
    private static int _batchSize = 16;
    private static int _iterations = 85;
    private static float _alpha = 0.001f;

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

        // TRAIN ON FULL DATA WITH MINI-BATCHES (no upfront copying)
        int batchSize = _batchSize;
        Console.WriteLine($"Batch size: {batchSize}, Training set: {trainImages.GetLength(1)} samples");

        // CREATE NETWORK
        var rand = new Random(42);
        var network = new NeuralNetwork("cross_entropy");
        network.AddLayer(new Layer(rand, "hidden1", 784, 128, "relu"));
        // network.AddLayer(new Layer(rand, "output", 256, 128, "relu"));
        network.AddLayer(new Layer(rand, "hidden2", 128, 64, "relu"));
        network.AddLayer(new Layer(rand, "output", 64, 10, "softmax"));

        // TRAIN MODEL
        var trainingTimer = Stopwatch.StartNew();
        network.Train(trainImages, trainLabels, _alpha, _iterations, batchSize);
        trainingTimer.Stop();
        Console.WriteLine($"Training completed in {trainingTimer.ElapsedMilliseconds} ms");

        // TEST MODEL
        var accuracy = ModelTester.TestModel(network, testImages, testLabels);
        Console.WriteLine($"Test accuracy: {accuracy:P2}\n");
        var elapsed = fullTimer.Elapsed;
        int minutes = (int)elapsed.TotalMinutes;
        int seconds = elapsed.Seconds;
        Console.WriteLine($"Model ran in {minutes}min {seconds}s");
    }

    static ((float[,] Images, int[] Labels) Train, (float[,] Images, int[] Labels) Test) LoadData(string projectRoot)
    {
        var loader = new DataLoader(projectRoot);
        var data = loader.LoadData();
        return data;
    }
}
