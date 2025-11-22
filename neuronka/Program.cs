using System.Diagnostics;
using neuronka.dataLoading;

namespace neuronka;

internal class Program
{
    private static readonly int _batchSize = 32;
    private static readonly int _iterations = 10;
    private static readonly float _alpha = 0.2f;
    private static readonly float _decayRate = 0.85f;
    private static readonly int _stepSize = 2;
    private static readonly float _momentumBeta = 0.95f;

    private static readonly LearningRateScheduler.ScheduleType _scheduleType =
        LearningRateScheduler.ScheduleType.StepDecay;

    
    private static void Main()
    {
        // LOADING
        var fullTimer = Stopwatch.StartNew();
        var projectRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", ".."));
        var loaderTimer = Stopwatch.StartNew();
        var (trainData, testData) = LoadData(projectRoot);
        // AFTER loading
        var ((trainImages, trainLabels), (valImages, valLabels)) = 
            DataLoader.SplitValidationSet(trainData.Images, trainData.Labels, 0.1f);

        var (testImages, testLabels) = testData;
        Console.WriteLine($"Loaded {trainLabels.GetLength(0)} images and labels for training");
        Console.WriteLine($"Loaded {testImages.GetLength(1)} images and labels for testing");
        loaderTimer.Stop();
        Console.WriteLine($"Data loaded in {loaderTimer.ElapsedMilliseconds} ms");

        // TRAIN ON FULL DATA WITH MINI-BATCHES (no upfront copying)
        var batchSize = _batchSize;
        Console.WriteLine($"Batch size: {batchSize}, Training set: {trainLabels.GetLength(0)} samples");
        
        // CREATE NETWORK
        var rand = new Random();
        var network = new NeuralNetwork();
        network.AddLayer(new Layer(rand, "hidden1", 784, 128));
        // network.AddLayer(new Layer(rand, "hidden2", 256, 128, "relu"));
        // network.AddLayer(new Layer(rand, "hidden3", 128, 64, "relu"));
        network.AddLayer(new Layer(rand, "output", 128, 10, "softmax"));


        // TRAIN MODEL with learning rate scheduling
        var trainingTimer = Stopwatch.StartNew();
        network.Train(trainImages, trainLabels, valImages,valLabels, _alpha, _decayRate, _stepSize, _iterations, _batchSize, _scheduleType,
            _momentumBeta);
        trainingTimer.Stop();
        Console.WriteLine($"Training completed in {trainingTimer.ElapsedMilliseconds} ms");
        
        // TEST MODEL
        float valAcc = ModelTester.TestModel(network, valImages, valLabels);
        Console.WriteLine($"Validation accuracy: {valAcc:P2}");
        
        float testAcc = ModelTester.TestModel(network, testData.Images, testData.Labels);
        Console.WriteLine($"Final test accuracy: {testAcc:P2}");
        
        var elapsed = fullTimer.Elapsed;
        var minutes = (int)elapsed.TotalMinutes;
        var seconds = elapsed.Seconds;
        Console.WriteLine($"Model ran in {minutes}min {seconds}s");
    }

    private static ((float[,] Images, int[] Labels) Train, (float[,] Images, int[] Labels) Test) LoadData(
        string projectRoot)
    {
        var loader = new DataLoader(projectRoot);
        var data = loader.LoadData();
        return data;
    }
}