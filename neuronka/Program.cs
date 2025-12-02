using System.Diagnostics;
using neuronka.dataLoading;
using neuronka.exporter;

namespace neuronka;

internal class Program
{


    private static void Main()
    {
        // LOADING
        var fullTimer = Stopwatch.StartNew();
        var projectRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", ".."));
        var loaderTimer = Stopwatch.StartNew();
        var (trainData, testData) = LoadData(projectRoot);

        // SPLIT TO TRAIN AND VALIDATION SET
        var ((trainImages, trainLabels), (valImages, valLabels)) =
            DataLoader.SplitValidationSet(trainData.Images, trainData.Labels, 0.1f);

        // Configure global training settings (adjust as desired)
        TrainingSettings.LogEvery = 1;
        TrainingSettings.BatchSize = 32;
        TrainingSettings.Epochs = 6;
        TrainingSettings.MomentumType = MomentumType.Classical;
        TrainingSettings.MomentumBeta = 0.95f;
        TrainingSettings.LearningRate = 0.01f;
        TrainingSettings.DecayRate = 0.9f;
        TrainingSettings.StepSize = 1;
        TrainingSettings.ScheduleType = LearningRateScheduler.ScheduleType.StepDecay;
        TrainingSettings.WeightDecay = 5e-4f; // mild L2 regularization for better generalization

        var (testImages, testLabels) = testData;
        Console.WriteLine($"Loaded {trainLabels.GetLength(0)} images and labels for training");
        Console.WriteLine($"Loaded {testImages.GetLength(1)} images and labels for testing");
        loaderTimer.Stop();
        Console.WriteLine($"Data loaded in {loaderTimer.ElapsedMilliseconds} ms");

        // TRAINING SETUP
        var batchSize = TrainingSettings.BatchSize;
        Console.WriteLine($"Batch size: {batchSize}, Training set: {trainLabels.GetLength(0)} samples");

        // CREATE NETWORK
        var rand = new Random();
        var network = new NeuralNetwork();
        network.AddLayer(new Layer(rand, "hidden1", 784, 256));
        network.AddLayer(new Layer(rand, "hidden2", 256, 128));
        network.AddLayer(new Layer(rand, "output", 128, 10, "softmax"));


        // TRAIN MODEL with learning rate scheduling
        var trainingTimer = Stopwatch.StartNew();
        network.Train(trainImages, trainLabels, valImages, valLabels);
        trainingTimer.Stop();
        Console.WriteLine($"Training completed in {trainingTimer.ElapsedMilliseconds} ms");

        // TEST MODEL
        var (valAcc, valPred) = ModelTester.TestModel(network, valImages, valLabels);
        Exporter.ExportTrain(projectRoot, valPred);
        Console.WriteLine($"Validation accuracy: {valAcc:P2}");

        var (testAcc, testPred) = ModelTester.TestModel(network, testData.Images, testData.Labels);
        Exporter.ExportTest(projectRoot, testPred);
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
