using System.Diagnostics;
using neuronka.dataLoading;
using neuronka.exporter;

namespace neuronka;

internal class Program
{
    private static void Main()
    {
        var fullTimer = Stopwatch.StartNew();
        var projectRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", ".."));
        
        // LOADING - Keep test data sealed until final evaluation
        var loaderTimer = Stopwatch.StartNew();
        var (trainData, testData) = LoadData(projectRoot);
        loaderTimer.Stop();
        
        var rand = new Random(42); // Fixed seed for reproducibility

        // SPLIT TRAINING DATA: 90% train, 10% validation
        var ((trainImages, trainLabels), (valImages, valLabels)) =
            DataLoader.SplitValidationSet(trainData.Images, trainData.Labels, rand);
        
        Console.WriteLine($"Loaded {trainLabels.Length} training samples, {valLabels.Length} validation samples");
        Console.WriteLine($"Kept {testData.Labels.Length} test samples sealed for final evaluation");
        Console.WriteLine($"Data loaded in {loaderTimer.ElapsedMilliseconds} ms\n");

        // HYPERPARAMETER TUNING (grid search on validation set only)
        var tuner = new HyperparameterTuner(rand, projectRoot);
        
        // Define search space (start small!)
        var searchResults = tuner.GridSearch(
            learningRates: new List<float> { 0.01f, 0.005f, 0.001f },
            decayRates: new List<float> { 0.9f, 0.95f },
            momentumBetas: new List<float> { 0.9f, 0.95f },
            batchSizes: new List<int> { 32, 64 },
            weightDecays: new List<float> { 0f, 5e-4f },
            trainImages, trainLabels,
            valImages, valLabels,
            epochs: 6
        );

        // FINAL TRAINING: Combine train+val and retrain with best hyperparameters
        Console.WriteLine("\n=== Final Training with Best Hyperparameters ===");
        
        // Apply best config
        var bestConfig = searchResults.bestConfig;
        TrainingSettings.LearningRate = bestConfig.LearningRate;
        TrainingSettings.DecayRate = bestConfig.DecayRate;
        TrainingSettings.MomentumBeta = bestConfig.MomentumBeta;
        TrainingSettings.BatchSize = bestConfig.BatchSize;
        TrainingSettings.WeightDecay = bestConfig.WeightDecay;
        TrainingSettings.Epochs = 10; // More epochs for final model
        
        // Combine training and validation sets for maximum training data
        var (combinedImages, combinedLabels) = CombineDatasets(trainImages, trainLabels, valImages, valLabels);
        
        // Create fresh network
        var finalNetwork = new NeuralNetwork();
        finalNetwork.AddLayer(new Layer(rand, "hidden1", 784, 256));
        finalNetwork.AddLayer(new Layer(rand, "hidden2", 256, 128));
        finalNetwork.AddLayer(new Layer(rand, "output", 128, 10, "softmax"));
        
        // Train final model (no validation needed in final pass)
        finalNetwork.Train(combinedImages, combinedLabels, new float[10,0], new int[0], rand);
        
        // FINAL EVALUATION: Test set used exactly ONCE
        Console.WriteLine("\n=== Final Evaluation on Test Set ===");
        var (testAcc, testPred) = ModelTester.TestModel(finalNetwork, testData.Images, testData.Labels);
        Exporter.ExportTest(projectRoot, testPred);
        Console.WriteLine($"\nFinal Test Accuracy: {testAcc:P2}");
        
        // Report total time
        var elapsed = fullTimer.Elapsed;
        Console.WriteLine($"\nTotal runtime: {(int)elapsed.TotalMinutes}min {elapsed.Seconds}s");
    }

    private static ((float[,] Images, int[] Labels) Train, (float[,] Images, int[] Labels) Test) LoadData(string projectRoot)
    {
        var loader = new DataLoader(projectRoot);
        return loader.LoadData();
    }

    private static (float[,] combinedImages, int[] combinedLabels) CombineDatasets(
        float[,] trainImages, int[] trainLabels, 
        float[,] valImages, int[] valLabels)
    {
        int features = trainImages.GetLength(0);
        int totalSamples = trainLabels.Length + valLabels.Length;
        
        var combinedImages = new float[features, totalSamples];
        var combinedLabels = new int[totalSamples];
        
        // Copy training data
        for (int i = 0; i < trainLabels.Length; i++)
        {
            combinedLabels[i] = trainLabels[i];
            for (int f = 0; f < features; f++)
                combinedImages[f, i] = trainImages[f, i];
        }
        
        // Copy validation data
        int offset = trainLabels.Length;
        for (int i = 0; i < valLabels.Length; i++)
        {
            combinedLabels[offset + i] = valLabels[i];
            for (int f = 0; f < features; f++)
                combinedImages[f, offset + i] = valImages[f, i];
        }
        
        return (combinedImages, combinedLabels);
    }
}