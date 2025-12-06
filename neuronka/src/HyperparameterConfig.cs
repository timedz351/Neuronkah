namespace neuronka;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

public class HyperparameterConfig
{
    public float LearningRate { get; set; }
    public float DecayRate { get; set; }
    public float MomentumBeta { get; set; }
    public int BatchSize { get; set; }
    public float WeightDecay { get; set; }
    public int Hidden1Size { get; set; } = 256;
    public int Hidden2Size { get; set; } = 128;
}

public class HyperparameterTuner
{
    private Random _rand;
    private string _projectRoot;
    
    public HyperparameterTuner(Random rand, string projectRoot)
    {
        _rand = rand;
        _projectRoot = projectRoot;
    }

    public (HyperparameterConfig bestConfig, float bestValAcc, List<(HyperparameterConfig config, float valAcc)> results) 
        GridSearch(
            List<float> learningRates,
            List<float> decayRates,
            List<float> momentumBetas,
            List<int> batchSizes,
            List<float> weightDecays,
            float[,] trainImages, int[] trainLabels,
            float[,] valImages, int[] valLabels,
            int epochs = 6)
    {
        var results = new List<(HyperparameterConfig config, float valAcc)>();
        HyperparameterConfig bestConfig = null;
        float bestValAcc = 0f;
        int trial = 0;
        
        Console.WriteLine($"Starting grid search over {learningRates.Count * decayRates.Count * momentumBetas.Count * batchSizes.Count * weightDecays.Count} combinations...\n");

        foreach (var lr in learningRates)
        {
            foreach (var decay in decayRates)
            {
                foreach (var beta in momentumBetas)
                {
                    foreach (var bs in batchSizes)
                    {
                        foreach (var wd in weightDecays)
                        {
                            trial++;
                            var config = new HyperparameterConfig
                            {
                                LearningRate = lr,
                                DecayRate = decay,
                                MomentumBeta = beta,
                                BatchSize = bs,
                                WeightDecay = wd
                            };

                            Console.WriteLine($"Trial {trial}: LR={lr}, Decay={decay}, Beta={beta}, BS={bs}, WD={wd}");
                            var valAcc = RunTrial(config, trainImages, trainLabels, valImages, valLabels, epochs);
                            results.Add((config, valAcc));

                            if (valAcc > bestValAcc)
                            {
                                bestValAcc = valAcc;
                                bestConfig = config;
                                Console.WriteLine($"→ New best validation accuracy: {valAcc:P2}\n");
                            }
                            else
                            {
                                Console.WriteLine($"→ Validation accuracy: {valAcc:P2}\n");
                            }
                        }
                    }
                }
            }
        }

        Console.WriteLine($"Grid search complete. Best config: Validation Acc={bestValAcc:P2}");
        Console.WriteLine($"LR={bestConfig.LearningRate}, Decay={bestConfig.DecayRate}, Beta={bestConfig.MomentumBeta}, BS={bestConfig.BatchSize}, WD={bestConfig.WeightDecay}");
        
        return (bestConfig, bestValAcc, results);
    }

    private float RunTrial(HyperparameterConfig config, float[,] trainImages, int[] trainLabels, 
                          float[,] valImages, int[] valLabels, int epochs)
    {
        // Apply config to global settings
        TrainingSettings.LearningRate = config.LearningRate;
        TrainingSettings.DecayRate = config.DecayRate;
        TrainingSettings.MomentumBeta = config.MomentumBeta;
        TrainingSettings.BatchSize = config.BatchSize;
        TrainingSettings.WeightDecay = config.WeightDecay;
        TrainingSettings.Epochs = epochs;

        // Create fresh network for each trial
        var network = new NeuralNetwork();
        network.AddLayer(new Layer(_rand, "hidden1", 784, 256));
        network.AddLayer(new Layer(_rand, "hidden2", 256, 128));
        network.AddLayer(new Layer(_rand, "output", 128, 10, "softmax"));

        // Train on training set only
        network.Train(trainImages, trainLabels, valImages, valLabels, _rand);

        // Evaluate on validation set only
        var (valAcc, _) = ModelTester.TestModel(network, valImages, valLabels);
        return valAcc;
    }
}