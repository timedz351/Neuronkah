namespace neuronka;

using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;

public class NeuralNetwork
{
  public List<Layer> Layers { get; }
  public string LossFunction { get; }

  public NeuralNetwork(string lossFunction = "cross_entropy")
  {
    Layers = new List<Layer>();
    LossFunction = lossFunction;
  }

  public void AddLayer(Layer layer)
  {
    Layers.Add(layer);
  }

  public float[,] Forward(float[,] X)
  {
    float[,] activation = X;

    foreach (var layer in Layers)
    {
      activation = layer.Forward(activation);
    }

    return activation;
  }

  public void Backward(float[,] X, int[] Y, float learningRate, int batchSize, float momentumBeta)
  {
    // Convert labels to one-hot encoding for the output layer
    float[,] Y_onehot = OneHot(Y, Layers[^1].OutputSize);

    // Initial gradient for output layer (softmax + cross entropy): A - Y
    float[,] dA = MatrixUtils.Subtract(Layers[^1].A, Y_onehot);

    // Backward pass through layers
    for (int i = Layers.Count - 1; i >= 0; i--)
    {
      var layer = Layers[i];
      float[,] prevActivation = i == 0 ? X : Layers[i - 1].A;

      // Backward for this layer: returns gradients and dA for previous layer
      var (dW, db, dA_prev) = layer.Backward(dA, prevActivation, batchSize);
      layer.UpdateParameters(dW, db, learningRate, momentumBeta);
      if (i > 0)
        dA = dA_prev;
    }
  }

  public void Train(float[,] X_train, int[] Y_train, float[,] X_val, int[] Y_val, float learningRate, float decayRate, int stepSize, int iterations, int batchSize = 64,
                  LearningRateScheduler.ScheduleType scheduleType = LearningRateScheduler.ScheduleType.Constant,
                  float momentumBeta = 0f)
  {
    var epochTimer = Stopwatch.StartNew();
    // Track time between logs to compute per-epoch average accurately, even at iter=0
    long lastLogMs = 0L;
    int lastLogIter = -1;
    const int logEvery = 10;
    int m = Y_train.Length;
    float bestValAcc = 0f;
    int patience = 0;
    
    // Initialize learning rate scheduler with provided decayRate & stepSize
    var scheduler = new LearningRateScheduler(scheduleType, learningRate, decayRate: decayRate, stepSize: stepSize);

    int batchesPerEpoch = (int)Math.Ceiling((double)m / batchSize);

    for (int iter = 0; iter < iterations; iter++)
    {
      // Get current learning rate from scheduler
      float currentLearningRate = scheduler.GetLearningRate();

      // Shuffle data at the start of each epoch
      int[] shuffledIndices = ShuffleIndices(m, iter);      
      float epochLoss = 0f;
      int batchCount = 0;

      // Process each batch in the epoch
      for (int batchIndex = 0; batchIndex < batchesPerEpoch; batchIndex++)
      {
        var (X_batch, Y_batch) =GetBatch(X_train, Y_train, shuffledIndices,batchSize, batchIndex);
        int currentBatchSize = Y_batch.Length;

        // Forward pass
        float[,] output = Forward(X_batch);

        // Calculate batch loss
        float batchLoss = CalculateLoss(output, Y_batch);
        epochLoss += batchLoss;
        batchCount++;

        // Backward pass with current learning rate
        Backward(X_batch, Y_batch, currentLearningRate, currentBatchSize, momentumBeta);
      }

      epochLoss /= batchCount;

      // Logging every 'logEvery' epochs
      if (iter % logEvery == 0)
      {
        // Evaluate on validation set (NO BACKWARD PASS!)
        float[,] valOutput = Forward(X_val);
        int[] valPreds = GetPredictions(valOutput);
        float valAcc = GetAccuracy(valPreds, Y_val);
            
        // Evaluate on training set for comparison
        float[,] trainOutput = Forward(X_train);
        int[] trainPreds = GetPredictions(trainOutput);
        float trainAcc = GetAccuracy(trainPreds, Y_train);

        Console.WriteLine($"Epoch {iter} | Train: {trainAcc:P2} | Val: {valAcc:P2}");
        // Early stopping
        if (valAcc > bestValAcc + 0.001f)
        {
          bestValAcc = valAcc;
          patience = 0;
        }
        else if (++patience >= 10)
        {
          Console.WriteLine($"Early stopping at epoch {iter}");
          break;
        }
        
        // Compute ETA based on average epoch time since last log.
        long nowMs = epochTimer.ElapsedMilliseconds;
        int epochsSinceLastLog = iter - lastLogIter; // at iter=0 => 1
        long segmentMs = nowMs - lastLogMs;
        double avgEpochMs = epochsSinceLastLog > 0 ? (double)segmentMs / epochsSinceLastLog : 0.0;
        int remainingEpochs = Math.Max(0, iterations - (iter + 1));
        double etaSeconds = (avgEpochMs * remainingEpochs) / 1000.0;
        var etaSpan = TimeSpan.FromSeconds(etaSeconds);
        int mins = (int)etaSpan.TotalMinutes;
        int secs = etaSpan.Seconds;
        // Update log anchors
        lastLogMs = nowMs;
        lastLogIter = iter;
        
        // Include current learning rate in logging
        Console.WriteLine($"Epoch {iter}, Loss: {epochLoss:F4}, Train: {trainAcc:P2} | Val: {valAcc:P2}, " +
                          $"LR: {currentLearningRate:E3}, " +
                          $"Time {epochTimer.ElapsedMilliseconds / 1000}s, " +
                          $"ETA: {mins}m {secs:0}s");
      }
    }
  }

  public int[] GetPredictions(float[,] output)
  {
    int rows = output.GetLength(0);
    int cols = output.GetLength(1);
    int[] predictions = new int[cols];

    for (int j = 0; j < cols; j++)
    {
      float max = float.MinValue;
      int maxIndex = 0;
      for (int i = 0; i < rows; i++)
      {
        if (output[i, j] > max)
        {
          max = output[i, j];
          maxIndex = i;
        }
      }
      predictions[j] = maxIndex;
    }
    return predictions;
  }

  public float GetAccuracy(int[] predictions, int[] Y)
  {
    int correct = 0;
    for (int i = 0; i < Y.Length; i++)
      if (predictions[i] == Y[i])
        correct++;

    return (float)correct / Y.Length;
  }

  private float[,] OneHot(int[] Y, int numClasses)
  {
    int m = Y.Length;
    var oneHot = new float[numClasses, m];
    for (int i = 0; i < m; i++)
      oneHot[Y[i], i] = 1f;
    return oneHot;
  }

  private float[,] ExtractBatch(float[,] X, int start, int batchSize)
  {
    int features = X.GetLength(0);
    var batch = new float[features, batchSize];

    for (int j = 0; j < batchSize; j++)
    {
      for (int i = 0; i < features; i++)
      {
        batch[i, j] = X[i, start + j];
      }
    }

    return batch;
  }
  private (float[,] X_batch, int[] Y_batch) GetBatch(float[,] X, int[] Y,int[] shuffledIndices, int batchSize, int batchIndex)
  {
    int m = Y.Length;
    int features = X.GetLength(0);

    int start = batchIndex * batchSize;
    int end = Math.Min(start + batchSize, m);
    int currentBatchSize = end - start;

    float[,] X_batch = new float[features, currentBatchSize];
    int[] Y_batch = new int[currentBatchSize];

    // Copy batch data
    for (int j = 0; j < currentBatchSize; j++)
    {
      int dataIndex = shuffledIndices[start + j];
      Y_batch[j] = Y[dataIndex];
      for (int f = 0; f < features; f++)
      {
        X_batch[f, j] = X[f, dataIndex];
      }
    }

    return (X_batch, Y_batch);
  }

/*
 * You have 60,000 training examples. You want to randomize their order each epoch. Instead of moving all the data,
 * you just shuffle numbers on a sticky note that say which example to use next.
 * The function creates a list [0, 1, 2, ..., 59999] and scrambles it to something
 * like [342, 5, 18000, ...]. When making a batch, you look at this list and say "okay, first use example #342, then #5,
 * then #18000..."
 */
  private int[] ShuffleIndices(int m, int epoch)
  {
    int[] indices = Enumerable.Range(0, m).ToArray();
    var rand = new Random(42 + epoch); // Fixed seed + epoch for reproducibility
    for (int i = m - 1; i > 0; i--)
    {
      int j = rand.Next(i + 1);
      (indices[i], indices[j]) = (indices[j], indices[i]);
    }
    return indices;
  }

  public float CalculateLoss(float[,] predictions, int[] Y)
  {
    int numClasses = predictions.GetLength(0);
    int m = Y.Length;
    float loss = 0f;

    // Cross-entropy loss
    for (int j = 0; j < m; j++)
    {
      int trueClass = Y[j];
      float predictedProb = predictions[trueClass, j];

      // Add small epsilon to avoid log(0)
      loss += -MathF.Log(predictedProb + 1e-8f);
    }

    return loss / m;
  }

  // Overload for one-hot encoded labels
  public float CalculateLoss(float[,] predictions, float[,] Y_onehot)
  {
    int m = predictions.GetLength(1);
    float loss = 0f;

    for (int j = 0; j < m; j++)
    {
      for (int i = 0; i < predictions.GetLength(0); i++)
      {
        if (Y_onehot[i, j] > 0.5f) // If this is the true class
        {
          float predictedProb = predictions[i, j];
          loss += -MathF.Log(predictedProb + 1e-8f);
          break; // Only one true class per sample
        }
      }
    }

    return loss / m;
  }
}
