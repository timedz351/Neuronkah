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

  public void Backward(float[,] X, int[] Y, float learningRate, int batchSize)
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
      // Update after computing dA_prev
      layer.UpdateParameters(dW, db, learningRate);
      if (i > 0)
        dA = dA_prev;
    }
  }

  public void Train(float[,] X, int[] Y, float learningRate, float decayRate, int stepSize, int iterations, int batchSize = 64,
                  LearningRateScheduler.ScheduleType scheduleType = LearningRateScheduler.ScheduleType.Constant)
  {
    var epochTimer = Stopwatch.StartNew();
    // Track time between logs to compute per-epoch average accurately, even at iter=0
    long lastLogMs = 0L;
    int lastLogIter = -1;
    const int logEvery = 10;
    int m = Y.Length;

    // Initialize learning rate scheduler
    var scheduler = new LearningRateScheduler(scheduleType, learningRate, decayRate: 0.85f, stepSize: 5);

    int batchesPerEpoch = (int)Math.Ceiling((double)m / batchSize);

    for (int iter = 0; iter < iterations; iter++)
    {
      // Get current learning rate from scheduler
      float currentLearningRate = scheduler.GetLearningRate();

      // Shuffle data at the start of each epoch
      var (_, X_shuffled, Y_shuffled) = ShuffleData(X, Y);
      float epochLoss = 0f;
      int batchCount = 0;

      // Process each batch in the epoch
      for (int batchIndex = 0; batchIndex < batchesPerEpoch; batchIndex++)
      {
        var (X_batch, Y_batch) = GetBatch(X_shuffled, Y_shuffled, batchSize, batchIndex, iter);
        int currentBatchSize = Y_batch.Length;

        // Forward pass
        float[,] output = Forward(X_batch);

        // Calculate batch loss
        float batchLoss = CalculateLoss(output, Y_batch);
        epochLoss += batchLoss;
        batchCount++;

        // Backward pass with current learning rate
        Backward(X_batch, Y_batch, currentLearningRate, currentBatchSize);
      }

      epochLoss /= batchCount;

      // Logging every 'logEvery' epochs
      if (iter % logEvery == 0)
      {
        float[,] output = Forward(X);
        int[] predictions = GetPredictions(output);
        float acc = GetAccuracy(predictions, Y);
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
        Console.WriteLine($"Epoch {iter}, Loss: {epochLoss:F4}, Accuracy: {acc:P2}, " +
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
  private (float[,] X_batch, int[] Y_batch) GetBatch(float[,] X, int[] Y, int batchSize, int batchIndex, int epoch)
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
      int dataIndex = start + j;
      Y_batch[j] = Y[dataIndex];

      for (int i = 0; i < features; i++)
      {
        X_batch[i, j] = X[i, dataIndex];
      }
    }

    return (X_batch, Y_batch);
  }

  private (int[] indices, float[,] X_shuffled, int[] Y_shuffled) ShuffleData(float[,] X, int[] Y)
  {
    int m = Y.Length;
    int features = X.GetLength(0);

    // Create index array and shuffle it
    int[] indices = new int[m];
    for (int i = 0; i < m; i++)
      indices[i] = i;

    // Fisher-Yates shuffle
    var rand = new Random();
    for (int i = m - 1; i > 0; i--)
    {
      int j = rand.Next(i + 1);
      (indices[i], indices[j]) = (indices[j], indices[i]);
    }

    // Create shuffled datasets
    float[,] X_shuffled = new float[features, m];
    int[] Y_shuffled = new int[m];

    for (int newIdx = 0; newIdx < m; newIdx++)
    {
      int oldIdx = indices[newIdx];
      Y_shuffled[newIdx] = Y[oldIdx];

      for (int f = 0; f < features; f++)
      {
        X_shuffled[f, newIdx] = X[f, oldIdx];
      }
    }

    return (indices, X_shuffled, Y_shuffled);
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
