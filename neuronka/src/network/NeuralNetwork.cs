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

    // Calculate initial gradient for output layer
    float[,] dA = MatrixUtils.Subtract(Layers[^1].A, Y_onehot);

    // Backward pass through layers
    for (int i = Layers.Count - 1; i >= 0; i--)
    {
      var layer = Layers[i];
      float[,] prevActivation = i == 0 ? X : Layers[i - 1].A;

      var (dW, db) = layer.Backward(dA, prevActivation, batchSize);
      layer.UpdateParameters(dW, db, learningRate);

      if (i > 0)
      {
        var prevWeights = Layers[i].Weights;
        dA = layer.CalculateGradient(dA, prevWeights);
      }
    }
  }

  public void Train(float[,] X, int[] Y, float learningRate, int iterations, int batchSize = 64)
  {
    var epochTimer = Stopwatch.StartNew();
    var epochDelta = 0L;
    int m = Y.Length;

    for (int iter = 0; iter < iterations; iter++)
    {
      for (int start = 0; start < m; start += batchSize)
      {
        // Extract batch (simplified - in practice you'd want proper batching)
        int end = Math.Min(start + batchSize, m);
        int currentBatchSize = end - start;

        float[,] X_batch = ExtractBatch(X, start, currentBatchSize);
        int[] Y_batch = Y.Skip(start).Take(currentBatchSize).ToArray();

        // Forward pass
        float[,] output = Forward(X_batch);

        // Backward pass
        Backward(X_batch, Y_batch, learningRate, currentBatchSize);
      }

      // Every 10 epochs
      if (iter % 10 == 0)
      {
        float[,] output = Forward(X);
        int[] predictions = GetPredictions(output);
        float acc = GetAccuracy(predictions, Y);
        var estTime = (epochTimer.ElapsedMilliseconds - epochDelta) / 1000f * ((iterations - iter) / 10f) / 60f;
        var mins = float.Floor(estTime);
        var secs = (estTime - mins) * 60f;
        epochDelta = epochTimer.ElapsedMilliseconds;
        Console.WriteLine($"Iteration {iter}, Accuracy: {acc:P2}, Time {epochTimer.ElapsedMilliseconds / 1000}s, est Time to finish: {mins}m {secs.ToString("0")}s");
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
}
