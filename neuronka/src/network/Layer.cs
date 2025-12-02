namespace neuronka;

public enum MomentumType
{
  None,
  Smoothed,   // v = beta*v + (1-beta)*grad; W -= lr * v
  Classical   // v = beta*v + grad;          W -= lr * v
}

public class Layer
{
  public string Name { get; }
  public int InputSize { get; }
  public int OutputSize { get; }
  public string Activation { get; }

  public float[,] Weights { get; set; } = default!;
  public float[,] Biases { get; set; } = default!;
  // Momentum velocities (optional)
  public float[,] Vw { get; set; } = default!;
  public float[,] Vb { get; set; } = default!;

  // Cache for backpropagation
  public float[,] Z { get; set; } = default!;
  public float[,] A { get; set; } = default!;
  public float[,] Input { get; set; } = default!;

  public Layer(System.Random rand, string name, int inputSize, int outputSize, string activation = "relu")
  {
    Name = name;
    InputSize = inputSize;
    OutputSize = outputSize;
    Activation = activation;

    InitializeParameters(rand);
  }

  private void InitializeParameters(System.Random rand)
  {
    Weights = new float[OutputSize, InputSize];
    Biases = new float[OutputSize, 1];
    // Initialize velocities to zero
    Vw = new float[OutputSize, InputSize];
    Vb = new float[OutputSize, 1];

    // He initialization for ReLU, Xavier for tanh/sigmoid
    float scale = Activation switch
    {
      "relu" => (float)System.Math.Sqrt(2.0 / InputSize),
      "tanh" or "sigmoid" => (float)System.Math.Sqrt(1.0 / InputSize),
      _ => 0.01f
    };

    for (int i = 0; i < OutputSize; i++)
    {
      for (int j = 0; j < InputSize; j++)
        Weights[i, j] = (float)((rand.NextDouble() - 0.5) * 2 * scale);
    }
  }

  public float[,] Forward(float[,] input)
  {
    Input = input;
    // Z = W * input + b
    Z = MatrixUtils.Add(MatrixUtils.Dot(Weights, input), Biases);

    A = Activation switch
    {
      "relu" => ActivationFunctions.ReLU(Z),
      "sigmoid" => ActivationFunctions.Sigmoid(Z),
      "tanh" => ActivationFunctions.Tanh(Z),
      "softmax" => ActivationFunctions.Softmax(Z),
      "linear" => Z,
      _ => throw new System.ArgumentException($"Unknown activation function: {Activation}")
    };

    return A;
  }

  public (float[,] dW, float[,] db, float[,] dA_prev) Backward(float[,] dA, float[,] prevActivation, int batchSize)
  {
    float[,] dZ;

    if (Activation == "softmax")
    {
      // For softmax, dZ is already computed as (A - Y) in the output layer
      dZ = dA;
    }
    else
    {
      float[,] activationDeriv = Activation switch
      {
        "relu" => ActivationFunctions.ReLU_Deriv(Z),
        "sigmoid" => ActivationFunctions.Sigmoid_Deriv(Z),
        "tanh" => ActivationFunctions.Tanh_Deriv(Z),
        "linear" => CreateOnesMatrix(Z.GetLength(0), Z.GetLength(1)),
        _ => throw new System.ArgumentException($"Unknown activation function: {Activation}")
      };

      dZ = MatrixUtils.Multiply(dA, activationDeriv);
    }

    float[,] dW = MatrixUtils.Scale(MatrixUtils.Dot(dZ, MatrixUtils.Transpose(prevActivation)), 1f / batchSize);
    // L2 regularization: add lambda * W to gradient if enabled
    if (TrainingSettings.WeightDecay > 0f)
    {
      for (int i = 0; i < dW.GetLength(0); i++)
      {
        for (int j = 0; j < dW.GetLength(1); j++)
        {
          dW[i, j] += TrainingSettings.WeightDecay * Weights[i, j];
        }
      }
    }
    float[,] db = MatrixUtils.Scale(MatrixUtils.SumColumns(dZ), 1f / batchSize);
    // gradient wrt previous layer activation: W^T * dZ (before updating parameters)
    float[,] dA_prev = MatrixUtils.Dot(MatrixUtils.Transpose(Weights), dZ);

    return (dW, db, dA_prev);
  }

  public void UpdateParameters(float[,] dW, float[,] db, float learningRate)
  {
    Weights = MatrixUtils.Subtract(Weights, MatrixUtils.Scale(dW, learningRate));
    Biases = MatrixUtils.Subtract(Biases, MatrixUtils.Scale(db, learningRate));
  }

  // Overload with momentum
  public void UpdateParameters(float[,] dW, float[,] db, float learningRate, float momentumBeta)
  {
    if (momentumBeta <= 0f)
    {
      UpdateParameters(dW, db, learningRate);
      return;
    }

    // Ensure velocity matrices match dimensions
    if (Vw == null || Vw.GetLength(0) != Weights.GetLength(0) || Vw.GetLength(1) != Weights.GetLength(1))
      Vw = new float[Weights.GetLength(0), Weights.GetLength(1)];
    if (Vb == null || Vb.GetLength(0) != Biases.GetLength(0) || Vb.GetLength(1) != Biases.GetLength(1))
      Vb = new float[Biases.GetLength(0), Biases.GetLength(1)];

    // Default to classical momentum for backward compatibility
    for (int i = 0; i < Vw.GetLength(0); i++)
    {
      for (int j = 0; j < Vw.GetLength(1); j++)
      {
        Vw[i, j] = momentumBeta * Vw[i, j] + dW[i, j];
        Weights[i, j] -= learningRate * Vw[i, j];
      }
    }
    for (int i = 0; i < Vb.GetLength(0); i++)
    {
      for (int j = 0; j < Vb.GetLength(1); j++)
      {
        Vb[i, j] = momentumBeta * Vb[i, j] + db[i, j];
        Biases[i, j] -= learningRate * Vb[i, j];
      }
    }
  }

  // Overload with explicit momentum type
  public void UpdateParameters(float[,] dW, float[,] db, float learningRate, float momentumBeta, MomentumType momentumType)
  {
    if (momentumBeta <= 0f || momentumType == MomentumType.None)
    {
      UpdateParameters(dW, db, learningRate);
      return;
    }

    if (momentumType == MomentumType.Smoothed)
    {
      // v = beta*v + (1-beta)*grad; W -= lr * v
      float blend = 1f - momentumBeta;
      for (int i = 0; i < Vw.GetLength(0); i++)
      {
        for (int j = 0; j < Vw.GetLength(1); j++)
        {
          Vw[i, j] = momentumBeta * Vw[i, j] + blend * dW[i, j];
          Weights[i, j] -= learningRate * Vw[i, j];
        }
      }
      for (int i = 0; i < Vb.GetLength(0); i++)
      {
        for (int j = 0; j < Vb.GetLength(1); j++)
        {
          Vb[i, j] = momentumBeta * Vb[i, j] + blend * db[i, j];
          Biases[i, j] -= learningRate * Vb[i, j];
        }
      }
    }
    else // MomentumType.Classical
    {
      // v = beta*v + grad; W -= lr * v
      for (int i = 0; i < Vw.GetLength(0); i++)
      {
        for (int j = 0; j < Vw.GetLength(1); j++)
        {
          Vw[i, j] = momentumBeta * Vw[i, j] + dW[i, j];
          Weights[i, j] -= learningRate * Vw[i, j];
        }
      }
      for (int i = 0; i < Vb.GetLength(0); i++)
      {
        for (int j = 0; j < Vb.GetLength(1); j++)
        {
          Vb[i, j] = momentumBeta * Vb[i, j] + db[i, j];
          Biases[i, j] -= learningRate * Vb[i, j];
        }
      }
    }
  }

  private float[,] CreateOnesMatrix(int rows, int cols)
  {
    var ones = new float[rows, cols];
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
        ones[i, j] = 1f;
    return ones;
  }
}
