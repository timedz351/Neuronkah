namespace neuronka;

public class Layer
{
  public string Name { get; }
  public int InputSize { get; }
  public int OutputSize { get; }
  public string Activation { get; }

  public float[,] Weights { get; set; }
  public float[,] Biases { get; set; }

  // Cache for backpropagation
  public float[,] Z { get; set; }
  public float[,] A { get; set; }
  public float[,] Input { get; set; }

  public Layer(Random rand, string name, int inputSize, int outputSize, string activation = "relu")
  {
    Name = name;
    InputSize = inputSize;
    OutputSize = outputSize;
    Activation = activation;

    InitializeParameters(rand);
  }

  private void InitializeParameters(Random rand)
  {
    Weights = new float[OutputSize, InputSize];
    Biases = new float[OutputSize, 1];

    // He initialization for ReLU, Xavier for tanh/sigmoid
    float scale = Activation switch
    {
      "relu" => (float)Math.Sqrt(2.0 / InputSize),
      "tanh" or "sigmoid" => (float)Math.Sqrt(1.0 / InputSize),
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
    Z = MatrixUtils.Add(MatrixUtils.Dot(Weights, input), Biases);

    A = Activation switch
    {
      "relu" => ActivationFunctions.ReLU(Z),
      "sigmoid" => ActivationFunctions.Sigmoid(Z),
      "tanh" => ActivationFunctions.Tanh(Z),
      "softmax" => ActivationFunctions.Softmax(Z),
      "linear" => Z,
      _ => throw new ArgumentException($"Unknown activation function: {Activation}")
    };

    return A;
  }

  public (float[,] dW, float[,] db) Backward(float[,] dA, float[,] prevActivation, int batchSize)
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
        _ => throw new ArgumentException($"Unknown activation function: {Activation}")
      };

      dZ = MatrixUtils.Multiply(dA, activationDeriv);
    }

    float[,] dW = MatrixUtils.Scale(MatrixUtils.Dot(dZ, MatrixUtils.Transpose(prevActivation)), 1f / batchSize);
    float[,] db = MatrixUtils.Scale(MatrixUtils.SumColumns(dZ), 1f / batchSize);

    return (dW, db);
  }

  public float[,] CalculateGradient(float[,] dA, float[,] currentWeights)
  {
    float[,] dZ;

    if (Activation == "softmax")
    {
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
        _ => throw new ArgumentException($"Unknown activation function: {Activation}")
      };

      dZ = MatrixUtils.Multiply(dA, activationDeriv);
    }

    // CORRECTED: Propagate gradient backward using current layer's weights
    return MatrixUtils.Dot(MatrixUtils.Transpose(currentWeights), dZ);
  }

  public void UpdateParameters(float[,] dW, float[,] db, float learningRate)
  {
    Weights = MatrixUtils.Subtract(Weights, MatrixUtils.Scale(dW, learningRate));
    Biases = MatrixUtils.Subtract(Biases, MatrixUtils.Scale(db, learningRate));
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
