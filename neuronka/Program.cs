using System.Diagnostics;
using System.Net.Mime;
using System.Reflection.Emit;
using neuronka;
using neuronka.dataLoading;

class Program
{
    private static int _batchSize = 32;
    private static int _epochs = 10;
    private static float _alpha = 0.1f;
    
    private static int _hidden_layer_size = 128;
    private static float _momentum = 0.9f;   // 0.8–0.9 is typical

    static void Main()
    {
        // LOADING
        string projectRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", ".."));
        var loader = new DataLoader(projectRoot);
        var (train, test) = loader.LoadData();
        var (trainX, trainY) = train;   // shape (784, 60 000) , length 60 000
        var (testX,  testY)  = test;

        Console.WriteLine($"Loaded {trainY.Length} train, {testY.Length} test");

        // ---------- train with mini-batches ----------
        var sw = Stopwatch.StartNew();
        var (W1, b1, W2, b2) = GradientDescent(trainX, trainY, _alpha, _epochs, _batchSize);
        sw.Stop();
        Console.WriteLine($"Training done in {sw.Elapsed.TotalMinutes:F1} min");

        // ---------- evaluate ----------
        float acc = ModelTester.TestModel(W1, b1, W2, b2, testX, testY);
        Console.WriteLine($"Test accuracy {acc:P2}");

    }   
    static void Shuffle(float[,] x, int[] y)
    {
        int m = y.Length;
        var rng = new Random();
        for (int i = m - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            // swap labels
            (y[i], y[j]) = (y[j], y[i]);
            // swap columns in x
            for (int r = 0; r < x.GetLength(0); r++)
                (x[r, i], x[r, j]) = (x[r, j], x[r, i]);
        }
    }
    
    public static (float[,] W1, float[,] b1, float[,] W2, float[,] b2)
        GradientDescent(float[,] X_full, int[] Y_full,
            float alpha, int epochs, int batchSize = 64)
    {
        var (W1, b1, W2, b2) = InitParams();
        int m = Y_full.Length;
        int batchesPerEpoch = (int)Math.Ceiling((double)m / batchSize);

        // velocity buffers (same shape as parameters)
        float[,] vW1 = new float[W1.GetLength(0), W1.GetLength(1)];
        float[,] vb1 = new float[b1.GetLength(0), b1.GetLength(1)];
        float[,] vW2 = new float[W2.GetLength(0), W2.GetLength(1)];
        float[,] vb2 = new float[b2.GetLength(0), b2.GetLength(1)];

            
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // optional: shuffle before every epoch
            Shuffle(X_full, Y_full);

            for (int b = 0; b < batchesPerEpoch; b++)
            {
                int start = b * batchSize;
                var (X, Y) = GetMiniBatch(X_full, Y_full, start, batchSize);

                // ----- classic forward / backward / update -----
                var (Z1, A1, Z2, A2) = ForwardProp(W1, b1, W2, b2, X);
                var (dW1, db1, dW2, db2) = BackwardProp(Z1, A1, Z2, A2, W2, X, Y);
                // update parameters with momentum
                UpdateParamsWithMomentum(
                    ref W1, ref b1, ref W2, ref b2,
                    dW1, db1, dW2, db2,
                    ref vW1, ref vb1, ref vW2, ref vb2,
                    alpha, _momentum);
            }

            // ---- progress print ----
            if (epoch % 10 == 0)
            {
                var (_, _, _, A2) = ForwardProp(W1, b1, W2, b2, X_full); // full set for accuracy
                float acc = GetAccuracy(GetPredictions(A2), Y_full);
                Console.WriteLine($"Epoch {epoch,-3}  acc {acc:P2}");
            }
        }
        return (W1, b1, W2, b2);
    }
    
    static void UpdateParamsWithMomentum(
        ref float[,] W1, ref float[,] b1, ref float[,] W2, ref float[,] b2,
        float[,] dW1, float[,] db1, float[,] dW2, float[,] db2,
        ref float[,] vW1, ref float[,] vb1, ref float[,] vW2, ref float[,] vb2,
        float alpha, float beta)
    {
        int h  = W1.GetLength(0);
        int d  = W1.GetLength(1);
        int k  = W2.GetLength(0);
    
        // W1, b1
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < d; j++)
            {
                vW1[i, j] = beta * vW1[i, j] + alpha * dW1[i, j];
                W1[i, j] -= vW1[i, j];
            }
            vb1[i, 0] = beta * vb1[i, 0] + alpha * db1[i, 0];
            b1[i, 0] -= vb1[i, 0];
        }
    
        // W2, b2
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < h; j++)
            {
                vW2[i, j] = beta * vW2[i, j] + alpha * dW2[i, j];
                W2[i, j] -= vW2[i, j];
            }
            vb2[i, 0] = beta * vb2[i, 0] + alpha * db2[i, 0];
            b2[i, 0] -= vb2[i, 0];
        }
    }
    
    private static (float[,] W1, float[,] b1, float[,] W2, float[,] b2)
        InitParams(float factor = 0.01f)
    {
        Random rand = new Random();
        int h = _hidden_layer_size;
        float[,] W1 = new float[h, 784];
        float[,] b1 = new float[h, 1];
        float[,] W2 = new float[10, h];
        float[,] b2 = new float[10, 1];

        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < 784; j++)
                W1[i, j] = (float)(rand.NextDouble() * factor);
            b1[i, 0] = 0f;
        }
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < h; j++)
                W2[i, j] = (float)(rand.NextDouble() * factor);
            b2[i, 0] = 0f;
        }
        return (W1, b1, W2, b2);
    }
    
    public static (float[,] Z1, float[,] A1, float[,] Z2, float[,] A2) ForwardProp(
        float[,] W1, float[,] b1, float[,] W2, float[,] b2, float[,] X)
    {
        /*
        float[,] Z1 = MatrixUtils.Add(MatrixUtils.Dot(W1, X), b1);
        float[,] A1 = ActivationFunctions.ReLU(Z1);
        
        float[,] Z2 = MatrixUtils.Add(MatrixUtils.Dot(W2, A1), b2);
        float[,] A2 = ActivationFunctions.Softmax(Z2);
        */
        var (Z1, A1) = OneStepForwardProp(W1, b1, ActivationFunctions.ReLU, X);
        var (Z2, A2) = OneStepForwardProp(W2, b2, ActivationFunctions.Softmax, A1);
        return (Z1,A1, Z2, A2);
    }

    public static (float[,] Z, float[,] A) OneStepForwardProp(
        float[,] W1, float[,] b1, Func<float[,], float[,]> activationFun, float[,] X)
    {
        float[,] Z = MatrixUtils.Add(MatrixUtils.Dot(W1, X), b1);
        float[,] A = activationFun(Z);
        return (Z, A);
    }
    
    public static (float[,] dW1, float[,] db1, float[,] dW2, float[,] db2) BackwardProp(
        float[,] Z1, float[,] A1, float[,] Z2, float[,] A2,
        float[,] W2, float[,] X, int[] Y)
    {
        int m = Y.Length; // batch size
        float[,] Y_onehot = OneHot(Y, 10);

        float[,] dZ2 = MatrixUtils.Subtract(A2, Y_onehot);
        float[,] dW2 = MatrixUtils.Scale(MatrixUtils.Dot(dZ2, MatrixUtils.Transpose(A1)), 1f / m);
        float[,] db2 = MatrixUtils.Scale(MatrixUtils.SumColumns(dZ2), 1f / m);

        float[,] dZ1 = MatrixUtils.Multiply(MatrixUtils.Dot(MatrixUtils.Transpose(W2), dZ2), ActivationFunctions.ReLU_Deriv(Z1));
        float[,] dW1 = MatrixUtils.Scale(MatrixUtils.Dot(dZ1, MatrixUtils.Transpose(X)), 1f / m);
        float[,] db1 = MatrixUtils.Scale(MatrixUtils.SumColumns(dZ1), 1f / m);

        return (dW1, db1, dW2, db2);
    }

    public static (float[,] W1, float[,] b1, float[,] W2, float[,] b2) UpdateParams(
        float[,] W1, float[,] b1, float[,] W2, float[,] b2,
        float[,] dW1, float[,] db1, float[,] dW2, float[,] db2, float alpha)
    {
        b1 = MatrixUtils.Subtract(b1, MatrixUtils.Scale(db1, alpha));
        W1 = MatrixUtils.Subtract(W1, MatrixUtils.Scale(dW1, alpha));
        W2 = MatrixUtils.Subtract(W2, MatrixUtils.Scale(dW2, alpha));
        b2 = MatrixUtils.Subtract(b2, MatrixUtils.Scale(db2, alpha));
        return (W1, b1, W2, b2);
    }

    private static float[,] OneHot(int[] Y, int numClasses)
    {
        int m = Y.Length;
        var oneHot = new float[numClasses, m];
        for (int i = 0; i < m; i++)
            oneHot[Y[i], i] = 1f;
        return oneHot;
    }
    public static int[] GetPredictions(float[,] A2)
    {
        int rows = A2.GetLength(0);
        int cols = A2.GetLength(1);
        int[] predictions = new int[cols];

        for (int j = 0; j < cols; j++)
        {
            float max = float.MinValue;
            int maxIndex = 0;
            for (int i = 0; i < rows; i++)
            {
                if (A2[i, j] > max)
                {
                    max = A2[i, j];
                    maxIndex = i;
                }
            }
            predictions[j] = maxIndex;
        }
        return predictions;
    }
    
    public static float GetAccuracy(int[] predictions, int[] Y)
    {
        int correct = 0;
        for (int i = 0; i < Y.Length; i++)
            if (predictions[i] == Y[i])
                correct++;

        return (float)correct / Y.Length;
    }
    
    static ((float[,] Images, int[] Labels) Train, (float[,] Images, int[] Labels) Test) LoadData(string projectRoot)
    {
        var loader = new DataLoader(projectRoot);
        var data= loader.LoadData();
        return data;
    }
    
    // Returns a mini-batch sliced from the full data set.
    // X shape: (784, m)   Y shape: (m,)
    static (float[,] X, int[] Y) GetMiniBatch(float[,] fullX, int[] fullY, int startIdx, int batchSize)
    {
        int feat = fullX.GetLength(0);
        int last  = Math.Min(startIdx + batchSize, fullY.Length);
        int real  = last - startIdx;               // last batch can be smaller

        float[,] x = new float[feat, real];
        int[]    y = new int[real];

        for (int j = 0; j < real; j++)
        {
            for (int i = 0; i < feat; i++)
                x[i, j] = fullX[i, startIdx + j];
            y[j] = fullY[startIdx + j];
        }
        return (x, y);
    }
}
