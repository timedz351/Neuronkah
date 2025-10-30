using System.Diagnostics;
using System.Net.Mime;
using System.Reflection.Emit;
using neuronka;
using neuronka.dataLoading;

class Program
{
    private static int _batchSize = 1000;
    private static int _iterations = 500;
    private static float _alpha = 0.01f;
    static void Main()
    {
        // LOADING
        var fullTimer = Stopwatch.StartNew();
        string projectRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", ".."));
        var loaderTimer = Stopwatch.StartNew();
        var (trainData, testData) = LoadData(projectRoot);
        var (trainImages, trainLabels) = trainData;
        var (testImages, testLabels) = testData;
        Console.WriteLine($"Loaded {trainImages.GetLength(1)} images and labels for training");
        Console.WriteLine($"Loaded {testImages.GetLength(1)} images and labels for testing");
        loaderTimer.Stop();
        Console.WriteLine($"Data loaded in {loaderTimer.ElapsedMilliseconds} ms");

        // CREATE SMALLER BATCH
        int batchSize = _batchSize;
        float[,] X_batch = new float[trainImages.GetLength(0), batchSize];
        int[] Y_batch = new int[batchSize];
        for (int j = 0; j < batchSize; j++)
        {
            for (int i = 0; i < trainImages.GetLength(0); i++)
                X_batch[i, j] = trainImages[i, j]; // copy column
            Y_batch[j] = trainLabels[j];
        }
        Console.WriteLine($"X_Batch size: {X_batch.GetLength(0)}x{X_batch.GetLength(1)}");
        Console.WriteLine($"Y_Batch size: {Y_batch.GetLength(0)}");
        
        // TRAIN MODEL
        var trainingTimer = Stopwatch.StartNew();
        var (W1, b1, W2, b2) = GradientDescent(X_batch, Y_batch, _alpha, _iterations, batchSize);
        trainingTimer.Stop();
        Console.WriteLine($"Training completed in {trainingTimer.ElapsedMilliseconds} ms");

        // TEST MODEL
        var accuracy = ModelTester.TestModel(W1, b1, W2, b2, testImages, testLabels);
        Console.WriteLine($"Test accuracy: {accuracy:P2}\n");
        Console.WriteLine($"Model ran in {fullTimer.ElapsedMilliseconds/ 60000} min");

    }   
    
    public static (float[,] W1, float[,] b1, float[,] W2, float[,] b2) GradientDescent(
        float[,] X_batch, int[] Y_batch, float alpha, int iterations, int batchSize = 64)
    {
        var epochTimer = Stopwatch.StartNew();
        var epochDelta = 0l;
        
        var (W1, b1, W2, b2) = InitParams();
        int m = Y_batch.Length;
        
        for (int iter = 0; iter < iterations; iter++)
        {
            for (int start = 0; start < m; start += batchSize)
            {
                // Forward pass
                var (Z1, A1, Z2, A2) = ForwardProp(W1, b1, W2, b2, X_batch);

                // Backward pass
                var (dW1, db1, dW2, db2) = BackwardProp(Z1, A1, Z2, A2, W2, X_batch, Y_batch);

                // Update parameters
                (W1, b1, W2, b2) = UpdateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha);
            }
            // Every 10 epochs
            if (iter % 10 == 0)
            {
                var (_, _, _, A2_full) = ForwardProp(W1, b1, W2, b2, X_batch);
                int[] predictions = GetPredictions(A2_full);
                float acc = GetAccuracy(predictions, Y_batch);
                var estTime = (epochTimer.ElapsedMilliseconds - epochDelta) / 1000f * ((iterations - iter) / 10f) / 60f;
                var mins = float.Floor(estTime);
                var secs = (estTime - mins) * 60f;
                epochDelta = epochTimer.ElapsedMilliseconds;
                Console.WriteLine($"Iteration {iter}, Accuracy: {acc:P2}, Time {epochTimer.ElapsedMilliseconds/1000}s, est Time to finish: {mins}m {secs.ToString("0")}s");
            }
        }

        return (W1, b1, W2, b2);
    }
    
    private static (float[,] W1, float[,] b1, float[,] W2, float[,] b2) InitParams(float factor = 0.01f)
    {
        var rand = new Random();
        float[,] W1 = new float[10, 784];
        float[,] b1 = new float[10, 1];
        float[,] W2 = new float[10, 10];
        float[,] b2 = new float[10, 1];

        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 784; j++)
                W1[i, j] = (float)(rand.NextDouble() * factor);

            for (int j = 0; j < 10; j++)
                W2[i, j] = (float)(rand.NextDouble() * factor);
        }

        return (W1, b1, W2, b2);
    }
    
    public static (float[,] Z1, float[,] A1, float[,] Z2, float[,] A2) ForwardProp(
        float[,] W1, float[,] b1, float[,] W2, float[,] b2, float[,] X)
    {
        float[,] Z1 = MatrixUtils.Add(MatrixUtils.Dot(W1, X), b1);
        float[,] A1 = ActivationFunctions.ReLU(Z1);
        float[,] Z2 = MatrixUtils.Add(MatrixUtils.Dot(W2, A1), b2);
        float[,] A2 = ActivationFunctions.Softmax(Z2);
        return (Z1, A1, Z2, A2);
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
}
