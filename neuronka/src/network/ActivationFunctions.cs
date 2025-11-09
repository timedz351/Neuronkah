namespace neuronka;

public class ActivationFunctions
{
    public static float[,] ReLU(float[,] Z)
    {
        int rows = Z.GetLength(0), cols = Z.GetLength(1);
        var A = new float[rows, cols];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            A[i, j] = Math.Max(0, Z[i, j]);
        return A;
    }

    public static float[,] ReLU_Deriv(float[,] Z)
    {
        int rows = Z.GetLength(0), cols = Z.GetLength(1);
        var dZ = new float[rows, cols];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            dZ[i, j] = Z[i, j] > 0 ? 1f : 0f;
        return dZ;
    }

    public static float[,] Softmax(float[,] Z)
    {
        int rows = Z.GetLength(0), cols = Z.GetLength(1);
        var A = new float[rows, cols];
        for (int j = 0; j < cols; j++)
        {
            float max = float.MinValue;
            for (int i = 0; i < rows; i++) max = Math.Max(max, Z[i, j]);
            float sumExp = 0f;
            for (int i = 0; i < rows; i++) 
                sumExp += MathF.Exp(Z[i, j] - max);
            for (int i = 0; i < rows; i++)
                A[i, j] = MathF.Exp(Z[i, j] - max) / sumExp;
        }
        return A;
    }
    
    public static float[,] Sigmoid(float[,] Z)
    {
        int rows = Z.GetLength(0), cols = Z.GetLength(1);
        var A = new float[rows, cols];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            A[i, j] = 1f / (1f + MathF.Exp(-Z[i, j]));
        return A;
    }
    
    public static float[,] Sigmoid_Deriv(float[,] Z)
    {
        var sigmoid = Sigmoid(Z);
        int rows = Z.GetLength(0), cols = Z.GetLength(1);
        var deriv = new float[rows, cols];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            deriv[i, j] = sigmoid[i, j] * (1 - sigmoid[i, j]);
        return deriv;
    }
    
    public static float[,] Tanh(float[,] Z)
    {
        int rows = Z.GetLength(0), cols = Z.GetLength(1);
        var A = new float[rows, cols];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            A[i, j] = MathF.Tanh(Z[i, j]);
        return A;
    }
    
    public static float[,] Tanh_Deriv(float[,] Z)
    {
        var tanh = Tanh(Z);
        int rows = Z.GetLength(0), cols = Z.GetLength(1);
        var deriv = new float[rows, cols];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            deriv[i, j] = 1 - tanh[i, j] * tanh[i, j];
        return deriv;
    }
}