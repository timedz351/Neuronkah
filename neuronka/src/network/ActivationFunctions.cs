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
}