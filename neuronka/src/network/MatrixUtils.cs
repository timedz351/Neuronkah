namespace neuronka;

using System.Threading.Tasks;

public class MatrixUtils
{
    public static float[,] Dot(float[,] A, float[,] B)
    {
        int aRows = A.GetLength(0), aCols = A.GetLength(1);
        int bRows = B.GetLength(0), bCols = B.GetLength(1);
        // if (aCols != bRows) throw new ArgumentException("Incompatible shapes for Dot.");

        var result = new float[aRows, bCols];

        Parallel.For(0, aRows, i =>
        {
            for (int j = 0; j < bCols; j++)
            {
                float sum = 0;
                for (int k = 0; k < aCols; k++)
                    sum += A[i, k] * B[k, j];
                result[i, j] = sum;
            }
        });
        return result;
    }

    public static float[,] Add(float[,] Z, float[,] b)
    {
        int rows = Z.GetLength(0), cols = Z.GetLength(1);
        var result = new float[rows, cols];

        Parallel.For(0, rows, i =>
        {
            for (int j = 0; j < cols; j++)
                result[i, j] = Z[i, j] + b[i, 0];
        });
        return result;
    }

    public static float[,] Subtract(float[,] A, float[,] B)
    {
        int rows = A.GetLength(0), cols = A.GetLength(1);
        var result = new float[rows, cols];

        Parallel.For(0, rows, i =>
        {
            for (int j = 0; j < cols; j++)
                result[i, j] = A[i, j] - B[i, j];
        });
        return result;
    }

    public static float[,] Multiply(float[,] A, float[,] B)
    {
        int rows = A.GetLength(0), cols = A.GetLength(1);
        var result = new float[rows, cols];

        Parallel.For(0, rows, i =>
        {
            for (int j = 0; j < cols; j++)
                result[i, j] = A[i, j] * B[i, j];
        });
        return result;
    }

    public static float[,] Scale(float[,] A, float scalar)
    {
        int rows = A.GetLength(0), cols = A.GetLength(1);
        var result = new float[rows, cols];

        Parallel.For(0, rows, i =>
        {
            for (int j = 0; j < cols; j++)
                result[i, j] = A[i, j] * scalar;
        });
        return result;
    }

    public static float[,] Transpose(float[,] A)
    {
        int rows = A.GetLength(0), cols = A.GetLength(1);
        var result = new float[cols, rows];

        Parallel.For(0, rows, i =>
        {
            for (int j = 0; j < cols; j++)
                result[j, i] = A[i, j];
        });
        return result;
    }

    public static float[,] SumColumns(float[,] A)
    {
        int rows = A.GetLength(0), cols = A.GetLength(1);
        var result = new float[rows, 1];

        Parallel.For(0, rows, i =>
        {
            float sum = 0;
            for (int j = 0; j < cols; j++)
                sum += A[i, j];
            result[i, 0] = sum;
        });
        return result;
    }

}
