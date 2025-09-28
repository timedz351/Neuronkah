namespace neuronka;

public class MatrixUtils
{
public static float[,] Dot(float[,] A, float[,] B)
{
    int aRows = A.GetLength(0), aCols = A.GetLength(1);
    int bRows = B.GetLength(0), bCols = B.GetLength(1);
    if (aCols != bRows) throw new ArgumentException("Incompatible shapes for Dot.");

    var result = new float[aRows, bCols];
    for (int i = 0; i < aRows; i++)
        for (int j = 0; j < bCols; j++)
            for (int k = 0; k < aCols; k++)
                result[i, j] += A[i, k] * B[k, j];
    return result;
}

public static float[,] Add(float[,] Z, float[,] b)
{
    int rows = Z.GetLength(0), cols = Z.GetLength(1);
    var result = new float[rows, cols];
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result[i, j] = Z[i, j] + b[i, 0];
    return result;
}

public static float[,] Subtract(float[,] A, float[,] B)
{
    int rows = A.GetLength(0), cols = A.GetLength(1);
    var result = new float[rows, cols];
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result[i, j] = A[i, j] - B[i, j];
    return result;
}

public static float[,] Multiply(float[,] A, float[,] B)
{
    int rows = A.GetLength(0), cols = A.GetLength(1);
    var result = new float[rows, cols];
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result[i, j] = A[i, j] * B[i, j];
    return result;
}

public static float[,] Scale(float[,] A, float scalar)
{
    int rows = A.GetLength(0), cols = A.GetLength(1);
    var result = new float[rows, cols];
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result[i, j] = A[i, j] * scalar;
    return result;
}

public static float[,] Transpose(float[,] A)
{
    int rows = A.GetLength(0), cols = A.GetLength(1);
    var result = new float[cols, rows];
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result[j, i] = A[i, j];
    return result;
}

public static float[,] SumColumns(float[,] A)
{
    int rows = A.GetLength(0), cols = A.GetLength(1);
    var result = new float[rows, 1];
    for (int i = 0; i < rows; i++)
    {
        float sum = 0;
        for (int j = 0; j < cols; j++)
            sum += A[i, j];
        result[i, 0] = sum;
    }
    return result;
}
}