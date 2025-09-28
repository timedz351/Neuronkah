using System.Net.Mime;
using System.Reflection.Emit;
using neuronka.dataLoading;

class Program
{
    static void Main()
    {
        string projectRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", ".."));

        var (trainData, testData) = LoadData(projectRoot);
        var (trainImages, trainLabels) = trainData;
        var (testImages, testLabels) = testData;

        Console.WriteLine($"Loaded {trainImages.Count} images and labels for training");
        Console.WriteLine($"Loaded {testImages.Count} images and labels for testing");
    }

    static T[,] Transpose<T>(T[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        var result = new T[cols, rows];

        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result[j, i] = matrix[i, j];

        return result;
    }
    
    static ((List<int[]> Images, List<int> Labels) Train, (List<int[]> Images, List<int> Labels) Test) LoadData(string projectRoot)
    {
        var loader = new DataLoader(projectRoot);
        var data = loader.LoadData();
        return data;
    }
}