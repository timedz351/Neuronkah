using System.Net.Mime;
using System.Reflection.Emit;
using neuronka.dataLoading;

class Program
{
    static void Main()
    {
        string projectRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", ".."));

        var (trainData, testData) = LoadData(projectRoot);
        Console.WriteLine($"Loaded {trainData.Count()} images and labels for training");
        Console.WriteLine($"Loaded {testData.Count()} images and labels for testing");
    }

    static (List<(int[] Image, int Label)> Train, List<(int[] Image, int Label)> Test) LoadData(string projectRoot)
    {
        var loader = new DataLoader(projectRoot);
        var data = loader.LoadData();
        return data;
    }
}