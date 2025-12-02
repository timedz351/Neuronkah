namespace neuronka.exporter;

public class Exporter
{
    private static void Export(string projectRoot, int[] predictions, string fileName)
    {
        File.WriteAllLines(
            Path.Combine(projectRoot, fileName),
            predictions.Select(p => p.ToString())
        );
    }

    public static void ExportTrain(string projectRoot,  int[] predictions)
    {
        Export(projectRoot, predictions, "train_predictions.csv");
    }
    
    
    public static void ExportTest(string projectRoot, int[] predictions)
    {
        Export(projectRoot, predictions, "test_predictions.csv");
    }
}