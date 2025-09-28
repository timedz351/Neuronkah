namespace neuronka
{
    public static class ModelTester
    {
        /// <summary>
        /// Tests the neural network on a dataset and returns accuracy.
        /// </summary>
        public static float TestModel(
            float[,] W1, float[,] b1, float[,] W2, float[,] b2,
            float[,] X, int[] Y)
        {
            // Forward pass
            var (_, _, _, A2) = Program.ForwardProp(W1, b1, W2, b2, X);

            // Get predictions
            int[] predictions = Program.GetPredictions(A2);

            // Compute accuracy
            float accuracy = Program.GetAccuracy(predictions, Y);

            return accuracy;
        }

    }
}
