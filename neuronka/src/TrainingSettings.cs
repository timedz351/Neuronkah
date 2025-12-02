namespace neuronka;

public static class TrainingSettings
{
  // Core schedule / loop
  public static float LearningRate { get; set; } = 0.01f;
  public static float DecayRate { get; set; } = 0.85f;
  public static int StepSize { get; set; } = 2;
  public static int Epochs { get; set; } = 10;
  public static int BatchSize { get; set; } = 32;
  public static LearningRateScheduler.ScheduleType ScheduleType { get; set; } = LearningRateScheduler.ScheduleType.StepDecay;

  // Momentum
  public static float MomentumBeta { get; set; } = 0.95f;
  public static MomentumType MomentumType { get; set; } = MomentumType.Classical;

  // Other
  public static int LogEvery { get; set; } = 1;
  // L2 regularization (weight decay). Set to 0f to disable.
  public static float WeightDecay { get; set; } = 0f;
}
