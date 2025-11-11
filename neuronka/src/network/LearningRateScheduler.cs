
namespace neuronka;

public class LearningRateScheduler
{
  public enum ScheduleType
  {
    Constant,
    StepDecay,
    Exponential,
    Cosine
  }

  private ScheduleType _type;
  private float _initialRate;
  private float _decayRate;
  private int _stepSize;
  private int _currentEpoch;

  public LearningRateScheduler(ScheduleType type, float initialRate, float decayRate = 0.1f, int stepSize = 10)
  {
    _type = type;
    _initialRate = initialRate;
    _decayRate = decayRate;
    _stepSize = stepSize;
    _currentEpoch = 0;
  }

  public float GetLearningRate()
  {
    _currentEpoch++;
    return _type switch
    {
      ScheduleType.Constant => _initialRate,
      ScheduleType.StepDecay => StepDecay(),
      ScheduleType.Exponential => ExponentialDecay(),
      ScheduleType.Cosine => CosineDecay(),
      _ => _initialRate
    };
  }

  private float StepDecay()
  {
    return _initialRate * MathF.Pow(_decayRate, MathF.Floor((float)_currentEpoch / _stepSize));
  }

  private float ExponentialDecay()
  {
    return _initialRate * MathF.Exp(-_decayRate * _currentEpoch);
  }

  private float CosineDecay()
  {
    // Simple cosine decay over total epochs (assuming 1000 total for calculation)
    float totalEpochs = 1000;
    return _initialRate * 0.5f * (1 + MathF.Cos(MathF.PI * _currentEpoch / totalEpochs));
  }

  public void Reset()
  {
    _currentEpoch = 0;
  }
}
