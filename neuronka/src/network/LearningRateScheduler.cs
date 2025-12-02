
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

  // Returns LR for the CURRENT epoch (starting at 0), then advances internal epoch counter.
  public float GetLearningRate()
  {
    float lr = _type switch
    {
      ScheduleType.Constant => _initialRate,
      ScheduleType.StepDecay => StepDecay(),
      ScheduleType.Exponential => ExponentialDecay(),
      _ => _initialRate
    };
    _currentEpoch++; // advance AFTER computing lr so first call corresponds to epoch 0
    return lr;
  }

  private float StepDecay()
  {
    // Epoch 0: floor(0/stepSize)=0 -> initialRate
    return _initialRate * MathF.Pow(_decayRate, MathF.Floor((float)_currentEpoch / _stepSize));
  }

  private float ExponentialDecay()
  {
    // Standard exponential decay: lr = lr0 * exp(-decayRate * epoch)
    return _initialRate * MathF.Exp(-_decayRate * _currentEpoch);
  }

  public void Reset()
  {
    _currentEpoch = 0;
  }
}
