package optimize;

/**
 * A function that can be applied to two double vectors via {@link DoubleVector}
 * #apply({@link DoubleVector} v, {@link DoubleDoubleVectorFunction} f);
 * 
 */
public interface DoubleDoubleVectorFunction {

  /**
   * Calculates the result of the left and right value of two vectors at a given
   * index.
   */
  public double calculate(int index, double left, double right);

}
