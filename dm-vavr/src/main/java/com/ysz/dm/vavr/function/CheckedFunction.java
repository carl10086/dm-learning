package com.ysz.dm.vavr.function;

import io.vavr.CheckedFunction1;
import io.vavr.Function0;
import io.vavr.collection.Array;
import java.util.Arrays;
import org.junit.Test;

public class CheckedFunction {

  public int biz(int i) throws IllegalArgumentException {
    if (i == 0) {
      throw new IllegalArgumentException();
    } else {
      return i + 1;
    }
  }

  @Test
  public void tstChkedFunction() {
    /*lift 包装为 Optional*/
    /*liftTry 包装为 Try*/
    Arrays.asList(0, 1, 2, 3).stream().map(CheckedFunction1.lift(
        i -> biz(i)
    )).forEach(
        System.out::println
    );
  }


  @Test
  public void tstMemoized() {
    Array<String> of = Array.of("1", "2", "3", "1");
    System.out.println(of.distinct());
    Function0<Double> hashCache =
        Function0.of(Math::random).memoized();
  }
}
