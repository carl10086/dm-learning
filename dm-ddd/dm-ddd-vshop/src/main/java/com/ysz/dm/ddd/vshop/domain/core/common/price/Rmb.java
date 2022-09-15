package com.ysz.dm.ddd.vshop.domain.core.common.price;

import com.google.common.base.Preconditions;
import java.math.BigDecimal;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/9/9
 **/
public class Rmb {

  /**
   * 单位分
   */
  private final BigDecimal fen;

  public Rmb(BigDecimal fen) {
    this.fen = fen;
  }


  public static Rmb ofFens(long fens) {
    Preconditions.checkArgument(fens >= 0L);
    return new Rmb(new BigDecimal(fens));
  }


  public static Rmb ofYuans(long yuans) {
    Preconditions.checkArgument(yuans >= 0L);
    return new Rmb(new BigDecimal(yuans).multiply(new BigDecimal(100L)));
  }
}
