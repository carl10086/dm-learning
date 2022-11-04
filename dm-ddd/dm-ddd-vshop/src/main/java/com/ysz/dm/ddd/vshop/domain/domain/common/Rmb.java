package com.ysz.dm.ddd.vshop.domain.domain.common;

import java.math.BigDecimal;
import lombok.Getter;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-10-25 4:07 PM
 **/
@ToString
@Getter
public class Rmb {

  private static final BigDecimal YUAN_TO_FEN = new BigDecimal(100L);

  private final BigDecimal fen;

  public Rmb(BigDecimal fen) {
    this.fen = fen;
  }

  public static Rmb ofYuans(long yuans) {
    return new Rmb(new BigDecimal(yuans).multiply(YUAN_TO_FEN));
  }

  public static Rmb ofFens(long fens) {
    return new Rmb(new BigDecimal(fens));
  }
}
