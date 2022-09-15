package com.ysz.dm.ddd.vshop.domain.core.common.rmb;

import java.math.BigDecimal;
import lombok.Getter;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-09-15 12:12 PM
 **/
@ToString
@Getter
public class Rmb implements Comparable<Rmb> {

  private final Long fen;

  private Rmb(long fen) {
    this.fen = fen;
  }


  public static Rmb ofYuan(long yuans) {
    return new Rmb(new BigDecimal(yuans).multiply(new BigDecimal(100)).longValue());
  }

  @Override
  public int compareTo(Rmb o) {
    return this.fen.compareTo(o.fen);
  }
}
