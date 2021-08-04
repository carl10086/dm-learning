package com.ysz.biz.piper;

import java.util.function.Predicate;
import lombok.Data;

@Data
public class FullPiperCfg {


  public Predicate<Void> piperOnTimeLimit = new Predicate<Void>() {
    @Override
    public boolean test(final Void unused) {
      final long l = System.currentTimeMillis();
      /*判断当前时间*/
      return true;
    }
  };

  private FullPiperReadCfg readCfg;
  private FullPiperWriteCfg writeCfg;

}
