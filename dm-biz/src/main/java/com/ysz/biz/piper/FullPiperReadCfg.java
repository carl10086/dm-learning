package com.ysz.biz.piper;

import lombok.Data;

@Data
public class FullPiperReadCfg {

  /**
   * <pre>
   *   读取的线程池数目
   * </pre>
   */
  private int threadNum = 16;


}
