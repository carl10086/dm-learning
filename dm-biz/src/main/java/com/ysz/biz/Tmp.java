package com.ysz.biz;

import java.math.BigDecimal;
import java.math.RoundingMode;

public class Tmp {


  public static void cal(int originInGb, int actual, int cpus) {

    int originInKB = originInGb * (1024 * 1024);
    int steal = originInKB - actual;
    System.out.printf("场景: %sg, vcpus:%s, stealInKb:%sKb, stealPerCpu:%s, stealInGb:%sGb stealPercent:%s%%\n",
                      originInGb,
                      cpus,
                      steal,
                      steal / cpus,
                      new BigDecimal(steal).divide(new BigDecimal(1024 * 1024), 2, RoundingMode.CEILING).floatValue(),
                      100f * new BigDecimal(steal).divide(new BigDecimal(actual), 4, RoundingMode.CEILING).floatValue()
    );
  }


  public static void main(String[] args) throws Exception {
    String src = "data:_;;;:;base64_______%2CPHNDcklwdCA%2BcHJvbXB0KDk2ODgpPCAvU2NSaXBUPg== HTTP/1.";


  }

  private static void jdMemSteal() {
    cal(4, 3880536, 1);
    cal(8, 8009156, 2);
    cal(16, 16266668, 2);
    cal(16, 16266336, 4);
    cal(32, 32779776, 8);
    cal(64, 65806636, 16);
    cal(128, 131860300, 32);
  }

}
