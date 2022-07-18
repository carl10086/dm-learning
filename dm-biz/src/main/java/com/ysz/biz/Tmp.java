package com.ysz.biz;

import java.io.File;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.charset.Charset;
import java.util.List;
import org.apache.commons.io.FileUtils;

public class Tmp {


  public static void cal(int originInGb, int actual, int cpus) {

    int originInKB = originInGb * (1024 * 1024);
    int steal = originInKB - actual;
    System.out
        .printf("场景: %sg, vcpus:%s, stealInKb:%sKb, stealPerCpu:%s, stealInGb:%sGb stealPercent:%s%%\n",
                originInGb,
                cpus,
                steal,
                steal / cpus,
                new BigDecimal(steal).divide(new BigDecimal(1024 * 1024), 2, RoundingMode.CEILING).floatValue(),
                100f * new BigDecimal(steal).divide(new BigDecimal(actual), 4, RoundingMode.CEILING).floatValue()
        );
  }


  public static void main(String[] args) throws Exception {

    final List<String> lines = FileUtils.readLines(new File("/Users/carl/work/dt/tmp/nature.txt"),
                                                   Charset.defaultCharset()
    );



  }

  private static void jdMemSteal() {l
    cal(4, 3880536, 1);
    cal(8, 8009156, 2);
    cal(16, 16266668, 2);
    cal(16, 16266336, 4);
    cal(32, 32779776, 8);
    cal(64, 65806636, 16);
    cal(128, 131860300, 32);
  }

}
