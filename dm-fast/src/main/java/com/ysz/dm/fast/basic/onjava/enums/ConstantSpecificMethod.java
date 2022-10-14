package com.ysz.dm.fast.basic.onjava.enums;

import java.text.DateFormat;
import java.util.Date;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/9/26
 **/
public enum ConstantSpecificMethod {
  DATE_TIME {
    @Override
    String getInfo() {
      return DateFormat.getDateInstance()
          .format(new Date());
    }
  },
  CLASSPATH {
    @Override
    String getInfo() {
      return System.getenv("CLASSPATH");
    }
  },
  VERSION {
    @Override
    String getInfo() {
      return System.getProperty("java.version");
    }
  };;

  abstract String getInfo();

  public static void main(String[] args) {
    for (ConstantSpecificMethod csm : values()) {
      System.out.println(csm.getInfo());
    }
  }
}
