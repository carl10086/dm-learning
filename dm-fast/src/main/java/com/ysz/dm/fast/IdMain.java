package com.ysz.dm.fast;

import cn.hutool.core.util.IdUtil;

/**
 * @author carl.yu
 * @date 2020/3/17
 */
public class IdMain {

  private static final String DT_APP_ENV = "DT_APP_ENV";

  private static void tstTimed() {
  }

  public static void main(String[] args) {
    tstTimed();
    String dt_app_env = System.getenv("DT_APP_ENV");
    System.out.println(dt_app_env);
    System.out.println(IdUtil.objectId());
  }


}
