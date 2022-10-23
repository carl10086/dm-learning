package com.ysz.arm.blog.server.config;

import com.linecorp.armeria.spring.ArmeriaSettings;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/10/23
 **/
public class ArmeriaDebugBean {
  private ArmeriaSettings settings;

  public ArmeriaDebugBean() {
  }

  public ArmeriaDebugBean setSettings(ArmeriaSettings settings) {
    this.settings = settings;
    return this;
  }
}
