package com.ysz.dm.netty.custom.sofa.bolt.remoting.cfg;

import java.util.concurrent.ConcurrentHashMap;

/**
 *  options 的容器
 */
public class MyBoltOptionsCfg {

  private ConcurrentHashMap<MyBoltOption<?>, Object> options;

  public MyBoltOptionsCfg(final int expectedSize) {
    this.options = new ConcurrentHashMap<>(expectedSize);
  }


  public <T> MyBoltOptionsCfg updateOption(
      MyBoltOption<T> option, T value
  ) {
    if (value == null) {
      this.options.remove(option);
    } else {
      options.put(option, value);
    }

    return this;
  }

  @SuppressWarnings("unchecked")
  public <T> T getOption(
      MyBoltOption<T> option
  ) {
    final Object val = options.getOrDefault(option, option.getDefaultValue());
    return (T) val;
  }

}
