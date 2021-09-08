package com.ysz.dm.netty.custom.sofa.bolt.remoting.cfg;

import com.google.common.base.Preconditions;
import lombok.Getter;
import lombok.ToString;

/**
 * <pre>
 *   封装一个具体的 配置项
 * </pre>
 * @param <T> 配置项对应的泛型
 */
@ToString
@Getter
public final class MyBoltOption<T> {

  private final String name;
  private T defaultValue;


  private MyBoltOption(final String name, final T defaultValue) {
    Preconditions.checkNotNull(name);
    this.name = name;
    this.defaultValue = defaultValue;
  }


  public static <T> MyBoltOption<T> valueOf(String name) {
    return new MyBoltOption<T>(name, null);
  }

  public static <T> MyBoltOption<T> valueOf(String name, T defaultValue) {
    return new MyBoltOption<T>(name, defaultValue);
  }

  @Override
  public boolean equals(final Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }

    final MyBoltOption<?> that = (MyBoltOption<?>) o;

    return name.equals(that.name);
  }

  @Override
  public int hashCode() {
    return name.hashCode();
  }
}
