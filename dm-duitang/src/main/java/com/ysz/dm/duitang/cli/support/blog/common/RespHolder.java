package com.ysz.dm.duitang.cli.support.blog.common;

import java.io.Serializable;
import lombok.Getter;
import lombok.ToString;

@Getter
@ToString
public class RespHolder<T> implements Serializable {

  /**
   * 0 表示没有错误
   */
  private int errorNo;

  private T data;

  private RespHolder() {
  }

  public static <T> RespHolder<T> ok(T data) {
    RespHolder<T> res = new RespHolder<>();
    res.errorNo = ErrorNoEnum.OK.code;
    res.data = data;
    return res;
  }


  public boolean success() {
    return this.errorNo == ErrorNoEnum.OK.code;
  }


  private enum ErrorNoEnum {
    OK(1);
    private final int code;

    ErrorNoEnum(final int code) {
      this.code = code;
    }
  }
}
